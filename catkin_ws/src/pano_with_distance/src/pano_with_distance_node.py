#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import numpy as np
import cv2
from collections import deque
from multiprocessing import Process, Queue
import threading
import os
import message_filters
from geometry_msgs.msg import PolygonStamped, Point32

from worker_cuda import worker  # worker會輸出 pano, pano_mask, frame_id ...

def load_matrix_file(file_path):
    if os.path.isfile(file_path):
        try:
            M = np.load(file_path)
            rospy.loginfo(f"[Node] Loaded global matrix: {file_path}")
            return M
        except Exception as e:
            rospy.logwarn(f"[Node] Failed to load global matrix {file_path}: {e}")
    return None

def parse_focals(raw):
    if isinstance(raw, str):
        raw = raw.strip('[] ')
        vals = [float(x) for x in raw.split(',') if x.strip()]
    elif isinstance(raw, (list, tuple)):
        vals = [float(x) for x in raw]
    else:
        vals = [float(raw)]
    if len(vals) == 1:
        vals = vals * 3
    return vals[:3]

def decode_and_resize(data, resize_w):
    arr = np.frombuffer(data, np.uint8)
    im  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    h, w = im.shape[:2]
    return cv2.resize(im, (resize_w, int(h*resize_w/w)))

def parse_crop_rect(raw):
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = raw.strip("[] ")
        vals = [int(x) for x in raw.split(',') if x.strip()]
    elif isinstance(raw, (list, tuple)):
        vals = [int(x) for x in raw]
    else:
        vals = [int(raw)]
    if len(vals) != 4:
        raise ValueError("crop_rect 必須是4個整數 [x, y, w, h]")
    return tuple(vals)

class PanoStitcherNodeMP:
    def __init__(self, num_workers=3):
        rospy.init_node("cuda_orb_pano_stitcher_mp")
        # --- ROS Params ---
        p = rospy.get_param  # 假設你已經 import rospy 且用 p 取代 get_param

        left_topic   = p("~left_topic",  "/camera1_fix/color/image_raw/compressed")
        mid_topic    = p("~mid_topic",   "/camera2_fix/color/image_raw/compressed")
        right_topic  = p("~right_topic", "/camera3_fix/color/image_raw/compressed")
        output_topic = p("~output_topic", "/camera_pano_stitched/image_raw/compressed")
        pano_mask_topic = p("~output_mask_topic", "/camera_pano_masked/image_raw/compressed")
        resize_w     = int(p("~resize_w", 640))
        focals = parse_focals(p("~focals", [260,260,260]))
        crop_rect = parse_crop_rect(p("~crop_rect", None))

        # -- mask core/marker params --
        marker_radius     = int(p("~marker_radius", 2))
        marker_thickness  = int(p("~marker_thickness", 2))
        marker_font_scale = float(p("~marker_font_scale", 0.4))
        marker_text_dx    = int(p("~marker_text_dx", 4))
        marker_text_dy    = int(p("~marker_text_dy", 2))
        marker_colors     = eval(p("~marker_colors", "[(255,0,0),(0,255,0),(0,0,255)]"))
        camera_angle      = eval(p("~camera_angle", "[-60,0,60]"))
        angle_colors      = eval(p("~angle_colors", "[(255,0,255),(0,255,255),(255,255,0)]"))  # 預設與點顏色相同
        # -- horizon core/distance params --
        distance_mark_distances  = eval(p("~distance_mark_distances",  "[0,100,500,1000,2000,5000]"))
        distance_alpha_intervals = eval(p("~distance_alpha_intervals", "[2.5,2.5,2.5,2.5,2.5]"))
        distance_intervals       = eval(p("~distance_intervals",       "[0,100,500,1000,2000,5000]"))
        distance_camera_height   = float(p("~distance_camera_height",  3.0))
        distance_front_distance  = float(p("~distance_front_distance", 0.0))
        distance_D_horizon       = float(p("~distance_D_horizon",      5000))
        base_point    = eval(p("~base_point", "(320, 410)"))  # 支援 launch string
        resize_factor = float(p("~horizon_resize_factor", 1.0))
        roi_ratio     = eval(p("~roi_ratio", "[0.3,0.7,0.3,0.7]"))  # 用於裁切 pano 圖的比例

        # --- 其他參數 ---
        gl_L_file = p("~global_homo_left_file",  "$(find cylindrical_processing)/src/cylindrical_homo/global_H1/H1_000008.npy")
        gl_R_file = p("~global_homo_right_file", "$(find cylindrical_processing)/src/cylindrical_homo/global_H2/H2_000012.npy")
        per_dir_L = p("~per_homo_left_dir",      "$(find cylindrical_processing)/src/cylindrical_homo/per_H1")
        per_dir_R = p("~per_homo_right_dir",     "$(find cylindrical_processing)/src/cylindrical_homo/per_H2")
        save_dir  = p("~save_pano_dir",          "$(find cylindrical_processing)/src/cylindrical_homo/result")
        jpeg_q    = int(p("~jpeg_quality",      90))
        blend_w   = int(p("~blend_width",       30))
        orb_feats = int(p("~orb_max_features", 2000))

        self.global_M_L = load_matrix_file(gl_L_file)
        self.global_M_R = load_matrix_file(gl_R_file)
        self.per_dir_L = per_dir_L
        self.per_dir_R = per_dir_R
        self.save_dir  = save_dir
        self.jpeg_q    = jpeg_q

        # --- CUDA worker params ---
        self.stitcher_params = {
            'focals': focals,
            'blend_width': blend_w,
            'matcher_cfg': {"max_features": orb_feats},
            'crop_rect': crop_rect,
        }
        self.horizon_params = {
            'resize_factor': resize_factor,
            'base_point': base_point,
            'mark_distances': distance_mark_distances,
            'alpha_intervals': distance_alpha_intervals,
            'alpha_distance_intervals': distance_intervals,
            'camera_height': distance_camera_height,
            'front_distance': distance_front_distance,
            'D_horizon': distance_D_horizon,
            'roi_ratio': roi_ratio,
        }
        self.marker_params = {
            'radius': marker_radius,
            'thickness': marker_thickness,
            'crop_rect': crop_rect,
            'colors': marker_colors,
            'font_scale': marker_font_scale,
            'text_dx': marker_text_dx,
            'text_dy': marker_text_dy,
            'camera_angle': camera_angle,
            'D_horizon': distance_D_horizon,
            'angle_colors': angle_colors,  # 預設與點顏色相同
        }

        self.worker_params = {
            'stitcher': self.stitcher_params,
            'horizon':  self.horizon_params,
            'marker':   self.marker_params,
        }

        self.bridge = CvBridge()
        self.pub    = rospy.Publisher(output_topic, CompressedImage, queue_size=1)
        self.pub_mask = rospy.Publisher(pano_mask_topic, CompressedImage, queue_size=1)
        self.pub_horizon_pts = rospy.Publisher('/horizon_points_poly', PolygonStamped, queue_size=1)
        self.frame_id = 0

        # Latest buffer，只存最新 frame
        self.latest_msgs = deque(maxlen=1)
        self.latest_lock = threading.Lock()

        # multiprocessing queue
        self.input_q  = Queue(maxsize=1)
        self.output_q = Queue(maxsize=8)
        self.idx = 0

        # 啟動 worker process
        self.workers = []
        for i in range(num_workers):
            p = Process(
                target=worker,
                args=(self.input_q, self.output_q, self.worker_params),
                name=f"cuda_worker_{i}"
            )
            p.daemon = True
            p.start()
            self.workers.append(p)

        # ROS 訂閱
        subL = message_filters.Subscriber(left_topic,  CompressedImage)
        subM = message_filters.Subscriber(mid_topic,   CompressedImage)
        subR = message_filters.Subscriber(right_topic, CompressedImage)
        ats  = message_filters.ApproximateTimeSynchronizer([subL, subM, subR], queue_size=6, slop=0.2)
        ats.registerCallback(self.store_latest, resize_w)

        # 定時投遞/發佈
        self.t_submit = rospy.Timer(rospy.Duration(1.0/20.0), self.submit_job)
        self.t_pub    = rospy.Timer(rospy.Duration(1.0/20.0), self.publish_result)

        self.shutdown_called = False
        rospy.on_shutdown(self.shutdown)

        rospy.loginfo("[Node] Multiprocess CUDA pano stitcher ready.")
        rospy.spin()

    def store_latest(self, msgL, msgM, msgR, resize_w):
        with self.latest_lock:
            self.latest_msgs.clear()
            self.latest_msgs.append((msgL, msgM, msgR, resize_w))

    def submit_job(self, event):
        with self.latest_lock:
            if not self.latest_msgs:
                return
            msgL, msgM, msgR, resize_w = self.latest_msgs.popleft()
        try:
            imgs = [decode_and_resize(m.data, resize_w) for m in (msgL, msgM, msgR)]
        except Exception as e:
            rospy.logwarn(f"[Node] Decode/resize failed: {e}")
            return
        job = {
            'idx': self.idx,
            'imgs': imgs,
            'global_M_L': self.global_M_L,
            'global_M_R': self.global_M_R,
            'frame_id': self.frame_id
        }
        if not self.input_q.full():
            self.input_q.put(job)
            self.idx += 1
            self.frame_id += 1

    def publish_result(self, event):
        while not self.output_q.empty():
            result = self.output_q.get()
            pano = result['result']
            pano_mask = result.get('pano_mask')
            frame_id = result['frame_id']
            new_M_L = result['new_M_L']
            new_M_R = result['new_M_R']
            horizon_pts = result.get('horizon_pts', None)   # <<<< 新增
            if pano is None:
                rospy.logwarn(f"[Node] Pano failed at frame {frame_id}: {result.get('err','')}")
                continue

            per_key = f"{frame_id:06d}"
            saved_L = saved_R = False
            if self.global_M_L is None and new_M_L is not None:
                os.makedirs(self.per_dir_L, exist_ok=True)
                np.save(os.path.join(self.per_dir_L, f"H1_{per_key}.npy"), new_M_L)
                saved_L = True
            if self.global_M_R is None and new_M_R is not None:
                os.makedirs(self.per_dir_R, exist_ok=True)
                np.save(os.path.join(self.per_dir_R, f"H2_{per_key}.npy"), new_M_R)
                saved_R = True

            ok, buf = cv2.imencode('.jpg', pano, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_q])
            if ok:
                out = CompressedImage()
                out.header.stamp    = rospy.Time.now()
                out.header.frame_id = "pano"
                out.format          = "jpeg"
                out.data            = buf.tobytes()
                self.pub.publish(out)
            if pano_mask is not None:
                ok_mask, buf_mask = cv2.imencode('.jpg', pano_mask, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_q])
                if ok_mask:
                    out_mask = CompressedImage()
                    out_mask.header.stamp    = rospy.Time.now()
                    out_mask.header.frame_id = "pano_mask"
                    out_mask.format          = "jpeg"
                    out_mask.data            = buf_mask.tobytes()
                    self.pub_mask.publish(out_mask)
                    
            if horizon_pts is not None:
                msg = PolygonStamped()
                msg.header.stamp = rospy.Time.now()
                msg.header.frame_id = "pano"
                for cam_points in horizon_pts:
                    for (x, y, D) in cam_points:
                        msg.polygon.points.append(Point32(float(x), float(y), float(D)))
                self.pub_horizon_pts.publish(msg)

            if self.save_dir and (saved_L or saved_R):
                os.makedirs(self.save_dir, exist_ok=True)
                cv2.imwrite(os.path.join(self.save_dir, f"pano_{frame_id:06d}.jpg"), pano)
                if pano_mask is not None:
                    cv2.imwrite(os.path.join(self.save_dir, f"pano_mask_{frame_id:06d}.jpg"), pano_mask)

            if frame_id % 10 == 0:
                rospy.loginfo(f"[Node] {frame_id} frames processed")

    def shutdown(self):
        if getattr(self, "shutdown_called", False):
            return
        self.shutdown_called = True
        rospy.loginfo("Shutting down worker processes ...")
        for _ in self.workers:
            self.input_q.put(None)
        for p in self.workers:
            try:
                p.join(timeout=3)
            except Exception as e:
                rospy.logwarn(f"Failed to join process: {e}")
        rospy.loginfo("All workers exited.")

if __name__ == '__main__':
    node = None
    try:
        node = PanoStitcherNodeMP(num_workers=3)
    except rospy.ROSInterruptException:
        pass
    finally:
        if node:
            node.shutdown()
