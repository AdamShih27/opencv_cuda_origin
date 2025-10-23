#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import numpy as np
import threading
import os
from collections import deque
from multiprocessing import Process, Queue
import message_filters
from nvjpeg import NvJpeg

from worker_cuda import worker  # 請確定 worker_cuda.py 同目錄

import time

def log_with_time(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

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

def scale_homography(H, factor):
    if H is None:
        return None
    if H.shape == (3, 3):
        S = np.array([[factor, 0, 0],
                      [0, factor, 0],
                      [0, 0, 1]])
        return S @ H @ np.linalg.inv(S)
    elif H.shape == (2, 3):
        # 仿射，只平移向量要縮放
        A = H[:, :2]
        t = H[:, 2] * factor
        return np.hstack([A, t.reshape(2, 1)])
    else:
        raise ValueError(f"Unsupported H shape: {H.shape}")

def scale_crop_rect(crop_rect, factor):
    if crop_rect is None:
        return None
    x, y, w, h = crop_rect
    return (
        int(round(x * factor)),
        int(round(y * factor)),
        int(round(w * factor)),
        int(round(h * factor))
    )

def scale_focals(focals, factor):
    if focals is None:
        return None
    if isinstance(focals, (int, float)):
        return focals * factor
    return [float(f) * factor for f in focals]

class PanoStitcherNodeMP:
    def __init__(self, num_workers=3):
        rospy.init_node("cuda_orb_pano_stitcher_mp")
        p = rospy.get_param

        # --- ROS Params ---
        left_topic   = p("~left_topic",  "/camera1_fix/color/image_raw/compressed")
        mid_topic    = p("~mid_topic",   "/camera2_fix/color/image_raw/compressed")
        right_topic  = p("~right_topic", "/camera3_fix/color/image_raw/compressed")
        output_topic = p("~output_topic","/camera_pano_stitched/image_raw/compressed")
        focals = parse_focals(p("~focals", [260,260,260]))
        crop_rect = parse_crop_rect(p("~crop_rect", None))

        gl_L_file = p("~global_homo_left_file",  "$(find cylindrical_processing)/src/cylindrical_homo/global_H1/H1_000008.npy")
        gl_R_file = p("~global_homo_right_file", "$(find cylindrical_processing)/src/cylindrical_homo/global_H2/H2_000012.npy")
        per_dir_L = p("~per_homo_left_dir",     "$(find cylindrical_processing)/src/cylindrical_homo/per_H1")
        per_dir_R = p("~per_homo_right_dir",    "$(find cylindrical_processing)/src/cylindrical_homo/per_H2")

        save_dir   = p("~save_pano_dir",     "$(find cylindrical_processing)/src/cylindrical_homo/result")
        jpeg_q     = int(p("~jpeg_quality",      90))
        blend_w    = int(p("~blend_width",       30))
        orb_feats  = int(p("~orb_max_features", 2000))

        factor = 1  # 依需求更改，或寫死 ex: 4.05

        self.global_M_L = scale_homography(load_matrix_file(gl_L_file), factor)
        self.global_M_R = scale_homography(load_matrix_file(gl_R_file), factor)
        focals = scale_focals(focals, factor)
        crop_rect = scale_crop_rect(crop_rect, factor)

        self.per_dir_L = per_dir_L
        self.per_dir_R = per_dir_R
        self.save_dir  = save_dir
        self.jpeg_q    = jpeg_q

        # CUDA worker params
        self.stitcher_params = {
            'focals': focals,
            'blend_width': int(blend_w * factor),
            'matcher_cfg': {"max_features": orb_feats},
            'crop_rect': crop_rect,
        }

        self.bridge = CvBridge()
        self.pub    = rospy.Publisher(output_topic, CompressedImage, queue_size=1)
        self.frame_id = 0

        self.latest_msgs = deque(maxlen=1)
        self.latest_lock = threading.Lock()

        self.input_q  = Queue(maxsize=1)
        self.output_q = Queue(maxsize=8)
        self.idx = 0

        # 啟動 worker process
        self.workers = []
        for i in range(num_workers):
            p = Process(target=worker, args=(self.input_q, self.output_q, self.stitcher_params), name=f"cuda_worker_{i}")
            p.daemon = True
            p.start()
            self.workers.append(p)

        # ROS 訂閱
        subL = message_filters.Subscriber(left_topic,  CompressedImage)
        subM = message_filters.Subscriber(mid_topic,   CompressedImage)
        subR = message_filters.Subscriber(right_topic, CompressedImage)
        ats  = message_filters.ApproximateTimeSynchronizer([subL, subM, subR], queue_size=6, slop=0.2)
        ats.registerCallback(self.store_latest)

        self.t_submit = rospy.Timer(rospy.Duration(1.0/20.0), self.submit_job)
        self.t_pub    = rospy.Timer(rospy.Duration(1.0/20.0), self.publish_result)

        self.shutdown_called = False
        rospy.on_shutdown(self.shutdown)

        rospy.loginfo("[Node] Multiprocess CUDA pano stitcher ready.")
        rospy.spin()

    def store_latest(self, msgL, msgM, msgR):
        with self.latest_lock:
            self.latest_msgs.clear()
            self.latest_msgs.append((msgL, msgM, msgR))

    def submit_job(self, event):
        with self.latest_lock:
            if not self.latest_msgs:
                return
            msgL, msgM, msgR = self.latest_msgs.popleft()
        try:
            imgs = [m.data for m in (msgL, msgM, msgR)]
        except Exception as e:
            # rospy.logwarn(f"[Node] Read bytes failed: {e}")
            return
        job = {
            'idx': self.idx,
            'imgs': imgs,
            'global_M_L': self.global_M_L,
            'global_M_R': self.global_M_R,
            'frame_id': self.frame_id
        }
        if not self.input_q.full():
            # log_with_time(f"submit_job: put job {self.idx} (frame {self.frame_id}) to input_q")
            self.input_q.put(job)
            self.idx += 1
            self.frame_id += 1
        else:
            log_with_time(f"submit_job: input_q FULL! drop frame {self.frame_id}")

    def publish_result(self, event):
        # log_with_time(f"publish_result: output_q size={self.output_q.qsize()}")
        while not self.output_q.empty():
            result = self.output_q.get()
            # log_with_time(f"publish_result: got result frame {result.get('frame_id','?')}")
            jpeg_bytes = result['result']
            frame_id = result['frame_id']
            new_M_L = result['new_M_L']
            new_M_R = result['new_M_R']
            if jpeg_bytes is None:
                # rospy.logwarn(f"[Node] Pano failed at frame {frame_id}: {result.get('err','')}")
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

            out = CompressedImage()
            out.header.stamp    = rospy.Time.now()
            out.header.frame_id = "pano"
            out.format          = "jpeg"
            out.data            = jpeg_bytes
            self.pub.publish(out)

            if self.save_dir and (saved_L or saved_R):
                os.makedirs(self.save_dir, exist_ok=True)
                # 若要本地存 png 或 jpg，可以用 OpenCV 再解開，但可直接存 bytes 為 jpg
                with open(os.path.join(self.save_dir, f"pano_{frame_id:06d}.jpg"), "wb") as f:
                    f.write(jpeg_bytes)

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
