#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import numpy as np
import os
from collections import deque
from multiprocessing import Process, Queue
import threading
import message_filters
# from worker_cuda_m1 import worker
from worker_cuda_m2 import worker
from core.ObjectTracker_core import ObjectTracker
import tf_conversions
import tf2_ros
import cv2
import math  # ✅ for radians/cos/sin

# === 物件高度（公尺），用於 position 的 z = height / 2.0 ===
OBJECT_HEIGHTS = {
    "Buoy": 1.0,
    "GuardBoat": 3.0,
    "RedBall": 3.0,
    "YellowBall": 3.0,
    "GreenBall": 3.0,
}
DEFAULT_HEIGHT = 1.0


# =========================
# 工具：縮放相關（factor）
# =========================

def scale_homography(H, factor):
    """將在舊解析度下求得的 Homography H 調整到新解析度。
    - 3x3: S @ H @ S_inv
    - 2x3: 僅平移量乘上 factor
    """
    if H is None:
        return None
    if H.shape == (3, 3):
        S = np.array([[factor, 0, 0],
                      [0, factor, 0],
                      [0, 0, 1]])
        return S @ H @ np.linalg.inv(S)
    elif H.shape == (2, 3):
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
        int(round(h * factor)),
    )


def scale_focals(focals, factor):
    if focals is None:
        return None
    if isinstance(focals, (int, float)):
        return float(focals) * factor
    return [float(f) * factor for f in focals]


# =========================
# 其他工具
# =========================

def transform_position(tf_buffer, position, source_frame, target_frame):
    try:
        tf = tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(0.1))
        trans = tf.transform.translation
        rot = tf.transform.rotation
        T = tf_conversions.transformations.quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
        T[0, 3] = trans.x
        T[1, 3] = trans.y
        T[2, 3] = trans.z

        point = np.array([position[0], position[1], position[2], 1.0])
        transformed = T @ point
        return transformed[:3].tolist()
    except Exception as e:
        rospy.logwarn(f"[Transform] Failed to transform from {source_frame} to {target_frame}: {e}")
        return position  # fallback: 原始位置


def load_matrix_file(file_path):
    if file_path and os.path.isfile(file_path):
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


# =========================
# 主節點
# =========================
class PanoStitcherNodeMP:
    def __init__(self, num_workers=3):
        self.drop_count = 0
        self.drop_every_n = 2
        self.drop_frame_mod = 0
        self.last_sent_frame_number = -1
        self.queue_previously_full = False
        self.tracker = ObjectTracker()  # ✅ 追蹤器現在平滑 distance/angle
        self.previous_ids = set()

        rospy.init_node("cuda_orb_pano_stitcher_mp")
        p = rospy.get_param

        # ============ 讀參數 ============
        left_topic = p("~left_topic", "/camera1_fix/color/image_raw/compressed")
        mid_topic = p("~mid_topic", "/camera2_fix/color/image_raw/compressed")
        right_topic = p("~right_topic", "/camera3_fix/color/image_raw/compressed")
        pano_mask_topic = p("~output_mask_topic", "/camera_pano_masked/image_raw/compressed")

        focals_raw = p("~focals", [260, 260, 260])
        crop_rect_raw = p("~crop_rect", None)

        marker_radius = int(p("~marker_radius", 2))
        marker_thickness = int(p("~marker_thickness", 2))
        marker_font_scale = float(p("~marker_font_scale", 0.4))
        camera_angle = eval(p("~camera_angle", "[-60,0,60]"))
        angle_colors = eval(p("~angle_colors", "[(255,0,255),(0,255,255),(255,255,0)]"))

        resize_factor = float(p("~horizon_resize_factor", 1.0))
        roi_ratio = eval(p("~roi_ratio", "[0.3,0.7,0.3,0.7]"))

        gl_L_file = p("~global_homo_left_file", None)
        gl_R_file = p("~global_homo_right_file", None)
        per_dir_L = p("~per_homo_left_dir", None)
        per_dir_R = p("~per_homo_right_dir", None)
        save_dir = p("~save_pano_dir", None)

        blend_w = int(p("~blend_width", 30))  # 🔔 使用者指定：混合區寬度不用乘 factor
        orb_feats = int(p("~orb_max_features", 2000))

        detr_model_path = p("~detr_model_path", "models/rtdetr-l.pt")
        segformer_model_path = p("~seg_model_path", "models/Segformer/segformer_model")
        conf_thres = p("~conf", 0.5)
        imgsz = p("~imgsz", 640)
        left_bound_angle = p("left_bound_angle", -105)
        right_bound_angle = p("right_bound_angle", 105)

        focal_length_px_raw = p("~focal_length_px", 200)  # horizon/距離推估用（像素）

        # 參數：factor（解析度縮放比例）
        factor = float(p("~scale_factor", 1.0))  # 預設 1.0，不縮放

        # ============ 解析 + 套用 factor ============
        focals = parse_focals(focals_raw)
        crop_rect = parse_crop_rect(crop_rect_raw)

        scaled_focals = scale_focals(focals, factor)
        scaled_crop_rect = scale_crop_rect(crop_rect, factor)

        # global homography 也要縮放（若有）
        self.global_M_L = scale_homography(load_matrix_file(gl_L_file), factor)
        self.global_M_R = scale_homography(load_matrix_file(gl_R_file), factor)

        # focal_length_px（像素）同樣屬於像素標度，應跟隨解析度縮放
        focal_length_px = float(focal_length_px_raw) * factor

        # 其他參數紀錄
        self.per_dir_L = per_dir_L
        self.per_dir_R = per_dir_R
        self.save_dir = save_dir

        # ============ 建立傳給 worker 的設定 ============
        self.worker_params = {
            'stitcher': {
                'focals': scaled_focals,
                'blend_width': int(blend_w * factor),
                'matcher_cfg': {"max_features": orb_feats},
                'crop_rect': scaled_crop_rect,
            },
            'traditional_horizon': {
                'resize_factor': resize_factor,
                'roi_ratio': roi_ratio,
            },
            'segformer': {
                'model_path': segformer_model_path,
                'roi_ratio': roi_ratio,
            },
            'marker': {
                'radius': marker_radius,
                'thickness': marker_thickness,
                'crop_rect': scaled_crop_rect,
                'font_scale': marker_font_scale,
                'camera_angle': camera_angle,
                'angle_colors': angle_colors,
            },
            'rtdetr': {
                'model_path': detr_model_path,
                'conf': conf_thres,
                'imgsz': imgsz,
            },
            'mapper': {
                'crop_rect': scaled_crop_rect,
                'focals': scaled_focals,
            },
            'distance_estimator': {
                'focal_length_px': focal_length_px,
            },
            'angle_estimator': {
                'crop_rect': scaled_crop_rect,
                'camera_angle': camera_angle,
                'left_bound_angle': left_bound_angle,
                'right_bound_angle': right_bound_angle,
            },
        }

        # ============ 其他初始化 ============
        self.bridge = CvBridge()
        self.pub_mask = rospy.Publisher(pano_mask_topic, CompressedImage, queue_size=1)
        self.marker_pub = rospy.Publisher("/visualization_marker_array", MarkerArray, queue_size=1)

        self.latest_msgs = deque(maxlen=1)
        self.latest_lock = threading.Lock()
        self.input_q = Queue(maxsize=1)
        self.output_q = Queue(maxsize=8)
        self.idx = 0
        self.frame_number = 0

        self.source_frame = rospy.get_param("~source_frame", "js/front_lidar_link")
        self.target_frame = rospy.get_param("~target_frame", "map")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.workers = []
        for i in range(num_workers):
            p_ = Process(target=worker, args=(self.input_q, self.output_q, self.worker_params))
            p_.daemon = True
            p_.start()
            self.workers.append(p_)

        subL = message_filters.Subscriber(left_topic, CompressedImage)
        subM = message_filters.Subscriber(mid_topic, CompressedImage)
        subR = message_filters.Subscriber(right_topic, CompressedImage)
        ats = message_filters.ApproximateTimeSynchronizer([subL, subM, subR], queue_size=6, slop=0.2, reset=True)
        ats.registerCallback(self.store_latest)

        self.t_submit = rospy.Timer(rospy.Duration(1.0/20.0), self.submit_job)
        self.t_pub = rospy.Timer(rospy.Duration(1.0/20.0), self.publish_result)
        rospy.on_shutdown(self.shutdown)
        rospy.loginfo("[Node] Multiprocess CUDA pano stitcher ready (with factor).")
        rospy.spin()

    # =========================
    # Callbacks & Publishing
    # =========================
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
            if self.frame_number == self.last_sent_frame_number:
                return  # 避免送出同一張

            job = {
                'idx': self.idx,
                'imgs_raw': [msgL.data, msgM.data, msgR.data],
                'global_M_L': self.global_M_L,
                'global_M_R': self.global_M_R,
            }

            if self.input_q.full():
                if not self.queue_previously_full:
                    self.queue_previously_full = True

                if self.drop_frame_mod == 0:
                    self.drop_count += 1
                else:
                    self.input_q.put(job)
                    self.idx += 1
                    self.last_sent_frame_number = self.frame_number

                self.drop_frame_mod = (self.drop_frame_mod + 1) % self.drop_every_n

            else:
                if self.queue_previously_full:
                    self.queue_previously_full = False

                self.input_q.put(job)
                self.idx += 1
                self.last_sent_frame_number = self.frame_number
                self.drop_frame_mod = 0

            self.frame_number += 1

        except Exception as e:
            rospy.logwarn(f"[Node] Failed to enqueue job: {e}")

    def publish_result(self, event):
        while not self.output_q.empty():
            result = self.output_q.get()
            raw_img = result.get('raw_img')
            marker_data = result.get('marker_data')  # [{'class','distance','angle','bbox','scale','color'}, ...]

            img_to_pub = None

            # ✅ 直接把 marker_data 丟給追蹤器：內部會濾距離/角度
            tracked_objects = None
            if marker_data is not None:
                tracked_objects = self.tracker.assign_track_ids(marker_data)

            if raw_img is not None and tracked_objects is not None:
                img = raw_img.copy()

                tracked_objects = sorted(tracked_objects, key=lambda x: x.get("track_id", -1))
                x0, y0 = 10, 20
                line_space = 20

                # === 由「已平滑」的 distance/angle 轉 pose，用你指定公式 ===
                for i, obj in enumerate(tracked_objects):
                    cls    = obj.get("class", "")
                    dist   = float(obj.get("distance", 0.0))
                    ang_deg= float(obj.get("angle", 0.0))
                    ang_rad= math.radians(ang_deg)

                    # 1) 位置
                    x = dist * math.cos(ang_rad)
                    y = -1.0 * dist * math.sin(ang_rad)
                    z = (OBJECT_HEIGHTS.get(cls, DEFAULT_HEIGHT)) * 0.5
                    obj["position"] = (x, y, z)

                    # 2) 朝向（yaw + π）
                    yaw = ang_rad + math.pi
                    qx, qy, qz, qw = tf_conversions.transformations.quaternion_from_euler(0.0, 0.0, yaw)
                    obj["orientation"] = (qx, qy, qz, qw)

                    # 左上角資訊
                    tid = obj.get("track_id", -1)
                    line = f"ID:{tid}  {cls} {dist:.1f}m/{ang_deg:.1f}deg"
                    y_text = y0 + i * line_space
                    cv2.putText(img, line, (x0, y_text),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

                img_to_pub = img

            # 影像發布
            if img_to_pub is not None:
                ret, jpeg_mask = cv2.imencode('.jpg', img_to_pub, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                if ret:
                    msg = CompressedImage()
                    msg.header.stamp = rospy.Time.now()
                    msg.header.frame_id = self.target_frame
                    msg.format = "jpeg"
                    msg.data = jpeg_mask.tobytes()
                    self.pub_mask.publish(msg)

            # RViz Marker（已含 position/orientation）
            if tracked_objects is not None:
                marker_array = MarkerArray()
                now = rospy.Time.now()
                current_ids = set()

                for obj in tracked_objects:
                    marker = Marker()
                    marker.header.stamp = now
                    marker.header.frame_id = self.target_frame
                    marker.ns = obj["class"]
                    marker.id = obj["track_id"]
                    marker.type = Marker.CUBE
                    marker.action = Marker.ADD

                    # 轉到 target_frame
                    x, y, z = obj["position"]
                    new_pos = transform_position(
                        self.tf_buffer, [x, y, z], self.source_frame, self.target_frame
                    )
                    marker.pose.position.x = new_pos[0]
                    marker.pose.position.y = new_pos[1]
                    marker.pose.position.z = new_pos[2]

                    qx, qy, qz, qw = obj["orientation"]
                    marker.pose.orientation.x = qx
                    marker.pose.orientation.y = qy
                    marker.pose.orientation.z = qz
                    marker.pose.orientation.w = qw

                    size_factor = 3.0
                    sx, sy, sz = obj.get("scale", (1.0, 1.0, OBJECT_HEIGHTS.get(obj["class"], DEFAULT_HEIGHT)))
                    marker.scale.x = sx * size_factor
                    marker.scale.y = sy * size_factor
                    marker.scale.z = sz * size_factor

                    if "color" in obj:
                        r, g, b = obj["color"]
                        marker.color = ColorRGBA(r=r, g=g, b=b, a=0.8)
                    else:
                        marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.8)

                    marker.lifetime = rospy.Duration(0.2)
                    marker_array.markers.append(marker)
                    current_ids.add(obj["track_id"])

                # 清除消失的 ID
                deleted_ids = self.previous_ids - current_ids
                for track_id in deleted_ids:
                    delete_marker = Marker()
                    delete_marker.header.stamp = now
                    delete_marker.header.frame_id = self.target_frame
                    delete_marker.ns = "delete"
                    delete_marker.id = track_id
                    delete_marker.action = Marker.DELETE
                    marker_array.markers.append(delete_marker)

                self.previous_ids = current_ids
                self.marker_pub.publish(marker_array)

            elif self.previous_ids:
                # 沒有資料則清除舊 marker
                now = rospy.Time.now()
                delete_array = MarkerArray()
                for track_id in self.previous_ids:
                    delete_marker = Marker()
                    delete_marker.header.stamp = now
                    delete_marker.header.frame_id = self.target_frame
                    delete_marker.ns = "delete"
                    delete_marker.id = track_id
                    delete_marker.action = Marker.DELETE
                    delete_array.markers.append(delete_marker)
                self.previous_ids.clear()
                self.marker_pub.publish(delete_array)

    # =========================
    # Shutdown
    # =========================
    def shutdown(self):
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
    try:
        node = PanoStitcherNodeMP(num_workers=3)
    except rospy.ROSInterruptException:
        pass