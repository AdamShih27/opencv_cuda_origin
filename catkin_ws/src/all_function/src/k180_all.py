#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from threading import Thread
from nvjpeg import NvJpeg
import rospkg
import math
import tf.transformations as tf_trans

from core.rtdetr_predict_core import RTDETRPredictor
from core.bbox2distance_core import BBoxDistanceEstimator
from core.ObjectTracker_core import ObjectTracker
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

# === 類別高度 (m) 與色彩 (RGB 0~1) ===
OBJECT_HEIGHTS = {
    "Buoy": 1.0,
    "GuardBoat": 2.0,
    "RedBall": 3.0,
    "YellowBall": 3.0,
    "GreenBall": 3.0,
}
DEFAULT_HEIGHT = 1.0

CLASS_COLOR_MAP = {
    "Buoy": (1.0, 0.5, 0.0),
    "GuardBoat": (1.0, 0.0, 0.0),
    "RedBall": (1.0, 0.0, 0.0),
    "YellowBall": (1.0, 1.0, 0.0),
    "GreenBall": (0.0, 1.0, 0.0),
}

class RTSPCameraWithRTDETR:
    def __init__(self, name, rtsp_url, topic_name, model_path, conf_thres, imgsz, focal_length_px, 
                 crop_rect, left_bound_angle, right_bound_angle, marker_frame_id="odom"):
        self.name = name
        self.rtsp_url = rtsp_url
        self.topic_name = topic_name
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.imgsz = imgsz
        self.focal_length_px = focal_length_px

        self.crop_rect = crop_rect
        self.left_bound_angle = left_bound_angle
        self.right_bound_angle = right_bound_angle
        self.marker_frame_id = marker_frame_id

        self.publisher = rospy.Publisher(topic_name, CompressedImage, queue_size=1)
        self.marker_pub = rospy.Publisher(topic_name.replace("image_raw", "marker_array"), MarkerArray, queue_size=1)

        self.nvjpeg = NvJpeg()
        self.detector = RTDETRPredictor(model_path, conf_thres, imgsz)
        self.distance_estimator = BBoxDistanceEstimator(focal_length_px)

        # ✅ 使用你提供的新 ObjectTracker（含 KalmanTrack）
        #    - 內部會自動 unwrap angle、近/遠距調參與暖機
        self.tracker = ObjectTracker(auto_reset=True, empty_frames_to_reset=30)

        self.previous_alive = set()  # 保存 (ns, id) 做刪除
        self.pipeline = (
            f"rtspsrc location={rtsp_url} latency=0 protocols=udp drop-on-latency=true ! "
            "rtph265depay ! h265parse ! avdec_h265 ! videoconvert ! "
            "videoconvert n-threads=2 ! video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )

        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            rospy.logerr(f"[{self.name}] ❌ 無法開啟 RTSP 串流：{rtsp_url}")
            self.cap = None

        self.thread = Thread(target=self.stream_loop, name=f"{self.name}_stream")
        self.thread.daemon = True
        self.running = True
        self.last_time = rospy.Time.now()  # ✅ 用來估 dt

    def start(self):
        if self.cap:
            self.thread.start()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

    def _estimate_scale_and_color(self, cls, bbox):
        # 以 bbox 高度 + 類別高度推估 scale：(w,d,h)
        x1, y1, x2, y2 = bbox
        px_w = max(1.0, float(x2 - x1))
        px_h = max(1.0, float(y2 - y1))
        real_h = OBJECT_HEIGHTS.get(cls, DEFAULT_HEIGHT)
        ratio = real_h / px_h  # m / pixel
        real_w = px_w * ratio
        scale = (real_w, real_w, real_h)
        color = CLASS_COLOR_MAP.get(cls, (1.0, 1.0, 1.0))
        return scale, color

    def _crop_safe(self, img):
        x, y, w, h = self.crop_rect
        H, W = img.shape[:2]
        x2 = min(W, max(0, x + w))
        y2 = min(H, max(0, y + h))
        x1 = min(W, max(0, x))
        y1 = min(H, max(0, y))
        if x2 <= x1 or y2 <= y1:
            return img  # 不裁
        return img[y1:y2, x1:x2]

    def stream_loop(self):
        while not rospy.is_shutdown() and self.running:
            if self.cap is None:
                rospy.sleep(1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                rospy.logwarn_throttle(5.0, f"[{self.name}] ⚠️ 影像讀取失敗")
                rospy.sleep(0.05)
                continue

            # --- 裁切掉黑邊 ---
            frame = self._crop_safe(frame)

            try:
                boxes, classes, scores = self.detector.infer(frame)

                detections = []
                frame_h, frame_w = frame.shape[:2]
                for (box, cls, score) in zip(boxes, classes, scores):
                    det = {"bbox": box, "class": cls, "score": float(score)}

                    # 1) 估距（bbox 法）
                    try:
                        det = self.distance_estimator.estimate_distance(det)
                        det["distance"] = float(det.get("distance_bbox", 0.0))
                    except Exception as e:
                        rospy.logwarn_throttle(2.0, f"[{self.name}] 測距錯誤: {e}")
                        det["distance"] = 0.0

                    # 2) 算 angle（畫面寬度線性映射到 [left,right]）
                    x1, y1, x2, y2 = box
                    bbox_cx = 0.5 * (x1 + x2)
                    angle = (bbox_cx / max(1.0, frame_w)) * (self.right_bound_angle - self.left_bound_angle) + self.left_bound_angle
                    det["angle"] = float(angle)

                    # 3) scale / color
                    scale, color = self._estimate_scale_and_color(cls, box)
                    det["scale"] = scale
                    det["color"] = color

                    detections.append(det)

                # === 追蹤 & 濾波（distance/angle）===
                now = rospy.Time.now()
                dt = (now - self.last_time).to_sec()
                # ✅ 保底 dt，限制更新頻率範圍（配合你的 KalmanTrack.set_dt 限制亦可）
                if not (0.0 < dt < 1.0):
                    dt = 1.0 / 20.0
                self.last_time = now

                tracked = self.tracker.assign_track_ids(detections, dt=dt)  # ✅ 回傳已平滑 distance/angle

                # === 由已平滑的 distance/angle → pose ===
                marker_array = MarkerArray()
                alive_this_frame = set()
                draw_list = []

                for info in tracked:
                    cls = info["class"]
                    dist = float(info.get("distance", 0.0))
                    ang_deg = float(info.get("angle", 0.0))
                    ang_rad = math.radians(ang_deg)

                    # position（水平面上 x 前、y 左，z 高）
                    x = dist * math.cos(ang_rad)
                    y = -dist * math.sin(ang_rad)
                    z = OBJECT_HEIGHTS.get(cls, DEFAULT_HEIGHT) * 0.5

                    # orientation: yaw = angle + π（物體面向相機）
                    yaw = ang_rad + math.pi
                    qx, qy, qz, qw = tf_trans.quaternion_from_euler(0.0, 0.0, yaw)

                    # 畫 bbox & label 用
                    x1, y1, x2, y2 = map(int, info["bbox"])
                    color_bgr = tuple(int(255 * c) for c in info["color"])
                    tid = int(info.get("track_id", -1))
                    label = f"ID:{tid} {cls} {dist:.1f}m/{ang_deg:.1f}deg"
                    draw_list.append((x1, y1, x2, y2, color_bgr, label))

                    # RViz Marker
                    marker = Marker()
                    marker.header.stamp = now
                    marker.header.frame_id = self.marker_frame_id
                    marker.ns = cls                  # ✅ 用類別作 namespace
                    marker.id = tid                  # ✅ 用 track_id 當 id
                    marker.type = Marker.CUBE
                    marker.action = Marker.ADD

                    marker.pose.position.x = x
                    marker.pose.position.y = y
                    marker.pose.position.z = z
                    marker.pose.orientation.x = qx
                    marker.pose.orientation.y = qy
                    marker.pose.orientation.z = qz
                    marker.pose.orientation.w = qw

                    size_factor = 3.0
                    sx, sy, sz = info["scale"]
                    marker.scale.x = sx * size_factor
                    marker.scale.y = sy * size_factor
                    marker.scale.z = sz * size_factor

                    r, g, b = info["color"]
                    marker.color = ColorRGBA(r=r, g=g, b=b, a=0.85)

                    # 讓 RViz 若掉訊息不會殘留太久，但仍可平順顯示
                    marker.lifetime = rospy.Duration(0.3)
                    marker_array.markers.append(marker)

                    alive_this_frame.add((marker.ns, marker.id))

                # === 刪除不存在的 Marker（要用相同 ns+id 才會刪得乾淨）===
                to_delete = self.previous_alive - alive_this_frame
                for ns, mid in to_delete:
                    del_marker = Marker()
                    del_marker.header.stamp = now
                    del_marker.header.frame_id = self.marker_frame_id
                    del_marker.ns = ns
                    del_marker.id = mid
                    del_marker.action = Marker.DELETE
                    marker_array.markers.append(del_marker)
                self.previous_alive = alive_this_frame

                if marker_array.markers:
                    self.marker_pub.publish(marker_array)

                # --- 繪製 bbox/label 並發佈影像 ---
                for (x1, y1, x2, y2, color_bgr, label) in draw_list:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
                    cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 2, cv2.LINE_AA)

                msg = CompressedImage()
                msg.header.stamp = now
                msg.format = "jpeg"
                jpeg_bytes = self.nvjpeg.encode(frame, 100)
                msg.data = jpeg_bytes
                self.publisher.publish(msg)

            except Exception as e:
                rospy.logerr_throttle(2.0, f"[{self.name}] 推論或發佈出錯：{e}")
                rospy.sleep(0.05)

def main():
    rospy.init_node("rtsp_infer_publish_node", anonymous=True)
    rp = rospkg.RosPack()
    model_path = rp.get_path("all_function") + "/models/RT_DETR/200_6_best.pt"
    conf_thres = rospy.get_param("~conf_thres", 0.6)
    imgsz = rospy.get_param("~imgsz", 960)
    focal_length_px = rospy.get_param("~focal_length_px", 200)

    crop_rect = tuple(rospy.get_param("~crop_rect", [120, 0, 5855, 1080]))
    left_bound_angle = rospy.get_param("~left_bound_angle", -90.0)
    right_bound_angle = rospy.get_param("~right_bound_angle", 90.0)
    marker_frame_id = rospy.get_param("~marker_frame_id", "odom")

    cameras = rospy.get_param("~cameras", [
        {
            "name": "cam1",
            "url": "rtsp://192.168.10.92:8554/s2_1234",
            "topic": "/camera/color/image_raw/inferenced/compressed"
        },
    ])

    nodes = []
    for cam in cameras:
        node = RTSPCameraWithRTDETR(
            cam["name"], cam["url"], cam["topic"],
            model_path, conf_thres, imgsz, focal_length_px,
            crop_rect, left_bound_angle, right_bound_angle,
            marker_frame_id=marker_frame_id
        )
        node.start()
        nodes.append(node)

    rospy.loginfo("✅ RTSP + RT-DETR + 距離/角度追蹤（新版 ObjectTracker/Kalman）啟動完成")
    rospy.spin()

    for node in nodes:
        node.stop()

if __name__ == "__main__":
    main()
