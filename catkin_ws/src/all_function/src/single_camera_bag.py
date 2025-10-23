#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal")

import os
import time
import csv
import errno
import cv2
import numpy as np
import rospkg
import rospy
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
from collections import deque
from math import hypot

from core.rtdetr_predict_core import RTDETRPredictor
from core.yolo_predict_core import YOLOPredictor
from core.bbox2distance_core import BBoxDistanceEstimator


def resolve_rtdetr_weights_with_rospkg() -> str:
    rp = rospkg.RosPack()
    pkg_path = rp.get_path('all_function')
    p = os.path.join(pkg_path, 'models', 'RT_DETR', 'best.pt')
    if os.path.isfile(p):
        return p
    raise FileNotFoundError("找不到 RT-DETR 權重檔")

def resolve_yolo_weights_with_rospkg() -> str:
    rp = rospkg.RosPack()
    pkg_path = rp.get_path('all_function')
    p = os.path.join(pkg_path, 'models', 'YOLO', 'best.pt')
    if os.path.isfile(p):
        return p
    raise FileNotFoundError("找不到 YOLO 權重檔")

def _make_cuda_stream():
    try:
        return cv2.cuda.Stream()
    except AttributeError:
        return cv2.cuda_Stream()


# === 小工具 ===
def safe_float(v, default=0.0):
    try:
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)

def safe_hypot(a, b):
    return hypot(safe_float(a), safe_float(b))

def fmt_or_na(val, fmt="{:.2f}"):
    if val is None:
        return "N/A"
    try:
        return fmt.format(val)
    except Exception:
        return "N/A"


# === 固定水平海平線：直接以 y 距離估算距離 ===
def horizon_distance_flat_for_detections(
    detections,
    horizon_row: int,       # v_h（像素）
    fy_px: float,           # 像素（垂直焦距）
    camera_height_m: float, # 公尺
    min_dy_px: float = 5.0  # 小於此像素距離視為不穩定，回傳 None
):
    """
    距離 D ≈ (h * fy) / |y0 - v_h|
      - y0：偵測框的接地點（取 bbox 下緣）
      - v_h：海平線所在的 row（像素）
    計算後寫入 det["distance_horizon"]（None 代表不穩定或無法計算）。
    """
    if not detections:
        return detections

    if fy_px is None or fy_px <= 0:
        for det in detections:
            det["distance_horizon"] = None
        return detections

    v_h = int(horizon_row)

    for det in detections:
        if "original" in det and det["original"] is not None:
            y0 = float(det["original"].get("y", None))
        else:
            x1, y1, x2, y2 = det["bbox"]
            y0 = float(y2)

        if y0 is None:
            det["distance_horizon"] = None
            continue

        dy = abs(y0 - v_h)
        if dy < float(min_dy_px):
            det["distance_horizon"] = None
            continue

        D = (float(camera_height_m) * float(fy_px)) / float(dy)
        det["distance_horizon"] = float(D)

    return detections


class TopicCameraInferenceNode:
    def __init__(self,
                 detector_type="rtdetr",
                 factor=1.25,
                 manual_fy_px=445.0,
                 csv_path=os.path.expanduser("~/inference_distances.csv"),
                 csv_flush=True,
                 horizon_y_ratio=0.5,           # 基準海平線比例（0~1）
                 horizon_offset_px=0,           # 海平線像素偏移（+往下 / -往上）
                 csv_overwrite=True             # ★ 如果檔案已存在，是否先刪除再新建
                 ):
        rospy.init_node("topic_camera_inference", anonymous=True)

        self.detector_type = detector_type.lower()  # "rtdetr" 或 "yolo"

        # 影像 topic
        self.image_topic = "/inference/image_raw/compressed"
        self.topic_udt  = "/inference/image_udt/compressed"
        self.topic_vis  = "/inference/image/compressed"
        self.jpeg_quality = 90
        self.publish_udt = True
        self.use_udt_for_infer = True

        # 允許保留 factor，K 也同步乘上 factor（僅供去畸變 remap 用）
        self.factor = float(factor)
        self.K = np.array([[268.06554187 * self.factor, 1.51263281 * self.factor, 320 * self.factor],
                           [0., 356.57309093 * self.factor, 231.71146684 * self.factor],
                           [0., 0., 1.]])
        self.D = np.array([[0.05184647], [0.01756823], [-0.02638422], [0.00762106]])
        self.balance = 1.0
        self.enable_udt = True

        # 水平視角裁切
        self.INIT_HFOV_DEG = 150.0
        self.CUT_LEFT_DEG  = 30.0
        self.CUT_RIGHT_DEG = 30.0

        # 幾何/門檻
        self.CAMERA_HEIGHT_M = 3.75
        self.MIN_PERP_PX     = 5.0
        self.min_bbox_h_px   = 3

        # 海平線（固定水平 + 偏移）
        self.horizon_y_ratio = float(horizon_y_ratio)
        self.horizon_offset_px = int(horizon_offset_px)

        # 推論
        self.conf_thres = 0.5
        self.imgsz = 640
        self.device = None

        # Publisher
        self.pub_udt = rospy.Publisher(self.topic_udt, CompressedImage, queue_size=1)
        self.pub_vis = rospy.Publisher(self.topic_vis, CompressedImage, queue_size=1)

        # 初始化
        self.stream   = None
        self.gpu_map1 = None
        self.gpu_map2 = None
        self.maps_ready = False
        self.map_dim    = None

        # 用於「最大 IoU」選框的狀態
        self.prev_bbox = None  # 上一幀輸出的 bbox [x1,y1,x2,y2]

        # 選擇模型
        if self.detector_type == "rtdetr":
            model_path = resolve_rtdetr_weights_with_rospkg()
            self.predictor = RTDETRPredictor(model_path, conf_thres=self.conf_thres,
                                             imgsz=self.imgsz, device=self.device)
            rospy.loginfo("[topic_camera_inference] 使用 RT-DETR 模型")
        elif self.detector_type == "yolo":
            model_path = resolve_yolo_weights_with_rospkg()
            self.predictor = YOLOPredictor(model_path, conf_thres=self.conf_thres,
                                           imgsz=self.imgsz, device=self.device)
            rospy.loginfo("[topic_camera_inference] 使用 YOLO 模型")
        else:
            raise ValueError("detector_type 必須是 'rtdetr' 或 'yolo'")

        # === 焦距（唯一來源） ===
        self.fy_px = float(manual_fy_px)
        rospy.loginfo(f"[focal] Using MANUAL fy = {self.fy_px:.2f} px")

        # BBox 高度法估距器（用同一個 fy）
        self.box_dis_estimator = BBoxDistanceEstimator(focal_length_px=self.fy_px)

        # Odom
        self.odom_topic = "/mavros/global_position/local"
        self.odom_buffer = deque(maxlen=500)
        self.odom_time_tolerance = 0.20
        self.sub_odom = rospy.Subscriber(self.odom_topic, Odometry, self.cb_odom, queue_size=50)

        self.sub = rospy.Subscriber(
            self.image_topic, CompressedImage, self.cb_image,
            queue_size=1, buff_size=2**22)

        # === CSV 設定（由 main 傳入） ===
        self.csv_path = csv_path
        self.csv_flush = bool(csv_flush)

        # 確保目錄存在（若只有檔名，dirname 會是空字串，需跳過）
        csv_dir = os.path.dirname(self.csv_path)
        if csv_dir:
            try:
                os.makedirs(csv_dir, exist_ok=True)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        # ★ 若檔案已存在且要求覆寫：先刪除，再新建
        if csv_overwrite and os.path.exists(self.csv_path):
            try:
                os.remove(self.csv_path)
                rospy.loginfo(f"[csv] removed existing file: {self.csv_path}")
            except OSError as e:
                rospy.logwarn(f"[csv] remove existing file failed: {e}")

        # 以 w 模式新建並寫表頭（覆寫或首次）
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "stamp_sec",          # 影像時間戳（秒）
            "d_origin_m",         # Odom 到原點距離
            "det_box_m",          # BBox 高度法距離
            "det_horizon_m",      # 海平線 y 距離法
            "odom_dt_sec",        # Odom與影像時間差
            "fy_px",              # 使用的焦距（像素）
            "horizon_row_px",     # 當前海平線 row（含偏移）
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "cls", "score"
        ])
        self.csv_file.flush()

        rospy.on_shutdown(self._on_shutdown)

        rospy.loginfo("[topic_camera_inference] ready. Listening on %s", self.image_topic)
        rospy.loginfo(f"[csv] logging to: {self.csv_path} (flush={self.csv_flush})")
        rospy.loginfo(f"[horizon] ratio={self.horizon_y_ratio:.3f}, offset_px={self.horizon_offset_px:+d}")

    # === IoU 工具 ===
    def _bbox_iou(self, a, b):
        # a, b: [x1,y1,x2,y2]
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return inter / union

    # === Odom ===
    def cb_odom(self, msg: Odometry):
        try:
            t = msg.header.stamp.to_sec() if msg.header.stamp else rospy.Time.now().to_sec()
            x = safe_float(getattr(msg.pose.pose.position, "x", None))
            y = safe_float(getattr(msg.pose.pose.position, "y", None))
            self.odom_buffer.append((t, x, y))
        except Exception as e:
            rospy.logwarn(f"[odom] parse error: {e}")

    def _nearest_odom(self, t_img: float):
        if not self.odom_buffer:
            return None, None
        best = min(self.odom_buffer, key=lambda r: abs(r[0] - t_img))
        dt = abs(best[0] - t_img)
        return best, dt

    # === UDT ===
    def _ensure_maps(self, frame):
        """建立/更新去畸變 remap；不計算 new_K，直接用原始 K 當作輸出 P。"""
        if not self.enable_udt:
            return
        h, w = frame.shape[:2]
        cur_dim = (w, h)
        if (not self.maps_ready) or (self.map_dim != cur_dim):
            self.map_dim = cur_dim
            if self.stream is None:
                self.stream = _make_cuda_stream()
            P = self.K.copy()
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                self.K, self.D, np.eye(3), P, cur_dim, cv2.CV_32FC1
            )
            self.gpu_map1, self.gpu_map2 = cv2.cuda_GpuMat(), cv2.cuda_GpuMat()
            self.gpu_map1.upload(map1)
            self.gpu_map2.upload(map2)
            self.maps_ready = True
            rospy.loginfo("[UDT] remap initialized (using original K as P)")

    # === 主 callback ===
    def cb_image(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        # 去畸變 & 裁切
        frame_udt = frame
        if self.enable_udt:
            self._ensure_maps(frame)
            gpu = cv2.cuda_GpuMat()
            gpu.upload(frame, stream=self.stream)
            und = cv2.cuda.remap(gpu, self.gpu_map1, self.gpu_map2,
                                 interpolation=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 stream=self.stream)
            self.stream.waitForCompletion()
            frame_udt = und.download()
            frame_udt = self._crop_hfov(frame_udt)
            if self.publish_udt:
                self._publish_jpeg(self.pub_udt, frame_udt)

        img_for_infer = frame_udt if self.use_udt_for_infer else frame

        # 偵測
        boxes, classes, scores = self.predictor.infer(img_for_infer)

        # === 僅保留與上一幀 bbox IoU 最大的框（若無上一幀或 IoU 全為 0，退回面積最大） ===
        all_dets = []
        for box, cls, sc in zip(boxes, classes, scores):
            x1, y1, x2, y2 = map(int, box)
            if (y2 - y1) < self.min_bbox_h_px:
                continue
            all_dets.append({
                "class": cls,
                "score": float(sc),
                "bbox": [x1, y1, x2, y2],
                "original": {"x": 0.5*(x1+x2), "y": float(y2)}
            })

        if all_dets:
            if self.prev_bbox is not None:
                iou_pairs = [(self._bbox_iou(det["bbox"], self.prev_bbox), det) for det in all_dets]
                best_iou, best_det = max(iou_pairs, key=lambda t: t[0])
                if best_iou == 0.0:
                    best_det = max(
                        all_dets,
                        key=lambda d: max(0, d["bbox"][2]-d["bbox"][0]) * max(0, d["bbox"][3]-d["bbox"][1])
                    )
            else:
                best_det = max(
                    all_dets,
                    key=lambda d: max(0, d["bbox"][2]-d["bbox"][0]) * max(0, d["bbox"][3]-d["bbox"][1])
                )
            dets = [best_det]
            self.prev_bbox = best_det["bbox"]
        else:
            dets = []
            self.prev_bbox = None

        # 兩種估距統一用相同 fy（self.fy_px）
        self.box_dis_estimator.focal_length_px = self.fy_px

        # 固定水平海平線（基準比率 + 像素偏移）
        h, w = img_for_infer.shape[:2]
        v_h_base = int(h * self.horizon_y_ratio)
        v_h = int(np.clip(v_h_base + self.horizon_offset_px, 0, h - 1))
        dets = horizon_distance_flat_for_detections(
            detections=dets,
            horizon_row=v_h,
            fy_px=self.fy_px,
            camera_height_m=self.CAMERA_HEIGHT_M,
            min_dy_px=self.MIN_PERP_PX
        )

        # BBox 高度法距離（沿用外部估計器）
        dets = self.box_dis_estimator.estimate_distance(dets)

        # Odom距離
        t_img = msg.header.stamp.to_sec() if msg.header.stamp else time.time()
        (odom_rec, dt) = self._nearest_odom(t_img)
        d_origin = None
        if odom_rec is not None and dt <= self.odom_time_tolerance:
            _, x_val, y_val = odom_rec
            d_origin = safe_hypot(x_val, y_val)

        # 平均距離（目前只保留單框，等於該框距離）
        det_box_dists = [d.get("distance_bbox") for d in dets if d.get("distance_bbox")]
        det_hor_dists = [d.get("distance_horizon") for d in dets if d.get("distance_horizon")]
        avg_box = np.mean(det_box_dists) if det_box_dists else None
        avg_hor = np.mean(det_hor_dists) if det_hor_dists else None

        # === 寫入 CSV ===
        stamp_sec = msg.header.stamp.to_sec() if msg.header.stamp else time.time()
        if dets:
            bx1, by1, bx2, by2 = dets[0]["bbox"]
            bcls = dets[0].get("class", "")
            bscore = dets[0].get("score", "")
        else:
            bx1 = by1 = bx2 = by2 = ""
            bcls = ""
            bscore = ""

        row = [
            f"{stamp_sec:.6f}",
            "" if d_origin is None else f"{float(d_origin):.6f}",
            "" if avg_box  is None else f"{float(avg_box):.6f}",
            "" if avg_hor  is None else f"{float(avg_hor):.6f}",
            "" if dt is None else f"{float(dt):.6f}",
            f"{self.fy_px:.3f}",
            f"{v_h:d}",
            "" if bx1=="" else int(bx1),
            "" if by1=="" else int(by1),
            "" if bx2=="" else int(bx2),
            "" if by2=="" else int(by2),
            str(bcls) if bcls != "" else "",
            "" if bscore=="" else f"{float(bscore):.6f}",
        ]
        try:
            self.csv_writer.writerow(row)
            if self.csv_flush:
                self.csv_file.flush()
        except Exception as e:
            rospy.logwarn(f"[csv] write error: {e}")

        # === 多行文字輸出（換行顯示） ===
        odom_text = (
            f"d_origin={fmt_or_na(d_origin)} m\n"
            f"det_box={fmt_or_na(avg_box)} m\n"
            f"det_hor={fmt_or_na(avg_hor)} m"
        )

        vis = self._draw_detections(img_for_infer, dets)
        vis = self._draw_horizon_line(vis, v_h, v_h_base)
        vis = self._draw_overlay(vis, odom_text)
        self._publish_jpeg(self.pub_vis, vis)

    def _crop_hfov(self, frame):
        h, w = frame.shape[:2]
        px_per_deg = w / float(self.INIT_HFOV_DEG)
        left_px  = int(round(self.CUT_LEFT_DEG  * px_per_deg))
        right_px = int(round(self.CUT_RIGHT_DEG * px_per_deg))
        x0 = left_px
        x1 = w - right_px
        if x0 >= x1:
            return frame
        return frame[:, x0:x1]

    def _publish_jpeg(self, pub, img_bgr):
        ok, buf = cv2.imencode(".jpg", img_bgr,
                               [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if ok:
            msg = CompressedImage()
            msg.header.stamp = rospy.Time.now()
            msg.format = "jpeg"
            msg.data = np.array(buf).tobytes()
            pub.publish(msg)

    def _draw_detections(self, img, detections, color=(0, 255, 0)):
        out = img.copy()
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        return out

    def _draw_horizon_line(self, img, v_h, v_h_base):
        out = img.copy()
        h, w = out.shape[:2]
        # 實際海平線(藍) + 基準(紫)
        cv2.line(out, (0, int(v_h)), (w-1, int(v_h)), (255, 0, 0), 2)
        cv2.line(out, (0, int(v_h_base)), (w-1, int(v_h_base)), (255, 0, 255), 1)
        return out

    def _draw_overlay(self, img, text: str):
        """
        在畫面左上角輸出多行文字
        text: 可以包含 '\n' 或 '|' 來換行
        """
        out = img.copy()
        lines = text.replace("|", "\n").split("\n")
        y0 = 20
        dy = 22   # 每行間距
        for i, line in enumerate(lines):
            cv2.putText(out, line.strip(), (8, y0 + i*dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2, cv2.LINE_AA)
        return out

    def _on_shutdown(self):
        try:
            if hasattr(self, "csv_file") and self.csv_file:
                self.csv_file.flush()
                self.csv_file.close()
        except Exception:
            pass


def main():
    # ==== 在這裡設定（變數已整理）====
    USE_MODEL = "rtdetr"    # "rtdetr" 或 "yolo"

    FACTOR  = 1.25          # 影像縮放倍率（K 與 fy 同步乘上）
    BASE_FY = 440           # 標定原始 fy（未乘 factor）
    FY_MANUAL = BASE_FY * FACTOR  # 最終使用的手動焦距（像素）

    # === 海平線設定 ===
    HORIZON_Y_RATIO = 0.50      # 以影像高度的 50% 為基準
    HORIZON_OFFSET_PX = 15      # 正=往下、負=往上

    # === CSV 參數（存在「程式碼同目錄/logs/」）===
    script_dir = os.path.dirname(os.path.abspath(__file__))   # 目前程式碼的目錄
    logs_dir = os.path.join(script_dir, "logs")
    try:
        os.makedirs(logs_dir, exist_ok=True)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    CSV_PATH      = os.path.join(logs_dir, "inference_distances3.csv")
    CSV_FLUSH     = True   # True=每行立即落盤；False=效能較好但落盤延遲
    CSV_OVERWRITE = True   # ★ 啟動時若同名檔存在：先刪除再新建

    node = TopicCameraInferenceNode(
        detector_type=USE_MODEL,
        factor=FACTOR,
        manual_fy_px=FY_MANUAL,
        csv_path=CSV_PATH,
        csv_flush=CSV_FLUSH,
        horizon_y_ratio=HORIZON_Y_RATIO,
        horizon_offset_px=HORIZON_OFFSET_PX,
        csv_overwrite=CSV_OVERWRITE
    )
    rospy.spin()


if __name__ == "__main__":
    main()
