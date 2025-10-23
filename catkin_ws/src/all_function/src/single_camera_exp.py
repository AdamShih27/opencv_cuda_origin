#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import cv2
import numpy as np
import rospkg
import rospy
from sensor_msgs.msg import CompressedImage

# 物件偵測 + 距離估計（bbox 高度法）
from core.rtdetr_predict_core import RTDETRPredictor
from core.bbox2distance_core import BBoxDistanceEstimator
# 海平線幾何推距離（這版不跑分割，直接用固定海平線）
from core.horizon2distance import DetectionDistanceEstimator


def make_gst_pipeline(rtsp_url,
                      codec="h264",
                      transport="udp",
                      latency_ms=0,
                      drop_on_latency=True):
    assert codec in ("h264", "h265")
    assert transport in ("udp", "tcp")

    depay = "rtph264depay" if codec == "h264" else "rtph265depay"
    parse = "h264parse"    if codec == "h264" else "h265parse"
    decoder = "avdec_h264" if codec == "h264" else "avdec_h265"

    drop_str = "true" if drop_on_latency else "false"
    pipeline = (
        f"rtspsrc location={rtsp_url} latency={latency_ms} protocols={transport} drop-on-latency={drop_str} ! "
        f"{depay} ! {parse} ! {decoder} ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )
    return pipeline


def resolve_rtdetr_weights_with_rospkg() -> str:
    rp = rospkg.RosPack()
    pkg_path = rp.get_path('all_function')
    candidates = [
        os.path.join(pkg_path, 'models', 'RT_DETR', '200_6_best.pt'),
        os.path.join(pkg_path, 'models', 'RT_DETR', '200_6_best.p'),
        os.path.join(pkg_path, 'models', 'RT_DETR', 'best.pt'),
        os.path.join(pkg_path, 'models', 'RT_DETR', 'weights.pt'),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "找不到 RT-DETR 權重檔，請確認下列任一檔案存在：\n  - " + "\n  - ".join(candidates)
    )


class RTSPCameraReader:
    """
    RTSP 拉流 →（若提供 K/D 則）CUDA 去畸變 → 回傳 (FRAME, FRAME_UDT)
    並提供 get_focal_length_px(shape, use_udt)
    """
    def __init__(self, name, rtsp_url, *,
                 codec="h265", transport="udp",
                 latency_ms=0, dim=None, K=None, D=None, balance=0.9):
        self.name = name
        self.rtsp_url = rtsp_url
        self.K = K
        self.D = D
        self.balance = float(balance)
        self.dim = tuple(dim) if dim is not None else None
        self.calib_dim = self.dim

        self.stream = None
        self.gpu_map1 = None
        self.gpu_map2 = None
        self.maps_ready = False
        self.map_dim = None
        self.new_K = None

        pipe = make_gst_pipeline(rtsp_url, codec=codec, transport=transport, latency_ms=latency_ms)
        self.cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError(f"[{self.name}] 無法開啟 RTSP 串流：{rtsp_url}")

        self.enable_udt = (self.K is not None) and (self.D is not None)

    def _ensure_maps(self, frame):
        if not self.enable_udt:
            return
        h, w = frame.shape[:2]
        cur_dim = (w, h)
        if self.dim is None:
            self.dim = cur_dim
        if (not self.maps_ready) or (self.map_dim != cur_dim) or (self.dim != cur_dim):
            use_dim = cur_dim
            self.map_dim = cur_dim
            if self.stream is None:
                self.stream = getattr(cv2.cuda, "Stream", cv2.cuda_Stream)()
            self.new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                self.K, self.D, use_dim, np.eye(3), balance=self.balance)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                self.K, self.D, np.eye(3), self.new_K, use_dim, cv2.CV_32FC1)
            self.gpu_map1 = cv2.cuda_GpuMat(); self.gpu_map2 = cv2.cuda_GpuMat()
            self.gpu_map1.upload(map1); self.gpu_map2.upload(map2)
            self.maps_ready = True

    def get_focal_length_px(self, frame_shape, use_udt=True):
        if use_udt and self.enable_udt and (self.new_K is not None):
            return float(self.new_K[1, 1])
        if self.K is None:
            return None
        h, w = frame_shape[:2]
        if self.calib_dim is None:
            return float(self.K[1, 1])
        _, calib_h = self.calib_dim
        sy = (h / float(calib_h)) if calib_h else 1.0
        return float(self.K[1, 1] * sy)

    def read(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None, None
        FRAME = frame
        if self.enable_udt:
            self._ensure_maps(FRAME)
            gpu = cv2.cuda_GpuMat(); gpu.upload(FRAME, stream=self.stream)
            und = cv2.cuda.remap(gpu, self.gpu_map1, self.gpu_map2,
                                 interpolation=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 stream=self.stream)
            FRAME_UDT = und.download(stream=self.stream)
            self.stream.waitForCompletion()
        else:
            FRAME_UDT = FRAME
        return FRAME, FRAME_UDT

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None


def draw_detections(image, detections, color=(0, 255, 0)):
    """
    在影像上畫 bbox 與標籤（含類別/分數/兩種距離）。
    """
    img = image.copy()
    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    return img


def main():
    rospy.init_node("single_camera_exp", anonymous=True)
    # 新增一個原圖 topic（未去畸變、未標註）
    pub_raw = rospy.Publisher("/inference/image_raw/compressed", CompressedImage, queue_size=1)
    # 既有的視覺化 topic（去畸變 + 畫框）
    pub_vis = rospy.Publisher("/inference/image/compressed", CompressedImage, queue_size=1)

    # ===== 相機內參（依你原設定，factor 等比放大） =====
    factor = 1.25
    DIM = (int(640 * factor), int(480 * factor))
    K = np.array([[268.06554187*factor, 1.51263281*factor, 320*factor],
                  [0., 356.57309093*factor, 231.71146684*factor],
                  [0., 0., 1.]])
    D = np.array([[0.05184647], [0.01756823], [-0.02638422], [0.00762106]])

    # ===== 讀取器（可改 codec/transport） =====
    reader = RTSPCameraReader(
        name="cam1",
        rtsp_url="rtsp://admin:admin@192.168.1.108:554/video",
        codec="h265",            # 若 CPU 無 avdec_h265，可改 "h264"
        transport="udp",         # 不穩定時可改 "tcp"
        latency_ms=0,
        dim=DIM, K=K, D=D, balance=1.0,
    )

    # ===== 解析 RT-DETR 權重路徑（rospkg） =====
    rtdetr_model_path = resolve_rtdetr_weights_with_rospkg()
    rospy.loginfo(f"[info] RT-DETR weights: {rtdetr_model_path}")

    # ===== 物件偵測（RT-DETR） =====
    predictor = RTDETRPredictor(
        model_path=rtdetr_model_path,
        conf_thres=0.5,
        imgsz=640,
        device=None
    )

    USE_UDT_FOR_INFER = True  # 用去畸變影像做偵測與幾何

    # ===== 距離估計參數（從 main 指定）=====
    FY_PX = 200.0 * factor     # 對應目前輸入影像尺寸
    CAMERA_HEIGHT_M = 3.8
    MIN_PERP_PX = 20.0
    HORIZON_DEBUG = False

    dis_estimator = BBoxDistanceEstimator(focal_length_px=float(FY_PX))
    horizon_dis_est = DetectionDistanceEstimator(
        camera_height=float(CAMERA_HEIGHT_M),
        fy=float(FY_PX),
        debug_print=bool(HORIZON_DEBUG),
        min_perp_px=float(MIN_PERP_PX),
    )
    rospy.loginfo(f"[info] 初始化距離估計器：fy={FY_PX:.3f}px, h={CAMERA_HEIGHT_M}m")

    t0, frames = time.time(), 0

    try:
        while not rospy.is_shutdown():
            FRAME, FRAME_UDT = reader.read()
            if FRAME is None:
                rospy.logwarn("[cam1] 讀取失敗，略過此幀")
                continue

            # ---------- (A) 先把「原圖」發出去（未去畸變、未標註） ----------
            ok_raw, buf_raw = cv2.imencode(".jpg", FRAME, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if ok_raw:
                msg_raw = CompressedImage()
                msg_raw.header.stamp = rospy.Time.now()
                msg_raw.format = "jpeg"
                msg_raw.data = np.array(buf_raw).tobytes()
                pub_raw.publish(msg_raw)

            # ---------- (B) 偵測影像選擇 ----------
            img_for_infer = FRAME_UDT if USE_UDT_FOR_INFER else FRAME

            # ---------- (C) 物件偵測 ----------
            boxes, classes, scores = predictor.infer(img_for_infer)

            # ---------- (D) 固定海平線：斜率=0、位置=畫面中心 ----------
            h, w = img_for_infer.shape[:2]
            horizon_center = (int(w * 0.5), int(h * 0.5))
            horizon_slope_pts = [{"slope": 0.0, "center": horizon_center}]

            # ---------- (E) 組裝 + 兩種估距 ----------
            mapped_detections = []
            for box, cls, sc in zip(boxes, classes, scores):
                x1, y1, x2, y2 = box
                xg = 0.5 * (x1 + x2)
                yg = y2
                mapped_detections.append({
                    "class": cls,
                    "score": sc,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "src": 0,
                    "original": {"x": float(xg), "y": float(yg)},
                })
            mapped_detections = [d for d in mapped_detections if (d["bbox"][3] - d["bbox"][1]) >= 15]

            mapped_detections = horizon_dis_est.horizon2distance(mapped_detections, horizon_slope_pts)
            mapped_detections = dis_estimator.estimate_distance(mapped_detections)

            # ---------- (F) 畫框（畫在偵測用影像上）並發出去 ----------
            vis = draw_detections(img_for_infer, mapped_detections, color=(0, 255, 0))
            ok_vis, buf_vis = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if ok_vis:
                msg_vis = CompressedImage()
                msg_vis.header.stamp = rospy.Time.now()
                msg_vis.format = "jpeg"
                msg_vis.data = np.array(buf_vis).tobytes()
                pub_vis.publish(msg_vis)

            frames += 1
            if frames % 30 == 0:
                dt = time.time() - t0
                fps = frames / dt if dt > 0 else 0.0
                n_total = len(mapped_detections)
                n_h = sum(1 for d in mapped_detections if d.get("distance_horizon") is not None)
                n_b = sum(1 for d in mapped_detections if d.get("distance_bbox") is not None)
                rospy.loginfo(f"[cam1] FPS≈{fps:.1f}  dets={n_total}  "
                              f"with_horizon={n_h}  with_bbox={n_b}  "
                              f"h_center={horizon_center}")

    except KeyboardInterrupt:
        pass
    finally:
        reader.release()

if __name__ == "__main__":
    main()
