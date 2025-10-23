# -*- coding: utf-8 -*-
import traceback
import numpy as np

from core.stitching_core import TripletStitcher
from core.angle_mark_core import PanoHorizonAngleMarker
from core.rtdetr_predict_core import RTDETRPredictor
from core.bbox_to_original_core import BBoxToOriginalMapper
from core.bbox_draw_core import draw_bboxes_from_mapped
from core.bbox2distance_core import BBoxDistanceEstimator
from core.horizon2distance import DetectionDistanceEstimator
from core.angle_estimate_core import HorizonX2AngleMapper
from core.detections2marker_core import convert_detections_to_pose_info
from core.segformer_horizon_core import SegFormerPredictor
from core.distance_fused_core import DistanceFuser
from nvjpeg import NvJpeg  # ✅ nvJPEG

def worker(input_q, output_q, cfg):
    """
    修正重點：
      1) nvjpeg.decode 加入壞幀防呆
      2) RT-DETR 無偵測 / 映射後為空 → 提早回傳
      3) convert_detections_to_pose_info 對齊新版輸出（class/distance/angle/bbox/scale/color）
      4) 先做海平線距離、再做 bbox 距離（順序可互換；這裡採先海平線）
    """
    # ===== 初始化 =====
    nvjpeg   = NvJpeg()  # 每個 process 各自初始化
    stitcher = TripletStitcher(**cfg['stitcher'])
    marker   = PanoHorizonAngleMarker(**cfg['marker'])
    mapper   = BBoxToOriginalMapper(**cfg['mapper'])
    dis_estimator = BBoxDistanceEstimator(**cfg['distance_estimator'])
    horizon_distance_estimator = DetectionDistanceEstimator()
    ang_estimator = HorizonX2AngleMapper(**cfg['angle_estimator'])
    distance_fuser = DistanceFuser()
    rtdetr_predictor = RTDETRPredictor(
        model_path = cfg['rtdetr']['model_path'],
        conf_thres = cfg['rtdetr']['conf'],
        imgsz      = cfg['rtdetr']['imgsz']
    )
    segformer = SegFormerPredictor(model_path=cfg['segformer']['model_path'])

    while True:
        job = input_q.get()
        if job is None:
            break

        try:
            idx = job['idx']
            global_M_L = job['global_M_L']
            global_M_R = job['global_M_R']

            # ===== 1) nvJPEG 解碼（含壞幀防呆） =====
            try:
                imgs = [nvjpeg.decode(data) for data in job['imgs_raw']]
            except Exception:
                traceback.print_exc()
                output_q.put({'idx': idx, 'raw_img': None, 'marker_data': []})
                continue

            if not imgs or any(im is None for im in imgs):
                output_q.put({'idx': idx, 'raw_img': None, 'marker_data': []})
                continue

            img_height, img_width = imgs[0].shape[:2]

            # ===== 2) SegFormer 估算 horizon slope/center（批次）=====
            horizon_slope_pts = segformer.infer_horizon_slope_batch(imgs)
            # 期望格式：[
            #   {'slope': float, 'center': (x, y)},
            #   {'slope': float, 'center': (x, y)},
            #   {'slope': float, 'center': (x, y)}
            # ]

            # ===== 3) 三圖拼接 =====
            pano, _, _ = stitcher.stitch(imgs, global_M_L, global_M_R)

            # ===== 4) RT-DETR 偵測（於 pano） =====
            boxes, classes, scores = rtdetr_predictor.infer(pano)

            # 無偵測 → 直接畫角度後回傳
            if boxes is None or len(boxes) == 0:
                # pano_mask = marker.mark(pano, horizon_slope_pts, global_M_L, global_M_R, img_width)
                output_q.put({
                    'idx': idx,
                    # 'raw_img': pano_mask,
                    'raw_img' : pano,
                    'marker_data': [],
                })
                continue

            # ===== 5) bbox 映射回原圖來源（含 source/center 等欄位）=====
            mapped_detections = mapper.map_multiple(
                boxes, classes, scores,
                global_M_L, global_M_R,
                img_height, img_width,
                offset_w=img_width            # 顯式指定
            )
            if not mapped_detections:
                pano_mask = marker.mark(pano, horizon_slope_pts, global_M_L, global_M_R, img_width)
                output_q.put({
                    'idx': idx,
                    'raw_img': pano_mask,
                    'marker_data': [],
                })
                continue

            # ===== 6) 估角（由海平線資訊推水平角）=====
            mapped_detections = ang_estimator.process_with_horizon_slope(
                horizon_slope_pts, mapped_detections, global_M_L, global_M_R, img_width
            )
            # 期望增欄位：'angle'（度）, 'slope', 'center'

            # ===== 7) 海平線法估距 =====
            mapped_detections = horizon_distance_estimator.horizon2distance(
                mapped_detections, horizon_slope_pts
            )
            # 期望增欄位：'distance_horizon'

            # ===== 8) BBox 高度法估距 =====
            mapped_detections = dis_estimator.estimate_distance(mapped_detections)
            # 期望增欄位：'distance_bbox'

            # ===== 9) 融合距離（最終 distance）=====
            mapped_detections = distance_fuser.fuse(
                mapped_detections, img_width, img_height
            )
            # 期望增欄位：'distance'（融合後）

            # ===== 10) 繪製：角度標註 + 偵測框 =====
            # pano_mask = marker.mark(pano, horizon_slope_pts, global_M_L, global_M_R, img_width)
            # pano_mask = draw_bboxes_from_mapped(pano_mask, mapped_detections)
            pano_mask = draw_bboxes_from_mapped(pano, mapped_detections)

            # 若要輸出 JPEG，可於外層統一管理 encoder
            # pano_mask_jpeg = nvjpeg.encode(pano_mask, quality=100)

            # ===== 11) 產出 marker_data（新版：class/distance/angle/bbox/scale/color）=====
            marker_data = convert_detections_to_pose_info(mapped_detections)
            # 範例：
            # {
            #   "class": "RedBall",
            #   "distance": 42.3,
            #   "angle": -13.5,
            #   "bbox": [x1, y1, x2, y2],
            #   "scale": (w, d, h),
            #   "color": (1.0, 0.0, 0.0),
            # }

            # ===== 12) 回傳 =====
            output_q.put({
                'idx': idx,
                'raw_img': pano_mask,
                'marker_data': marker_data,
            })

        except Exception:
            traceback.print_exc()
            # 出錯也回傳一份，避免上游阻塞
            output_q.put({'idx': job.get('idx', -1), 'raw_img': None, 'marker_data': []})
