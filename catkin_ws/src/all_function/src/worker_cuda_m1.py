# -*- coding: utf-8 -*-
import traceback
import concurrent.futures
import numpy as np

from core.stitching_core import TripletStitcher
from core.traditional_horizon_core import TraditionalHorizonDetector
from core.angle_mark_core import PanoHorizonAngleMarker
from core.rtdetr_predict_core import RTDETRPredictor
from core.bbox_to_original_core import BBoxToOriginalMapper
from core.bbox_draw_core import draw_bboxes_from_mapped
from core.bbox2distance_core import BBoxDistanceEstimator
from core.horizon2distance import DetectionDistanceEstimator
from core.angle_estimate_core import HorizonX2AngleMapper
from core.detections2marker_core import convert_detections_to_pose_info
from core.distance_fused_core import DistanceFuser
from nvjpeg import NvJpeg


def worker(input_q, output_q, cfg):
    """
    主工作執行緒：
    1) nvJPEG 解碼三張 JPEG buffer
    2) 平行執行傳統海平線偵測與三圖拼接
    3) RT-DETR 偵測於全景圖上
    4) bbox 映射回原圖來源 & 以海平線/高度與 bbox 法估距
    5) 融合距離、估角度、畫標註
    6) 產出 marker_data（僅保留 distance/angle/bbox/scale/color）

    輸出格式：
    output_q.put({
        'idx': idx,
        'raw_img': pano_mask,        # numpy BGR 圖（已畫角度與 bbox）
        'marker_data': [             # 由 convert_detections_to_pose_info 產生
            {
                "class": "RedBall",
                "distance": 42.3,             # 融合後距離（m）
                "angle": -13.5,               # 水平角（度，原值）
                "bbox": [x1, y1, x2, y2],     # pano/mapper 後的 bbox
                "scale": (w, d, h),           # 估算的實際大小（公尺）
                "color": (1.0, 0.0, 0.0),     # RGB 0~1
            },
            ...
        ],
    })
    """

    # ===== 0) 初始化共用元件 =====
    nvjpeg = NvJpeg()  # 每個 process 各自初始化

    stitcher_params            = cfg['stitcher']
    traditional_horizon_params = cfg['traditional_horizon']
    marker_params              = cfg['marker']
    rtdetr_params              = cfg['rtdetr']
    mapper_params              = cfg['mapper']
    dis_estimator_params       = cfg['distance_estimator']
    ang_estimator_params       = cfg['angle_estimator']

    stitcher            = TripletStitcher(**stitcher_params)
    marker              = PanoHorizonAngleMarker(**marker_params)
    mapper              = BBoxToOriginalMapper(**mapper_params)
    dis_estimator       = BBoxDistanceEstimator(**dis_estimator_params)
    horizon_dis_est     = DetectionDistanceEstimator()
    ang_estimator       = HorizonX2AngleMapper(**ang_estimator_params)
    distance_fuser      = DistanceFuser()
    rtdetr_predictor    = RTDETRPredictor(
        model_path=rtdetr_params['model_path'],
        conf_thres=rtdetr_params['conf'],
        imgsz=rtdetr_params['imgsz']
    )

    # 若 TraditionalHorizonDetector 可重用（建議無狀態時採用）
    hdet = TraditionalHorizonDetector(**traditional_horizon_params)

    def detect_horizon_slope(img):
        # 回傳格式：{'slope': float, 'center': (x, y)}
        return hdet.detect_horizon_slope_and_center(img)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        while True:
            job = input_q.get()
            if job is None:
                break

            try:
                idx         = job['idx']
                global_M_L  = job['global_M_L']
                global_M_R  = job['global_M_R']

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

                # ===== 2) 平行提交：海平線偵測 與 三圖拼接 =====
                horizon_slope_futures = [executor.submit(detect_horizon_slope, img) for img in imgs]
                stitch_future         = executor.submit(stitcher.stitch, imgs, global_M_L, global_M_R)

                # 回收結果
                horizon_slope_pts = [f.result() for f in horizon_slope_futures]  # List[{'slope', 'center'}]
                pano, _, _        = stitch_future.result()

                # ===== 3) RT-DETR 偵測（於 pano） =====
                boxes, classes, scores = rtdetr_predictor.infer(pano)

                # 無偵測 → 直接標角度後回傳（空 marker_data）
                if boxes is None or len(boxes) == 0:
                    # pano_mask = marker.mark(pano, horizon_slope_pts, global_M_L, global_M_R, img_width)
                    output_q.put({
                        'idx': idx,
                        'raw_img': pano,
                        'marker_data': [],
                    })
                    continue

                # ===== 4) bbox 映射回原圖來源（保留 source/center 等欄位） =====
                mapped_detections = mapper.map_multiple(
                    boxes, classes, scores,
                    global_M_L, global_M_R,
                    img_height, img_width,
                    offset_w=img_width            # 顯式指定
                )

                # 映射後仍為空 → 直接回傳
                if not mapped_detections:
                    pano_mask = marker.mark(pano, horizon_slope_pts, global_M_L, global_M_R, img_width)
                    output_q.put({
                        'idx': idx,
                        'raw_img': pano_mask,
                        'marker_data': [],
                    })
                    continue

                # ===== 5) 估角（由海平線中點 x 與斜率推水平角）=====
                mapped_detections = ang_estimator.process_with_horizon_slope(
                    horizon_slope_pts, mapped_detections, global_M_L, global_M_R, img_width
                )
                # 期望增加欄位：'angle'（度）, 'slope', 'center'

                # ===== 6) 海平線法估距 =====
                mapped_detections = horizon_dis_est.horizon2distance(
                    mapped_detections, horizon_slope_pts
                )
                # 期望增加欄位：'distance_horizon'

                # ===== 7) BBox 高度法估距 =====
                mapped_detections = dis_estimator.estimate_distance(mapped_detections)
                # 期望增加欄位：'distance_bbox'

                # ===== 8) 融合距離（最終 distance）=====
                mapped_detections = distance_fuser.fuse(
                    mapped_detections, img_width, img_height
                )
                # 期望增加欄位：'distance'（融合後）

                # ===== 9) 繪製：角度標註 + 偵測框 =====
                # pano_mask = marker.mark(pano, horizon_slope_pts, global_M_L, global_M_R, img_width)
                # pano_mask = draw_bboxes_from_mapped(pano_mask, mapped_detections)
                pano_mask = draw_bboxes_from_mapped(pano, mapped_detections)
                

                # 若要 nvjpeg.encode，請於外層統一管理 encoder，這裡先略過
                # pano_mask_jpeg = nvjpeg.encode(pano_mask, 100)

                # ===== 10) 產出 marker_data（新版：class/distance/angle/bbox/scale/color）=====
                marker_data = convert_detections_to_pose_info(mapped_detections)
                # marker_data 示例：
                # {
                #     "class": "RedBall",
                #     "distance": 42.3,
                #     "angle": -13.5,
                #     "bbox": [x1, y1, x2, y2],
                #     "scale": (w, d, h),
                #     "color": (1.0, 0.0, 0.0),
                # }

                # ===== 11) 回傳 =====
                output_q.put({
                    'idx': idx,
                    'raw_img': pano_mask,
                    'marker_data': marker_data,
                })

            except Exception:
                traceback.print_exc()
                # 出錯也回傳，避免上游等待卡住
                output_q.put({'idx': job.get('idx', -1), 'raw_img': None, 'marker_data': []})