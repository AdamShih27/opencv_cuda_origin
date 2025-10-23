import numpy as np
from copy import deepcopy

class DistanceFuser:
    """
    根據 bbox 信心分數、位置，以及海平線點位置與斜率，融合 bbox 法與海平線法的距離推估結果。

    Parameters:
        score_weight_min       # BBox 信心分數最低時給的最小權重（0~1之間）
        score_weight_max       # BBox 信心分數最高時給的最大權重（0~1之間）
        pos_weight_min         # 畫面最邊緣時海平線法最低權重（0~1之間）
        pos_weight_max         # 畫面中心時海平線法的最大權重（0~1之間）
        center_ratio           # 多少比例畫面寬度內算作中心區（如0.6=中間60%）
        bbox_height_threshold  # BBox高度小於此值視為異常，降低權重
        horizon_outlier_pixel_ratio # 海平線點離中心超過此比例視為異常（以最短邊比例）
        horizon_slope_threshold    # 海平線斜率絕對值超過此值視為異常
        print_debug            # 是否在融合時印出三個距離與權重；若權重為 0 也會印出原因
        disagree_threshold_m   # 兩法距離差異超過此門檻（公尺）時，以 BBox 為主，Horizon 權重=0
    """
    def __init__(self,
                 score_weight_min=0.5,
                 score_weight_max=1.0,
                 pos_weight_min=0.1,
                 pos_weight_max=0.5,
                 center_ratio=0.6,
                 bbox_height_threshold=2,
                 horizon_outlier_pixel_ratio=0.2,
                 horizon_slope_threshold=0.2,
                 print_debug=False,
                 disagree_threshold_m=100.0):
        self.score_weight_min = score_weight_min
        self.score_weight_max = score_weight_max
        self.pos_weight_min = pos_weight_min
        self.pos_weight_max = pos_weight_max
        self.center_ratio = center_ratio
        self.bbox_height_threshold = bbox_height_threshold
        self.horizon_outlier_pixel_ratio = horizon_outlier_pixel_ratio
        self.horizon_slope_threshold = horizon_slope_threshold
        self.print_debug = print_debug
        self.disagree_threshold_m = float(disagree_threshold_m)

    @staticmethod
    def _is_valid(x):
        return (x is not None) and np.isfinite(x)

    def fuse(self, mapped_detections, img_width, img_height):
        """
        Args:
            mapped_detections: list，每個 dict 含：
                "class", "score", "bbox", "original", "src", "angle",
                "distance_bbox", "distance_horizon", "horizon_point", "horizon_slope"
            img_width:  int，圖片寬度
            img_height: int，圖片高度
        Returns:
            與 mapped_detections 同長度的 list，每個 dict 會新增：
                - "distance": 融合後距離
                - "distance_bbox_val": 高度法距離（回存，便於使用）
                - "distance_horizon_val": 海平線法距離（回存，便於使用）
                - "fusion_weights": {"wb": 加權後bbox權重, "wh": 加權後horizon權重}
        """
        results = []
        center_x = img_width / 2.0
        center_y = img_height / 2.0
        center_half = img_width * self.center_ratio / 2.0
        horizon_outlier_pixel_thresh = min(img_width, img_height) * self.horizon_outlier_pixel_ratio

        for det in mapped_detections:
            cls_name = det.get('class', 'unknown')
            score = float(det.get('score', 0.0))
            bbox = det.get('bbox', [0, 0, 0, 0])
            x_ori = float(det.get('original', {}).get('x', center_x))

            # ---- BBox 基礎權重（由 score 映射）----
            w_bbox = self.score_weight_min + (self.score_weight_max - self.score_weight_min) * np.clip(score, 0.0, 1.0)
            reasons_bbox = []

            # ---- Horizon 基礎權重（由水平位置 gate）----
            dx = abs(x_ori - center_x)
            if dx <= center_half:
                w_horizon = self.pos_weight_max
                horizon_base_reason = "in_center"
            else:
                w_horizon = 0.0
                horizon_base_reason = "out_of_center"
            reasons_horizon = [horizon_base_reason]

            # ---- BBox 異常：高度太小 → 降權 ----
            bbox_h = float(bbox[3] - bbox[1])
            if bbox_h < self.bbox_height_threshold:
                w_bbox *= 0.1
                reasons_bbox.append(f"small_bbox_height<{self.bbox_height_threshold}")

            # ---- Horizon 異常：horizon_point 距中心太遠 or 缺失 → 降權 ----
            horizon_point = det.get("horizon_point", {})
            x_h = horizon_point.get("x", None)
            y_h = horizon_point.get("y", None)
            if self._is_valid(x_h) and self._is_valid(y_h):
                d_horizon_center = float(np.hypot(float(x_h) - center_x, float(y_h) - center_y))
                if d_horizon_center > horizon_outlier_pixel_thresh:
                    w_horizon *= 0.1
                    reasons_horizon.append(
                        f"horizon_point_far>{horizon_outlier_pixel_thresh:.1f}px (d={d_horizon_center:.1f})"
                    )
            else:
                w_horizon *= 0.1
                reasons_horizon.append("missing_horizon_point")

            # ---- Horizon 異常：斜率異常 → 再降權 ----
            slope = det.get("horizon_slope", None)
            if (slope is None) or (not np.isfinite(slope)) or (abs(float(slope)) > self.horizon_slope_threshold):
                w_horizon *= 0.1
                reasons_horizon.append(f"slope_anomaly(|m|>{self.horizon_slope_threshold})")

            # === 取得兩法距離 ===
            D1 = det.get("distance_bbox", None)      # 高度法距離（m）
            D2 = det.get("distance_horizon", None)   # 海平線法距離（m）

            # 來源無效→權重歸零；有效→沿用上面計算的權重
            w_bbox_eff    = w_bbox    if self._is_valid(D1) else 0.0
            w_horizon_eff = w_horizon if self._is_valid(D2) else 0.0

            if not self._is_valid(D1):
                reasons_bbox.append("invalid_distance_bbox")
            if not self._is_valid(D2):
                reasons_horizon.append("invalid_distance_horizon")

            # === 新的限制：兩法差異過大時，以 BBox 為主，Horizon 權重 = 0 ===
            if self._is_valid(D1) and self._is_valid(D2):
                d1_val = float(D1)
                d2_val = float(D2)
                diff = abs(d1_val - d2_val)
                if diff > self.disagree_threshold_m:
                    # 直接把 horizon 的有效權重歸零（不動距離值本身）
                    w_horizon_eff = 0.0
                    reasons_horizon.append(
                        f"large_disagreement(|Δ|>{self.disagree_threshold_m:.1f}m, Δ={diff:.1f}m) -> wh=0"
                    )

            # === 融合（正規化 + 缺值保護）===
            sum_w = w_bbox_eff + w_horizon_eff
            eps = 1e-8

            if sum_w <= eps:
                D_fused = None
                wb = 0.0
                wh = 0.0
            else:
                wb = w_bbox_eff / (sum_w + eps)
                wh = w_horizon_eff / (sum_w + eps)
                d1 = float(D1) if self._is_valid(D1) else 0.0
                d2 = float(D2) if self._is_valid(D2) else 0.0
                D_fused = wb * d1 + wh * d2

            # === 輸出與記錄 ===
            det_out = deepcopy(det)
            det_out["distance"] = D_fused
            det_out["distance_bbox_val"] = (float(D1) if self._is_valid(D1) else None)
            det_out["distance_horizon_val"] = (float(D2) if self._is_valid(D2) else None)
            det_out["fusion_weights"] = {"wb": float(wb), "wh": float(wh)}
            results.append(det_out)

            # === Debug 印出 ===
            if self.print_debug:
                print(f"[{cls_name}] "
                      f"BBox: {det_out['distance_bbox_val']}, "
                      f"Horizon: {det_out['distance_horizon_val']}, "
                      f"Fused: {det_out['distance']}")
        return results
