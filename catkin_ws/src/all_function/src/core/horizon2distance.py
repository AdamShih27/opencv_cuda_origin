import numpy as np
from typing import List, Dict, Optional

class DetectionDistanceEstimator:
    def __init__(
        self,
        camera_height: float = 3.8,
        fy: float = 250,
        debug_print: bool = False,
        min_perp_px: float = 5.0,   # ← 新增：像素垂直距離門檻（預設 20 px）
    ):
        """
        Args:
            camera_height: 相機高度（公尺）
            fy: y 軸焦距（像素，請對應目前影像尺寸）
            debug_print: 是否列印 debug
            min_perp_px: 正交像素距離低於此值 → 視為不穩定，回傳 None
        """
        self.camera_height = float(camera_height)
        self.fy = float(fy)
        self.debug_print = bool(debug_print)
        self.min_perp_px = float(min_perp_px)

    def _compute_distance_from_horizon(
        self,
        x_obj: float,
        y_obj: float,
        slope_m: float,
        center_xy: tuple,
    ) -> Optional[float]:
        """
        使用海平線（可能有 roll，y = m x + b）的「正交距離」計算物點距離。
        回傳距離（公尺）或 None。
        """
        cx, cy = center_xy
        m = float(slope_m)
        b = cy - m * cx             # y = m x + b

        # 同一 x 下的海平線 y 值
        v_h = m * x_obj + b
        delta_v = y_obj - v_h       # 物點相對海平線的垂直差（未旋正）

        # 物點在海平線上方（或剛好在海平線上）→ 不可估距
        if delta_v <= 0:
            return None

        # 正交距離（像素）
        d_perp = abs(delta_v) / np.sqrt(1.0 + m * m)

        # ★ 新增：像素門檻（小於 20 px 視為不穩定）
        if d_perp < self.min_perp_px:
            return None

        # 邊界保護：避免非常小的像素差導致無限大（理論上前面已擋掉）
        eps = 1e-6
        d_perp = max(d_perp, eps)

        # 直接用 Z = h * fy / d_perp
        distance = self.camera_height * self.fy / d_perp
        return float(distance)

    def horizon2distance(
        self,
        mapped_detections: List[Dict],
        horizon_slope_pts: List[Dict],  # [{'slope': float, 'center': (x, y)}]
    ) -> List[Dict]:
        """
        以 horizon_slope_pts（含斜率）為海平線，對每個 detection 標註距離（key: 'distance_horizon'）
        支援有 roll/pitch（海平線為斜線）的情境。
        """
        results = []

        for det in mapped_detections:
            det_new = det.copy()

            # 來源相機索引（0=left, 1=center, 2=right ...）
            src = det.get("src", det.get("source", None))
            if src is None or not (0 <= src < len(horizon_slope_pts)):
                det_new["distance_horizon"] = None
                results.append(det_new)
                continue

            # 取點位（建議：接地點；如果沒有就用 det["original"]）
            ori = det.get("original", {})
            x_obj = float(ori.get("x", np.nan))
            y_obj = float(ori.get("y", np.nan))

            if np.isnan(x_obj) or np.isnan(y_obj):
                det_new["distance_horizon"] = None
                results.append(det_new)
                continue

            # 海平線參數（僅計算用，不存截距）
            hinfo = horizon_slope_pts[src]
            m = float(hinfo["slope"])
            cx, cy = hinfo["center"]  # (cx, cy)
            b = cy - m * cx           # 內部用來算 v_h
            v_h = m * x_obj + b       # 在 x_obj 位置上的海平線 y

            # 距離（公尺）
            distance = self._compute_distance_from_horizon(x_obj, y_obj, m, (cx, cy))
            det_new["distance_horizon"] = distance

            # 供 fuser 使用（不含任何截距儲存）
            det_new["horizon_slope"] = m
            det_new["horizon_point"] = {"x": x_obj, "y": v_h}
            det_new["horizon_center"] = (cx, cy)

            # Debug 列印
            if self.debug_print:
                if distance is None:
                    # 統一把 d_perp 算一下，只為了 debug 顯示（不影響 None 結果）
                    d_perp_dbg = abs(y_obj - v_h) / np.sqrt(1.0 + m*m)
                    reason = "above_or_on_horizon" if (y_obj - v_h) <= 0 else (
                        f"d_perp<{self.min_perp_px:.1f}px"
                        if d_perp_dbg < self.min_perp_px else "unknown"
                    )
                    print(f"[{det_new.get('class','unknown')}] obj=({x_obj:.1f},{y_obj:.1f}) "
                          f"horizon=({x_obj:.1f},{v_h:.1f}) m={m:.6f} -> distance=None ({reason})")
                else:
                    d_perp = abs(y_obj - v_h) / np.sqrt(1.0 + m*m)
                    print(f"[{det_new.get('class','unknown')}] obj=({x_obj:.1f},{y_obj:.1f}) "
                          f"horizon=({x_obj:.1f},{v_h:.1f}) m={m:.6f} d_perp={d_perp:.3f}px dist={distance:.3f}m")

            results.append(det_new)

        return results
