import numpy as np
from typing import List, Dict, Tuple, Optional

class BBoxToOriginalMapper:
    def __init__(self, crop_rect,
                 focals: Optional[Tuple[float, float, float]] = None):
        """
        初始化裁切資訊，並推定 pano 全寬度
        crop_rect: (x, y, w, h)
        focals 若提供，map_bbox/map_multiple 傳入 h,w 時會反圓柱化回原圖座標
        """
        self.crop_rect = crop_rect
        if crop_rect is None or len(crop_rect) < 4:
            raise ValueError("crop_rect 應為 (x, y, w, h) 格式")
        self.crop_x, self.crop_y, self.pano_width, _ = crop_rect
        self.focals = focals  # (f_L, f_M, f_R) 或 None

    def _prepare_matrix(self, M):
        if M is None:
            return None
        if M.shape == (2, 3):
            return np.vstack([M, [0, 0, 1]])
        elif M.shape == (3, 3):
            return M
        return None

    def _inverse_project(self, pt, M):
        try:
            M_inv = np.linalg.inv(M)
            p = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
            src_p = M_inv @ p
            if abs(src_p[2]) < 1e-6:
                return None
            return (src_p[0] / src_p[2], src_p[1] / src_p[2])
        except np.linalg.LinAlgError:
            return None

    def _inverse_cylindrical(self, u: float, v: float, f: float, h: int, w: int):
        """
        反圓柱化：圓柱面(u,v) -> 原圖(x,y)
        正向：u = f*tan((x-w/2)/f) + w/2
            v = (y - h/2) / cos((x-w/2)/f) + h/2
        反向：theta = atan((u - w/2) / f)
            x = f*theta + w/2
            y = (v - h/2) * cos(theta) + h/2
        """
        theta = float(np.arctan((u - w / 2.0) / float(f)))
        c = float(np.cos(theta))
        x = float(f * theta + w / 2.0)
        y = float((v - h / 2.0) * c + h / 2.0)
        return (x, y)

    # ===== 這個函式維持舊介面：仍需要 offset_w =====
    def map_bbox(self,
                 bbox: List[int],
                 offset_w: float,
                 M_L: np.ndarray,
                 M_R: np.ndarray,
                 h: Optional[int] = None,
                 w: Optional[int] = None) -> Tuple[float, float, int]:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = y2  # 下緣中點

        pano_x = cx + self.crop_x
        pano_y = cy + self.crop_y
        pano_x_adj = pano_x - offset_w

        region_w = self.pano_width / 3.0

        M_L = self._prepare_matrix(M_L)
        M_R = self._prepare_matrix(M_R)

        if pano_x_adj < region_w:
            mapped = self._inverse_project((pano_x_adj, pano_y), M_L)
            src = 0
        elif pano_x_adj < 2 * region_w:
            mapped = (pano_x_adj, pano_y)
            src = 1
        else:
            mapped = self._inverse_project((pano_x_adj, pano_y), M_R)
            src = 2

        if mapped is None:
            return None

        # 若提供了焦距與 h,w，做反圓柱化回原圖
        if self.focals is not None and (h is not None) and (w is not None):
            f = self.focals[src]
            inv_xy = self._inverse_cylindrical(mapped[0], mapped[1], f, h, w)
            if inv_xy is None:
                return None
            
            # print(f"Inverse cylindrical: {mapped} → {inv_xy} (src={src})")

            return inv_xy[0], inv_xy[1], src

        # 否則維持回傳圓柱面座標
        return mapped[0], mapped[1], src

    # ===== 這個函式改良：offset_w 可省略；預設用 w，否則退回 pano_width/3 =====
    def map_multiple(self,
                     boxes: List[List[int]],
                     classes: List[str],
                     scores: List[float],
                     M_L: np.ndarray,
                     M_R: np.ndarray,
                     h: Optional[int] = None,
                     w: Optional[int] = None,
                     offset_w: Optional[float] = None) -> List[Dict]:
        """
        Args:
            boxes, classes, scores: 偵測結果
            M_L, M_R: 左/右影像仿射/單應矩陣
            h, w:     原圖高度與寬度（若有傳且 self.focals 非 None，則做反圓柱化）
            offset_w: 中圖在 pano 的水平偏移；若省略則：
                      - 有提供 w 時 → 預設 offset_w = w
                      - 沒提供 w   → 預設 offset_w = pano_width / 3
        """
        # 自動決定 offset_w
        if offset_w is None:
            offset_w = float(w) if (w is not None) else (self.pano_width / 3.0)

        results = []
        for box, cls, score in zip(boxes, classes, scores):
            mapped = self.map_bbox(box, offset_w, M_L, M_R, h=h, w=w)
            if mapped is None:
                continue
            x_ori, y_ori, src = mapped
            results.append({
                "class": cls,
                "score": score,
                "bbox": box,
                "original": {"x": x_ori, "y": y_ori},
                "source": src
            })
        return results