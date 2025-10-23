# -*- coding: utf-8 -*-
"""
PanoHorizonMarker 類別
--------------------------

用途：
  在全景圖上標註「距離標記點」與「對應角度文字」，例如：500m、1000m、3000m，以及相機拍攝方向（如 -60deg、0deg、60deg）。

應用場景：
  結合距離投影模型後，將估算出的距離座標在 pano 圖上繪製，並可視覺化顯示角度資訊，協助分析航拍視角與距離感知。

初始化參數說明：
  - colors:          [左,中,右] 三台相機標記點顏色
  - angle_colors:    角度文字標記的顏色（預設與 colors 相同）
  - radius:          標記點的半徑（像素）
  - thickness:       圓點與文字線寬
  - font_scale:      文字縮放大小（OpenCV 標準）
  - text_dx/dy:      文字與點的位移（像素）
  - crop_rect:       若 pano 圖經過裁切，需傳入 (x, y, w, h)
  - camera_angle:    左中右相機的對應角度（預設 -60/0/60）
  - D_horizon:       想標示「海平線距離」的那個點（如 5000m）
  - angle_label_offset: 角度文字相對於點的垂直偏移（向上位移）

主要方法：
  - warp_point(pt, M): 將單點套用仿射或單應矩陣轉換
  - mark(pano, horizon_pts, M_L, M_R, offset): 在 pano 圖上畫出所有距離與角度標註

輸入輸出：
  - pano:           原始 pano 圖（np.array）
  - horizon_pts:    List of 3 groups，每組為該相機的一組 [(x, y, D), ...]
  - M_L/M_R:        左右相機的 pano ← 原圖 仿射或單應矩陣
  - offset:         拼接畫面時，每張圖佔寬度（單位：像素）

回傳：
  - pano_mask:      繪製標記後的 pano 圖
"""

import numpy as np
import cv2

class PanoHorizonMarker:
    def __init__(
        self,
        colors=[(255,255,0), (255,255,0), (255,255,0)],
        angle_colors=[(255,0,255), (255,0,255), (255,0,255)],  # 新增：三個角度文字顏色
        radius=1,
        thickness=1,
        font_scale=0.4,
        text_dx=4,
        text_dy=2,
        crop_rect=None,
        camera_angle=[-60,0,60],
        D_horizon=5000,
        angle_label_offset=16,
    ):
        self.colors = colors
        self.angle_colors = angle_colors if angle_colors is not None else colors
        self.radius = radius
        self.thickness = thickness
        self.font_scale = font_scale
        self.text_dx = text_dx
        self.text_dy = text_dy
        self.crop_rect = crop_rect
        self.camera_angle = camera_angle
        self.D_horizon = D_horizon
        self.angle_label_offset = angle_label_offset

    def warp_point(self, pt, M):
        """
        將點套用仿射或單應矩陣做座標轉換（支援 2x3 / 3x3）
        """
        if M is None or pt is None or np.any(np.isnan(pt)):
            print("[PanoHorizonMarker] Invalid point or matrix for warping.")
            return None
        px, py = pt
        p = np.array([px, py, 1.0], dtype=np.float32)
        if M.shape == (2, 3):
            x_warp = M[0, 0] * px + M[0, 1] * py + M[0, 2]
            y_warp = M[1, 0] * px + M[1, 1] * py + M[1, 2]
            return (int(round(x_warp)), int(round(y_warp)))
        elif M.shape == (3, 3):
            p_warp = M @ p
            if abs(p_warp[2]) < 1e-6:
                return None
            return (int(round(p_warp[0] / p_warp[2])), int(round(p_warp[1] / p_warp[2])))
        else:
            return None

    def mark(self, pano, horizon_pts, M_L, M_R, offset):
        """
        在 pano 圖上畫出距離點與角度標註

        輸入：
            pano: pano 全景圖
            horizon_pts: List[3]，每台相機的一組 [(x, y, D), ...]
            M_L/M_R: 左右影像 pano ← 原圖 的仿射或單應矩陣
            offset: 每張相機圖的寬度偏移（畫 pano 時的位置）

        輸出：
            pano_mask: 標記過後的 pano 圖（np.array）
        """
        assert len(horizon_pts) == 3
        pano_mask = pano.copy()
        h, w_img = pano.shape[:2]
        w = offset

        crop_x = crop_y = 0
        if self.crop_rect is not None:
            crop_x, crop_y = self.crop_rect[:2]

        mid_points = [None, None, None]
        for cam_idx, (points, color) in enumerate(zip(horizon_pts, self.colors)):
            for pt in points:
                if pt is None or len(pt) != 3 or np.any(np.isnan(pt)):
                    continue
                x, y, D = pt
                if cam_idx == 0:
                    mapped = self.warp_point((x, y), M_L)
                    if mapped is not None:
                        mapped = (mapped[0] + w, mapped[1])
                elif cam_idx == 1:
                    mapped = (int(round(x)) + w, int(round(y)))
                else:
                    mapped = self.warp_point((x, y), M_R)
                    if mapped is not None:
                        mapped = (mapped[0] + w, mapped[1])
                if mapped is None or np.any(np.isnan(mapped)):
                    continue
                pt_shifted = (mapped[0] - crop_x, mapped[1] - crop_y)
                if not (0 <= pt_shifted[0] < w_img and 0 <= pt_shifted[1] < h):
                    continue

                # 標記距離點
                cv2.circle(pano_mask, pt_shifted, self.radius, color, self.thickness)
                text = f"{int(D)}m" if abs(D - round(D)) < 1e-2 else f"{D:.1f}m"
                text_pos = (
                    pt_shifted[0] + self.radius + self.text_dx,
                    pt_shifted[1] + self.text_dy,
                )
                cv2.putText(
                    pano_mask,
                    text,
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale,
                    color,
                    self.thickness,
                    lineType=cv2.LINE_AA,
                )

                # 紀錄最接近 D_horizon 的點，作為角度文字位置
                if abs(D - self.D_horizon) < 1e-2 and mid_points[cam_idx] is None:
                    mid_points[cam_idx] = pt_shifted
            # 如果沒有精準對到 D_horizon，就找最近的那個點
            if mid_points[cam_idx] is None and points:
                idx = np.argmin([abs(D - self.D_horizon) for (_, _, D) in points])
                x, y, D = points[idx]
                if cam_idx == 0:
                    mapped = self.warp_point((x, y), M_L)
                    if mapped is not None:
                        mapped = (mapped[0] + w, mapped[1])
                elif cam_idx == 1:
                    mapped = (int(round(x)) + w, int(round(y)))
                else:
                    mapped = self.warp_point((x, y), M_R)
                    if mapped is not None:
                        mapped = (mapped[0] + w, mapped[1])
                if mapped is not None and not np.any(np.isnan(mapped)):
                    pt_shifted = (mapped[0] - crop_x, mapped[1] - crop_y)
                    if 0 <= pt_shifted[0] < w_img and 0 <= pt_shifted[1] < h:
                        mid_points[cam_idx] = pt_shifted

        # ====== 在 horizon 標記各相機的拍攝角度 ======
        for cam_idx, (pt, angle) in enumerate(zip(mid_points, self.camera_angle)):
            if pt is not None:
                angle_text = f"{angle}deg"
                angle_pos = (pt[0], pt[1] - self.radius - self.angle_label_offset)
                cv2.putText(
                    pano_mask,
                    angle_text,
                    angle_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale,
                    self.angle_colors[cam_idx],
                    self.thickness,
                    lineType=cv2.LINE_AA,
                )

        return pano_mask
