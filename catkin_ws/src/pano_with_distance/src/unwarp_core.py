# -*- coding: utf-8 -*-
"""
Pano2Horizon 類別
------------------
用途：
  提供全景圖上點位對應回原始相機影像座標的反推功能，支援仿射矩陣 (2x3) 或單應矩陣 (3x3)。
  通常用於全景圖上畫出物體（如水平線、距離標記）後，反推其對應原始三張相機圖像的點。

初始化參數：
  crop_rect : (x, y, w, h) 若全景圖經過裁切，需傳入裁切區域
  pano_offset : 在拼接過程中，pano 起始位置的水平偏移（像素）

屬性：
  pano_width : 全景圖寬度，從 crop_rect[2] 推得

主要方法：
  unwarp_point(pano_pt, M) → (x, y)
    反推 pano 圖上的一點 pano_pt，透過仿射/單應矩陣 M，映射回原始圖上的點

  pano_to_original(pt_shifted, M_L, M_R) → (x, y, idx)
    將已裁切過的 pano 點 pt_shifted 轉回原圖座標
    自動判斷屬於左中右哪一張相機，對左/右圖使用矩陣還原，回傳對應圖上的 (x, y) 和圖像編號 idx（0 左, 1 中, 2 右）
"""

import numpy as np
import cv2

class Pano2Horizon:
    def __init__(
        self,
        crop_rect=None,   # pano 圖有經過裁切時的 (x, y, w, h)
        pano_offset=0,    # 每張相機拼接時的寬度 offset
    ):
        self.crop_rect = crop_rect
        self.pano_offset = pano_offset
        # pano_width 直接由 crop_rect[2] 取得
        self.pano_width = None
        if crop_rect is not None and len(crop_rect) >= 3:
            self.pano_width = crop_rect[2]

    def unwarp_point(self, pano_pt, M):
        """
        反推 pano 點到原圖座標（支援 2x3 / 3x3 仿射/單應）

        輸入:
            pano_pt: tuple(float, float)   pano圖上的(x, y)
            M: 2x3 or 3x3 numpy.ndarray    pano <- 原圖 的矩陣
        輸出:
            tuple(float, float)  原圖座標，失敗回傳 None
        """
        if M is None or pano_pt is None or np.any(np.isnan(pano_pt)):
            print("[Pano2Horizon] Invalid point or matrix for unwarping.")
            return None
        x, y = pano_pt
        if M.shape == (2, 3):
            M33 = np.vstack([M, [0, 0, 1]])
        elif M.shape == (3, 3):
            M33 = M
        else:
            print("[Pano2Horizon] Matrix shape not supported.")
            return None
        try:
            M_inv = np.linalg.inv(M33)
        except np.linalg.LinAlgError:
            print("[Pano2Horizon] Matrix not invertible.")
            return None
        p = np.array([x, y, 1.0], dtype=np.float32)
        src_p = M_inv @ p
        if abs(src_p[2]) < 1e-6:
            return None
        return (src_p[0] / src_p[2], src_p[1] / src_p[2])

    def pano_to_original(self, pt_shifted, M_L, M_R):
        """
        pano 上的點還原到原圖座標，自動依 pano_x 判斷使用哪個相機，並回傳原圖 index

        輸入:
            pt_shifted: tuple(float, float)  已 -crop 的 pano 點座標
            M_L, M_R: numpy.ndarray   左/右相機的仿射/單應矩陣
        輸出:
            (x, y, idx)  # idx: 0=左, 1=中, 2=右
        """
        if self.pano_width is None:
            raise ValueError("pano_width 尚未設定，請在 crop_rect 傳入 [x, y, w, h] 格式")
        crop_x, crop_y = 0, 0
        if self.crop_rect is not None:
            crop_x, crop_y = self.crop_rect[:2]
        # 1. 加回裁切
        pano_x, pano_y = pt_shifted[0] + crop_x, pt_shifted[1] + crop_y
        # 2. 減掉 offset
        pano_x_adj = pano_x - self.pano_offset

        # 3. 依位置自動判斷相機區間（左/中/右）
        w = self.pano_width / 3.0
        if pano_x_adj < w:
            mapped = self.unwarp_point((pano_x_adj, pano_y), M_L)
            idx = 0
        elif pano_x_adj < 2 * w:
            mapped = (pano_x_adj, pano_y)
            idx = 1
        else:
            mapped = self.unwarp_point((pano_x_adj, pano_y), M_R)
            idx = 2
        if mapped is None:
            return None
        return mapped[0], mapped[1], idx
