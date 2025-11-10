#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal")

import os
import cv2
import numpy as np
from time import time
from math import pi, atan

# ==============================
# TraditionalHorizonDetector
# ==============================
class TraditionalHorizonDetector:
    """快速海平線檢測器 - 基於邊緣檢測和霍夫變換的海平線檢測算法"""

    def __init__(self, init_all=True, canny_th1=25, canny_th2=45, Th_ROI=2, Th_slope=0.57,
                 N_c=15, N_d=200, D_Y_hl_th=50, D_alpha_hl_th=2, max_outliers_th=4,
                 hough_D_rho=2, hough_D_theta=pi/180, resize_factor=0.6,
                 roi_ratio=None):

        if init_all:
            self.resize_factor = resize_factor
            self.canny_th1, self.canny_th2 = canny_th1, canny_th2
            self.Th_ROI = Th_ROI * self.resize_factor
            self.Th_slope = Th_slope
            self.N_c, self.N_d, self.N_d_org = N_c, N_d, N_d
            self.DY_th = D_Y_hl_th
            self.Dphi_th = D_alpha_hl_th
            self.Nth_F_out = max_outliers_th
            self.hough_D_rho = hough_D_rho
            self.hough_D_theta = hough_D_theta

            if not hasattr(cv2, "ximgproc") or not hasattr(cv2.ximgproc, "createFastLineDetector"):
                raise RuntimeError("需要 opencv-contrib-python：cv2.ximgproc.createFastLineDetector 不存在。")
            self.fsd = cv2.ximgproc.createFastLineDetector(
                _canny_th1=self.canny_th1, _canny_th2=self.canny_th2
            )

            self.Y_prv = self.phi_prv = np.nan
            self.DY = self.Dphi = np.nan
            self.N_F_out = 0
            self.D_rho, self.D_theta = 1, 1 * (pi/180)
            self.roi_ratio = roi_ratio if roi_ratio is not None else [0.3, 0.7, 0.3, 0.7]

        self._reset_processing_variables()

    def _reset_processing_variables(self):
        for attr in ['Segs_a', 'Segs_b', 'Segs_c', 'Segs_d', 'Segs_e', 'Segs_f']:
            setattr(self, attr, None)
        for attr in ['Len_a', 'Len_b', 'Len_c', 'Len_d', 'Len_e', 'Len_f', 'Len_b_sort_idxs']:
            setattr(self, attr, None)
        coord_attrs = ['xs_a', 'ys_a', 'xe_a', 'ye_a', 'xs_b', 'ys_b', 'xe_b', 'ye_b',
                       'xs_c', 'ys_c', 'xe_c', 'ye_c', 'xs_d', 'ys_d', 'xe_d', 'ye_d',
                       'xs_f', 'ys_f', 'xe_f', 'ye_f', 'xs_hl', 'ys_hl', 'xe_hl', 'ye_hl']
        for attr in coord_attrs:
            setattr(self, attr, None)
        self.F_continue = self.F_det = True
        self.F_out = False
        self.Y = self.phi = self.theta = self.rho = self.latency = np.nan
        self.x_hl_mid = self.y_hl_mid = np.nan
        self.roi_bbox = None  # (x, y, rw, rh)

    def detect_horizon_slope_and_center(self, img, roi_ratio=None):
        """
        偵測圖像中的海平線斜率與中心點座標。
        回傳: {'slope': float, 'center': (x, y)}  # center 已回投到原圖中心
        """
        H, W = img.shape[:2]          # 原圖尺寸
        self.src_width = W            # 保存原圖尺寸供繪製使用
        self.src_height = H

        self.start_time = time()
        self._reset_processing_variables()
        self.N_d = self.N_d_org
        self.F_det = True
        roi_ratio = roi_ratio if roi_ratio is not None else self.roi_ratio

        # === ROI 裁切（內部只在 ROI 尺寸運算）===
        roi_offset_x = roi_offset_y = 0
        roi_w = W
        roi_h = H
        roi_img = img
        if roi_ratio is not None and len(roi_ratio) == 4:
            x0, x1, y0, y1 = roi_ratio
            x = int(W * x0); y = int(H * y0)
            rw = int(W * (x1 - x0)); rh = int(H * (y1 - y0))
            roi_offset_x, roi_offset_y = x, y
            roi_w, roi_h = rw, rh
            roi_img = img[y:y+rh, x:x+rw].copy()
            self.roi_bbox = (x, y, rw, rh)
        else:
            self.roi_bbox = (0, 0, W, H)

        # 內部尺寸用 ROI 尺寸
        self.org_width = roi_w
        self.org_height = roi_h
        self.res_width = int(roi_w * self.resize_factor)
        self.res_height = int(roi_h * self.resize_factor)

        try:
            self.get_horizon_edges(img=roi_img)
            self.hough_transform()             # 只在 ROI 尺寸做，不放大
            self.outlier_handler_module()
            self.linear_least_square_fitting() # 以 ROI 座標擬合
        except Exception:
            self.F_det = False

        valid = (
            self.F_det and
            hasattr(self, "xs_hl") and self.xs_hl is not None and
            abs(self.xe_hl - self.xs_hl) >= 1e-6
        )

        if not valid:
            x_center = W // 2
            y_center = H // 2
            self.end_time = time()
            self.latency = round((self.end_time - self.start_time), 4)
            return {'slope': 0.0, 'center': (x_center, y_center)}

        # 用 ROI 內擬合的 (a,b)，把原圖中心的 y 回投
        a = (self.ye_hl - self.ys_hl) / (self.xe_hl - self.xs_hl)
        b = self.ys_hl - a * self.xs_hl
        x_center = W // 2
        y_center = int(round(a * (x_center - roi_offset_x) + b + roi_offset_y))

        self.end_time = time()
        self.latency = round((self.end_time - self.start_time), 4)

        return {'slope': float(a), 'center': (x_center, y_center)}

    def get_horizon_edges(self, img):
        """海平線邊緣檢測 - 使用長度-斜率濾波器與 ROI 濾波器（只在 ROI 內部）"""
        self.x_out = self.y_out = None
        self.in_img_bgr = img                    # 這裡是 ROI 影像
        self.img_with_hl = self.in_img_bgr.copy()

        # === 圖像預處理 ===
        if self.resize_factor < 1:
            self.in_img_red = cv2.resize(self.in_img_bgr[:, :, 2],
                                         dsize=(self.res_width, self.res_height))
        else:
            self.in_img_red = self.in_img_bgr[:, :, 2]

        self.F_continue = True

        if self.N_c > self.N_d:
            raise ValueError("參數 N_c 必須小於 N_d")

        # === 線段檢測 ===
        self.Segs_a = self.fsd.detect(self.in_img_red)
        if self.Segs_a is None:
            self.x_out = self.y_out = None
            self.F_det = False
            return self.x_out, self.y_out

        # === 多階段濾波 ===
        self.lsf()   # 長度-斜率濾波
        if self.F_continue:
            self.roif()  # ROI 濾波（ROI 內部邏輯）
        self.step()  # 邊緣點提取

        return self.x_out, self.y_out

    def lsf(self):
        """長度-斜率濾波器 (Length-Slope Filter)"""
        self.N_a = self.Segs_a.shape[0]
        self.Segs_a = np.reshape(self.Segs_a, newshape=(self.N_a, 4))

        # 斜率濾波
        self.xs_a, self.ys_a = self.Segs_a[:, 0], self.Segs_a[:, 1]
        self.xe_a, self.ye_a = self.Segs_a[:, 2], self.Segs_a[:, 3]
        dx = np.subtract(self.xe_a, self.xs_a)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.alpha_a = np.divide(np.subtract(self.ye_a, self.ys_a), dx)
            self.alpha_a = np.where(np.isfinite(self.alpha_a), self.alpha_a, 0)

        self.b_from_a_idxs, = np.where(np.abs(self.alpha_a) < 0.58)
        self.Segs_b = self.Segs_a[self.b_from_a_idxs]
        self.N_b = self.Segs_b.shape[0]

        if self.N_b <= self.N_c:
            self.Segs_f = self.Segs_b
            self.F_continue = False
            return

        # 長度濾波
        self.xs_b, self.ys_b = self.Segs_b[:, 0], self.Segs_b[:, 1]
        self.xe_b, self.ye_b = self.Segs_b[:, 2], self.Segs_b[:, 3]
        self.Len_b = np.sqrt(np.add(np.square(np.subtract(self.xs_b, self.xe_b)),
                                    np.square(np.subtract(self.ys_b, self.ye_b))))

        self.Len_b_sort_idxs = np.flip(np.argsort(self.Len_b))
        self.c_from_b_idxs = self.Len_b_sort_idxs[0:self.N_c]
        self.Segs_c = self.Segs_b[self.c_from_b_idxs]
        self.d_from_b_idxs = self.Len_b_sort_idxs[self.N_c:self.N_c + self.N_d]
        self.Segs_d = self.Segs_b[self.d_from_b_idxs]
        self.N_d = self.Segs_d.shape[0]

    def roif(self):
        """ROI 濾波器（相對於 ROI 影像自身）"""
        self.xs_c, self.ys_c = self.Segs_c[:, 0], self.Segs_c[:, 1]
        self.xe_c, self.ye_c = self.Segs_c[:, 2], self.Segs_c[:, 3]
        self.alpha_c = self.alpha_a[self.b_from_a_idxs][self.c_from_b_idxs]
        self.B_c = np.subtract(self.ys_c, np.multiply(self.alpha_c, self.xs_c))
        self.B_c = np.broadcast_to(np.reshape(self.B_c, newshape=(self.N_c, 1)),
                                   shape=(self.N_c, self.N_d))

        self.xs_d, self.ys_d = self.Segs_d[:, 0], self.Segs_d[:, 1]
        self.xe_d, self.ye_d = self.Segs_d[:, 2], self.Segs_d[:, 3]
        self.Ys_d = np.broadcast_to(np.reshape(self.ys_d, newshape=(1, self.N_d)),
                                    shape=(self.N_c, self.N_d))
        self.Ye_d = np.broadcast_to(np.reshape(self.ye_d, newshape=(1, self.N_d)),
                                    shape=(self.N_c, self.N_d))

        self.alpha_c = np.reshape(self.alpha_c, newshape=(self.N_c, 1))

        self.DYs = np.abs(np.subtract(np.add(np.multiply(self.alpha_c, self.xs_d), self.B_c), self.Ys_d))
        self.DYe = np.abs(np.subtract(np.add(np.multiply(self.alpha_c, self.xe_d), self.B_c), self.Ye_d))

        self.Qs = np.less_equal(self.DYs, self.Th_ROI)
        self.Qe = np.less_equal(self.DYe, self.Th_ROI)
        self.Q = np.logical_and(self.Qs, self.Qe)
        self.q = np.any(self.Q, axis=0)

        self.e_from_d_idxs, = np.where(self.q == True)
        self.N_e = self.e_from_d_idxs.shape[0]

        if self.N_e > 0:
            self.Segs_e = self.Segs_d[self.e_from_d_idxs]
        else:
            self.Segs_e = np.zeros((0, 4))

        self.Segs_f = np.concatenate((self.Segs_c, self.Segs_e), axis=0)

    def step(self):
        """線段轉邊緣點 (Segment to Edge Points) —— 逐段產生，確保 x/y 對齊"""
        self.x_out = np.zeros((0,), dtype=np.float32)
        self.y_out = np.zeros((0,), dtype=np.float32)
        self.N_f = int(self.Segs_f.shape[0])

        if self.N_f == 0:
            self.F_det = False
            return

        self.xs_f, self.ys_f = self.Segs_f[:, 0], self.Segs_f[:, 1]
        self.xe_f, self.ye_f = self.Segs_f[:, 2], self.Segs_f[:, 3]

        if self.F_continue:
            self.Len_c = self.Len_b[self.c_from_b_idxs]
            self.Len_e = self.Len_b[self.d_from_b_idxs][self.e_from_d_idxs]
            self.Len_f = np.concatenate((self.Len_c, self.Len_e)).astype(np.float32)
        else:
            self.Len_f = np.sqrt(
                np.square(self.xs_f - self.xe_f) + np.square(self.ys_f - self.ye_f)
            ).astype(np.float32)

        self.Len_f = np.maximum(self.Len_f - 1.0, 1.0)

        for L, xs, ys, xe, ye in zip(self.Len_f, self.xs_f, self.ys_f, self.xe_f, self.ye_f):
            Ln = int(max(1, round(L)))
            u_n = np.arange(Ln, dtype=np.float32)
            x_n = xs + (xe - xs) * (u_n / L)
            y_n = ys + (ye - ys) * (u_n / L)

            self.x_out = np.concatenate((self.x_out, x_n))
            self.y_out = np.concatenate((self.y_out, y_n))

    def hough_transform(self):
        """霍夫變換檢測直線（只在 ROI 尺寸；不放大回原圖）"""
        if not self.F_det:
            return
        if self.x_out is None or self.y_out is None:
            self.F_det = False
            return

        x = np.asarray(self.x_out).astype(np.int32).ravel()
        y = np.asarray(self.y_out).astype(np.int32).ravel()
        n = min(x.shape[0], y.shape[0])
        if n == 0:
            self.F_det = False
            return
        x, y = x[:n], y[:n]

        h, w = self.in_img_red.shape[:2]   # 這裡是 ROI 尺寸（或縮小後）
        # 邊界過濾
        valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        if not np.any(valid):
            self.F_det = False
            return
        x, y = x[valid], y[valid]

        # 在 ROI 尺寸建立邊緣圖（不放大）
        self.img_edges = np.zeros((h, w), dtype=np.uint8)
        self.img_edges[y, x] = 255

        self.hough_lines = cv2.HoughLines(
            image=self.img_edges,
            rho=self.hough_D_rho,
            theta=self.hough_D_theta,
            threshold=2,
            min_theta=np.pi/3,
            max_theta=np.pi*2/3
        )

        if self.hough_lines is None or len(self.hough_lines) == 0:
            self.phi = self.Y = self.latency = np.nan
            self.F_det = False

    def linear_least_square_fitting(self):
        """線性最小二乘擬合（座標皆為 ROI 影像座標）"""
        if not self.F_det:
            return

        self.get_inlier_edges()
        self.inlier_edges_xy = np.zeros((self.inlier_edges_x.size, 2), dtype=np.int32)
        self.inlier_edges_xy[:, 0], self.inlier_edges_xy[:, 1] = self.inlier_edges_x, self.inlier_edges_y
        if self.inlier_edges_xy.shape[0] < 2:
            self.F_det = False
            return

        [vx, vy, x, y] = cv2.fitLine(points=self.inlier_edges_xy, distType=cv2.DIST_L2,
                                     param=0, reps=1, aeps=0.01)

        self.hl_slope = float(vy / vx)
        self.hl_intercept = float(y - self.hl_slope * x)

        # 在 ROI 影像寬度上取端點
        self.xs_hl = int(0)
        self.xe_hl = int(self.org_width - 1)  # org_width 現在是 ROI 寬
        self.ys_hl = int(self.hl_intercept)
        self.ye_hl = int((self.xe_hl * self.hl_slope) + self.hl_intercept)

        self.phi = (-atan(self.hl_slope)) * (180 / pi)
        self.Y = ((((self.org_width - 1) / 2) * self.hl_slope + self.hl_intercept))

    def get_inlier_edges(self):
        """獲取內點邊緣（ROI 尺寸）"""
        self.y_j, self.x_j = np.where(self.img_edges == 255)
        theta_p = self.theta + self.D_theta
        theta_n = self.theta - self.D_theta
        self.x_cte = 0.5 * (np.cos(theta_p) - np.cos(theta_n))
        self.y_cte = 0.5 * (np.sin(theta_p) - np.sin(theta_n))

        self.D_rho_j = np.abs(np.add(np.multiply(self.x_j, self.x_cte),
                                     np.multiply(self.y_j, self.y_cte)))
        self.D_rho_g = np.add(self.D_rho_j, self.D_rho)

        self.rho_j = np.add(np.multiply(self.x_j, np.cos(self.theta)),
                            np.multiply(self.y_j, np.sin(self.theta)))
        inlier_condition = np.logical_and(self.rho_j <= (self.rho + self.D_rho_g / 2),
                                          self.rho_j >= (self.rho - self.D_rho_g / 2))

        self.inlier_edges_indexes = np.where(inlier_condition)
        self.inlier_edges_x = self.x_j[self.inlier_edges_indexes]
        self.inlier_edges_y = self.y_j[self.inlier_edges_indexes]

    def outlier_handler_module(self):
        """異常值處理模塊"""
        if not self.F_det:
            return
        self.outlier_checker()
        self.outlier_replacer()
        self.failure_state_handler()

    def outlier_checker(self):
        """檢查最強 Hough 峰值是否為異常值（ROI 尺寸）"""
        self.F_out = False
        self.rho, self.theta = self.hough_lines[0][0]
        self.phi = ((np.pi / 2) - self.theta) * (180 / np.pi)
        self.img_width = self.img_edges.shape[1]

        s = np.sin(self.theta)
        if abs(s) < 1e-6:
            s = 1e-6
        self.Y = (self.rho - 0.5 * self.img_width * np.cos(self.theta)) / s

        self.DY = abs(self.Y - (self.Y_prv if np.isfinite(self.Y_prv) else self.Y))
        self.Dphi = abs(self.phi - (self.phi_prv if np.isfinite(self.phi_prv) else self.phi))
        self.F_out = (self.DY > self.DY_th) or (self.Dphi > self.Dphi_th)

    def outlier_replacer(self, M=2):
        """尋找替代的 Hough 候選（當判定為異常時；ROI 尺寸）"""
        if not self.F_out:
            return

        self.img_width = self.img_edges.shape[1]
        self.hough_lines_nbr = self.hough_lines.shape[0]
        self.hough_lines = np.reshape(self.hough_lines, newshape=(self.hough_lines_nbr, 2))

        if M == -1:
            self.rho_cands = self.hough_lines[1::][:, 0]
            self.theta_cands = self.hough_lines[1::][:, 1]
        else:
            M = min(M, self.hough_lines_nbr - 1)
            self.rho_cands = self.hough_lines[1:(M + 1)][:, 0]
            self.theta_cands = self.hough_lines[1:(M + 1)][:, 1]

        s = np.sin(self.theta_cands)
        s = np.where(np.abs(s) < 1e-6, 1e-6, s)
        self.phi_cands = np.subtract((np.pi / 2), self.theta_cands) * (180 / np.pi)
        self.Y_cands = (self.rho_cands - 0.5 * self.img_width * np.cos(self.theta_cands)) / s

        self.DY_cands = np.abs((np.subtract(self.Y_cands, self.Y_prv if np.isfinite(self.Y_prv) else self.Y_cands)))
        self.Dphi_cands = np.abs((np.subtract(self.phi_cands, self.phi_prv if np.isfinite(self.phi_prv) else self.phi_cands)))

        self.Isrt_cands = np.argsort(self.DY_cands)
        self.DYsrt_cands = self.DY_cands[self.Isrt_cands]
        self.Dphisrt_cands = self.Dphi_cands[self.Isrt_cands]
        self.Ysrt_cands = self.Y_cands[self.Isrt_cands]

        self.ISub = np.logical_and(self.DYsrt_cands < self.DY_th, self.Dphisrt_cands < self.Dphi_th)
        self.Isub = np.where(self.ISub == True)[0]
        self.sub_nbr = self.Isub.shape[0]

        if self.sub_nbr > 0:
            self.Isub = self.Isub[0]
            self.Y = self.Ysrt_cands[self.Isub]
            self.phi = self.Dphisrt_cands[self.Isub]  # 保留原樣（可視需要修正）
            self.rhosrt_cands = self.rho_cands[self.Isrt_cands]
            self.thetasrt_cands = self.theta_cands[self.Isrt_cands]
            self.rho = self.rhosrt_cands[self.Isub]
            self.theta = self.thetasrt_cands[self.Isub]
            self.theta = (90 - self.phi) * (pi / 180)
            self.rho = (self.Y * np.sin(self.theta)) + (0.5 * self.img_width * np.cos(self.theta))
        else:
            self.F_det = False

    def failure_state_handler(self):
        """處理連續異常情況"""
        if self.F_out:
            self.N_F_out += 1
            if self.N_F_out > self.Nth_F_out:
                self.Y_prv = np.nan
                self.phi_prv = np.nan
                self.N_F_out = 0
        else:
            self.N_F_out = 0


# ==============================
# 工具函式（繪圖與 ROI）
# ==============================
def draw_roi_rect(image, roi_ratio, color=(0, 255, 255), thickness=2):
    """在原圖上畫出 ROI 的矩形範圍（比例座標）。"""
    h, w = image.shape[:2]
    x0, x1, y0, y1 = roi_ratio
    x = int(w * x0); y = int(h * y0)
    rw = int(w * (x1 - x0)); rh = int(h * (y1 - y0))
    out = image.copy()
    cv2.rectangle(out, (x, y), (x + rw, y + rh), color, thickness, cv2.LINE_AA)
    return out

def overlay_edges_on_image(image_bgr, edges_gray, alpha=0.8):
    """
    將邊緣圖 (單通道) 疊加到提供的影像上（大小需相同；不縮放）。
    """
    h, w = image_bgr.shape[:2]
    eg = edges_gray
    if eg is None:
        return image_bgr.copy()
    if eg.shape[:2] != (h, w):
        raise ValueError("overlay_edges_on_image: edges_gray 尺寸與 image_bgr 不符，避免縮放。")
    edge_bgr = np.dstack([eg, eg, eg])
    edge_bgr = (edge_bgr > 0).astype(np.uint8) * 255
    out = cv2.addWeighted(image_bgr, 1.0, edge_bgr, alpha, 0)
    return out

def draw_horizon_on_image(image_bgr, slope, center_xy, color=(0, 0, 255), thickness=2):
    """
    依照 (slope, center) 在原圖上畫出海平線：
      line: y = a * x + b；其中 b = center_y - a * center_x
    """
    h, w = image_bgr.shape[:2]
    cx, cy = center_xy
    a = float(slope)
    b = float(cy - a * cx)

    # 與邊界求交，避免畫出界
    pts = []
    pts.append((0, int(round(b))))                    # x=0
    pts.append((w - 1, int(round(a * (w - 1) + b)))) # x=w-1
    if abs(a) > 1e-9:
        pts.append((int(round(-b / a)), 0))                   # y=0
        pts.append((int(round((h - 1 - b) / a)), h - 1))      # y=h-1
    in_pts = [(x, y) for (x, y) in pts if (0 <= x < w and 0 <= y < h)]

    if len(in_pts) < 2:
        x0, x1 = 0, w - 1
        y0 = int(round(a * x0 + b))
        y1 = int(round(a * x1 + b))
        p0 = (x0, max(0, min(h - 1, y0)))
        p1 = (x1, max(0, min(h - 1, y1)))
    else:
        p0, p1 = in_pts[0], None
        for q in in_pts[1:]:
            if q != p0:
                p1 = q
                break
        if p1 is None:
            x0, x1 = 0, w - 1
            y0 = int(round(a * x0 + b))
            y1 = int(round(a * x1 + b))
            p0 = (x0, max(0, min(h - 1, y0)))
            p1 = (x1, max(0, min(h - 1, y1)))

    out = image_bgr.copy()
    cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)
    cv2.circle(out, (int(cx), int(cy)), 4, (0, 255, 0), -1, cv2.LINE_AA)
    return out


# ==============================
# 批次處理：單張流程封裝
# ==============================
def process_one_image(img_path, out_dir, det, roi_ratio):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[警告] 讀不到圖片：{img_path}，跳過。")
        return

    t0 = time()
    result = det.detect_horizon_slope_and_center(img, roi_ratio=roi_ratio)
    t1 = time()
    slope  = float(result["slope"])
    center = tuple(result["center"])

    # 只在 ROI 內做邊緣疊加
    x, y, rw, rh = det.roi_bbox
    roi_bgr = img[y:y+rh, x:x+rw].copy()
    roi_edges = getattr(det, "img_edges", None)
    roi_edges_vis = overlay_edges_on_image(roi_bgr, roi_edges, alpha=0.8)

    edges_vis = img.copy()
    edges_vis[y:y+rh, x:x+rw] = roi_edges_vis
    edges_vis = draw_roi_rect(edges_vis, roi_ratio, color=(0, 255, 255), thickness=2)

    # 海平線畫在整張圖
    horizon_vis = draw_horizon_on_image(img, slope=slope, center_xy=center,
                                        color=(0, 0, 255), thickness=2)
    horizon_vis = draw_roi_rect(horizon_vis, roi_ratio, color=(0, 255, 255), thickness=2)
    info_text = f"slope={slope:.4f}, center=({center[0]}, {center[1]})"
    cv2.putText(horizon_vis, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 50), 2, cv2.LINE_AA)

    # 存檔
    out_edges_full = os.path.join(out_dir, f"{stem}_edges.png")
    out_edges_roi  = os.path.join(out_dir, f"{stem}_roi_edges.png")
    out_hl         = os.path.join(out_dir, f"{stem}_horizon.png")
    cv2.imwrite(out_edges_full, edges_vis)
    cv2.imwrite(out_edges_roi, roi_edges_vis)  # 純 ROI 畫面
    cv2.imwrite(out_hl, horizon_vis)

    print(f"[完成] {os.path.basename(img_path)} | slope={slope:.4f}, center={center}, time={(t1-t0):.3f}s")
    print(f"  -> 邊緣特徵圖（原圖，僅 ROI 區塊）：{out_edges_full}")
    print(f"  -> 邊緣特徵圖（純 ROI 區域）：{out_edges_roi}")
    print(f"  -> 海平線結果圖：{out_hl}")


# ==============================
# 主程式：資料夾批次
# ==============================
def main():
    # ===== 參數設定 =====
    image_dir  = "images/test"             # 輸入資料夾
    out_dir    = "images/output2"           # 輸出資料夾
    roi_ratio     = [0.3, 0.7, 0.3, 0.7]   # [x0,x1,y0,y1]，0~1
    resize_factor = 1.0
    canny_th1     = 25
    canny_th2     = 45
    # ====================

    os.makedirs(out_dir, exist_ok=True)

    # 建立偵測器（重用模型、避免每張重建）
    det = TraditionalHorizonDetector(
        init_all=True,
        canny_th1=canny_th1,
        canny_th2=canny_th2,
        resize_factor=resize_factor,
        roi_ratio=roi_ratio
    )

    # 收集圖片清單
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
             if f.lower().endswith(exts)]
    if not files:
        raise FileNotFoundError(f"資料夾 {image_dir} 沒有圖片！")

    # 逐檔處理
    for img_path in sorted(files):
        process_one_image(img_path, out_dir, det, roi_ratio)


if __name__ == "__main__":
    main()
