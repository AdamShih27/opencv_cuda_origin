# -*- coding: utf-8 -*-
"""
FastHorizon 類別
----------------------------

用途：
    快速海平線檢測器，基於長線段檢測（Fast Line Detector）+ 多階段濾波 + 霍夫變換 + 最小二乘擬合。
    可擴展輸出視覺海平線 + 投影距離刻度點，並支援定量誤差分析與影片展示。

應用場景：
    - 無人載具水面導航中的地平線偵測
    - 搭配距離投影模型標記畫面尺度（例：距離為 500m、1000m 的位置）
    - 支援標準影片輸入、GT 標註資料分析與可視化

初始化參數重點：
    - canny_th1, canny_th2：邊緣檢測門檻
    - N_c, N_d：用於濾波的線段選擇數量
    - D_Y_hl_th, D_alpha_hl_th：前後幀 Y/角度變化容忍度
    - mark_distances：標記距離（例如 [0, 100, 500, 1000, 2000, 5000]）
    - alpha_intervals：控制非線性尺度轉換的指數項

主要功能：
    ✅ get_horizon()             → 執行整體流程，取得地平線座標與角度
    ✅ get_distance_scale_points() → 給定目前 horizon，推算各距離的 (x,y) 投影點
"""

# 模組導入
import math
import cv2
import numpy as np
import os
from time import time
from math import pi, atan

class FastHorizon:
    """快速海平線檢測器 - 基於邊緣檢測和霍夫變換的海平線檢測算法"""
    
    def __init__(self, init_all=True, canny_th1=25, canny_th2=45, Th_ROI=2, Th_slope=0.57, 
                 N_c=15, N_d=200, D_Y_hl_th=50, D_alpha_hl_th=2, max_outliers_th=4, 
                 hough_D_rho=2, hough_D_theta=pi/180, resize_factor=0.6,
                 # 這邊加 distance/horizon 相關參數
                 base_point=(320, 480),
                 D_horizon=5000,
                 camera_height=3.0,
                 front_distance=0.0,
                 mark_distances=None,
                 alpha_intervals=None,
                 alpha_distance_intervals=None,
                 roi_ratio= None):
        
        if init_all:
            self.resize_factor = resize_factor
            self.canny_th1, self.canny_th2 = canny_th1, canny_th2
            self.Th_ROI = Th_ROI * self.resize_factor
            self.Th_slope = Th_slope
            self.N_c, self.N_d, self.N_d_org = N_c, N_d, N_d
            self.DY_th = D_Y_hl_th
            self.Dphi_th = D_alpha_hl_th
            self.Nth_F_out = max_outliers_th
            self.hough_D_rho = Th_ROI
            self.hough_D_theta = hough_D_theta
            self.fsd = cv2.ximgproc.createFastLineDetector(_canny_th1=self.canny_th1, _canny_th2=self.canny_th2)
            self.Y_prv = self.phi_prv = np.nan
            self.DY = self.Dphi = np.nan
            self.N_F_out = 0
            self.D_rho, self.D_theta = 1, 1 * (pi/180)
            # ===============================
            # == distance scale 相關參數預設 ==
            # ===============================
            self.base_point = base_point
            self.D_horizon_param = D_horizon
            self.camera_height_param = camera_height
            self.front_distance_param = front_distance
            self.mark_distances = mark_distances if mark_distances is not None else [0,100,500,1000,2000,5000]
            self.alpha_intervals = alpha_intervals if alpha_intervals is not None else [2.5] * (len(self.mark_distances)-1)
            self.alpha_distance_intervals = alpha_distance_intervals if alpha_distance_intervals is not None else self.mark_distances
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
        # 新增：海平線中點
        self.x_hl_mid = self.y_hl_mid = np.nan
    
    def detect_horizon_and_scale_points(
        self, img,
        base_point=None,
        D_horizon=None,
        camera_height=None,
        front_distance=None,
        mark_distances=None,
        alpha_distance_intervals=None,
        alpha_intervals=None,
        roi_ratio=None,
    ):
        """
        執行海平線偵測並根據相機參數回傳對應距離刻度點 [(x, y, D), ...]
        若偵測失敗則回傳空陣列
        """
        h, w = img.shape[:2]
        self.org_width = w
        self.org_height = h
        self.res_width = int(w * self.resize_factor)
        self.res_height = int(h * self.resize_factor)

        self.start_time = time()
        self.__init__(init_all=False)
        self.N_d = self.N_d_org
        self.F_det = True
        
        roi_ratio = roi_ratio if roi_ratio is not None else self.roi_ratio

        # === ROI 處理 ===
        roi_offset_x = roi_offset_y = 0
        if roi_ratio is not None and len(roi_ratio) == 4:
            x0, x1, y0, y1 = roi_ratio
            x = int(w * x0)
            rw = int(w * (x1 - x0))
            y = int(h * y0)
            rh = int(h * (y1 - y0))
            roi_offset_x = x
            roi_offset_y = y
            img = img[y:y+rh, x:x+rw].copy()

        # --- 海平線處理流程 ---
        self.get_horizon_edges(img=img)
        self.hough_transform()
        self.outlier_handler_module()
        self.linear_least_square_fitting()

        if not self.F_det:
            self.img_edges = np.zeros(shape=self.in_img_red.shape, dtype=np.uint8)
            self.Y = self.phi = self.latency = np.nan
            self.img_with_hl = img
            return []

        self.Y_prv, self.phi_prv = self.Y, self.phi
        self.end_time = time()
        self.latency = round((self.end_time - self.start_time), 4)

        # --- 距離刻度投影 ---
        base_point = base_point if base_point is not None else self.base_point
        D_horizon = D_horizon if D_horizon is not None else self.D_horizon_param
        camera_height = camera_height if camera_height is not None else self.camera_height_param
        front_distance = front_distance if front_distance is not None else self.front_distance_param
        mark_distances = mark_distances if mark_distances is not None else self.mark_distances
        alpha_distance_intervals = alpha_distance_intervals if alpha_distance_intervals is not None else self.alpha_distance_intervals
        alpha_intervals = alpha_intervals if alpha_intervals is not None else self.alpha_intervals

        H = float(camera_height)
        base_x, base_y = base_point
        base_x -= roi_offset_x
        base_y -= roi_offset_y

        if not (hasattr(self, "xs_hl") and self.xs_hl is not None):
            return []
        if abs(self.xe_hl - self.xs_hl) < 1e-6:
            return []

        a = (self.ye_hl - self.ys_hl) / (self.xe_hl - self.xs_hl)
        b = self.ys_hl - a * self.xs_hl
        y_horizon = a * base_x + b

        if mark_distances is None:
            mark_distances = [0, 100, 500, 1000, 2000, D_horizon]
        if alpha_distance_intervals is None:
            alpha_distance_intervals = mark_distances
        if alpha_intervals is None:
            alpha_intervals = [2.5] * (len(alpha_distance_intervals) - 1)
        assert len(alpha_intervals) == len(alpha_distance_intervals) - 1, "alpha_intervals 必須比 alpha_distance_intervals 少一格"

        theta_0 = math.atan2(front_distance, H)
        theta_max = math.atan2(alpha_distance_intervals[-1], H)
        dy = base_y - y_horizon
        if abs(dy) < 1:
            return []

        points = []
        for D in mark_distances:
            seg_idx = next((i for i in range(len(alpha_distance_intervals) - 1)
                            if alpha_distance_intervals[i] <= D <= alpha_distance_intervals[i + 1]), None)
            if seg_idx is None:
                continue
            alpha = alpha_intervals[seg_idx]
            theta = math.atan2(D, H)
            t = (theta - theta_0) / (theta_max - theta_0 + 1e-8)
            t_alpha = (min(max(t, 0.0), 1.0)) ** alpha
            y_proj = int(round(base_y - t_alpha * dy)) + roi_offset_y
            points.append((int(base_x + roi_offset_x), int(y_proj), D))

        return points
    
    def get_horizon_edges(self, img):
        """
        海平線邊緣檢測 - 使用長度-斜率濾波器和ROI濾波器
        
        Args:
            img: 輸入圖像
            
        Returns:
            tuple: 邊緣點座標 (x_out, y_out)
        """
        self.x_out = self.y_out = None
        self.in_img_bgr = img
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
            print("未檢測到線段")
            self.x_out = self.y_out = None
            self.F_det = False
            return self.x_out, self.y_out
        
        # === 多階段濾波 ===
        self.lsf()    # 長度-斜率濾波
        if self.F_continue:
            self.roif()   # ROI濾波
        self.step()   # 邊緣點提取
        
        return self.x_out, self.y_out

    def lsf(self):
        """長度-斜率濾波器 (Length-Slope Filter)"""
        self.N_a = self.Segs_a.shape[0]
        self.Segs_a = np.reshape(self.Segs_a, newshape=(self.N_a, 4))
        
        # === 斜率濾波 ===
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
        
        # === 長度濾波 ===
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
        """ROI濾波器 (Region of Interest Filter)"""
        # === 獲取候選線段參數 ===
        self.xs_c, self.ys_c = self.Segs_c[:, 0], self.Segs_c[:, 1]
        self.xe_c, self.ye_c = self.Segs_c[:, 2], self.Segs_c[:, 3]
        self.alpha_c = self.alpha_a[self.b_from_a_idxs][self.c_from_b_idxs]
        self.B_c = np.subtract(self.ys_c, np.multiply(self.alpha_c, self.xs_c))
        self.B_c = np.broadcast_to(np.reshape(self.B_c, newshape=(self.N_c, 1)), 
                                  shape=(self.N_c, self.N_d))
        
        # === 計算距離 ===
        self.xs_d, self.ys_d = self.Segs_d[:, 0], self.Segs_d[:, 1]
        self.xe_d, self.ye_d = self.Segs_d[:, 2], self.Segs_d[:, 3]
        self.Ys_d = np.broadcast_to(np.reshape(self.ys_d, newshape=(1, self.N_d)), 
                                   shape=(self.N_c, self.N_d))
        self.Ye_d = np.broadcast_to(np.reshape(self.ye_d, newshape=(1, self.N_d)), 
                                   shape=(self.N_c, self.N_d))
        
        self.alpha_c = np.reshape(self.alpha_c, newshape=(self.N_c, 1))
        
        # === ROI檢查 ===
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
        """線段轉邊緣點 (Segment to Edge Points)"""
        self.x_out = np.zeros((0,))
        self.y_out = np.zeros((0,))
        self.N_f = self.Segs_f.shape[0]
        
        if self.N_f == 0:
            self.F_det = False
            return
            
        self.xs_f, self.ys_f = self.Segs_f[:, 0], self.Segs_f[:, 1]
        self.xe_f, self.ye_f = self.Segs_f[:, 2], self.Segs_f[:, 3]
        
        # === 計算線段長度 ===
        if self.F_continue:
            self.Len_c = self.Len_b[self.c_from_b_idxs]
            self.Len_e = self.Len_b[self.d_from_b_idxs][self.e_from_d_idxs]
            self.Len_f = np.concatenate((self.Len_c, self.Len_e))
        else:
            self.Len_f = np.sqrt(np.add(np.square(np.subtract(self.xs_f, self.xe_f)),
                                       np.square(np.subtract(self.ys_f, self.ye_f))))
        
        self.Len_f = np.uint16(np.subtract(self.Len_f, 1))
        self.u = np.arange(0, self.Len_f[0])
        
        # === 生成邊緣點 ===
        for self.Len_f_n, self.xs_f_n, self.ys_f_n, self.xe_f_n, self.ye_f_n in \
                zip(self.Len_f, self.xs_f, self.ys_f, self.xe_f, self.ye_f):
            self.u_n = self.u[0:self.Len_f_n]
            self.x_n = np.add(np.multiply(np.divide(np.subtract(self.xe_f_n, self.xs_f_n), self.Len_f_n), self.u_n),
                             self.xs_f_n)
            self.y_n = np.add(np.multiply(np.divide(np.subtract(self.ye_f_n, self.ys_f_n), self.Len_f_n), self.u_n),
                             self.ys_f_n)
            
            self.x_out = np.uint16(np.concatenate((self.x_out, self.x_n)))
            self.y_out = np.uint16(np.concatenate((self.y_out, self.y_n)))

    def hough_transform(self):
        """霍夫變換檢測直線"""
        if not self.F_det:
            return
            
        # === 創建邊緣圖 ===
        self.img_edges = np.zeros(shape=self.in_img_red.shape, dtype=np.uint8)
        self.img_edges[self.y_out, self.x_out] = 255
        
        # === 縮放處理 ===
        if self.resize_factor < 1:
            self.img_edges = cv2.resize(self.img_edges, dsize=(self.org_width, self.org_height))
            self.img_edges = np.int16(self.img_edges)
            self.img_edges = cv2.Canny(self.img_edges, self.img_edges, 254, 254)
        
        # === 霍夫變換 ===
        self.hough_lines = cv2.HoughLines(image=self.img_edges, rho=self.hough_D_rho, 
                                        theta=self.hough_D_theta, threshold=2, 
                                        min_theta=np.pi/3, max_theta=np.pi*2/3)
        
        if self.hough_lines is None:
            self.phi = self.Y = self.latency = np.nan
            self.F_det = False

    def linear_least_square_fitting(self):
        """線性最小二乘擬合"""
        if not self.F_det:
            return
            
        self.get_inlier_edges()
        self.inlier_edges_xy = np.zeros((self.inlier_edges_x.size, 2), dtype=np.int32)
        self.inlier_edges_xy[:, 0], self.inlier_edges_xy[:, 1] = self.inlier_edges_x, self.inlier_edges_y
        
        [vx, vy, x, y] = cv2.fitLine(points=self.inlier_edges_xy, distType=cv2.DIST_L2,
                                   param=0, reps=1, aeps=0.01)
        
        self.hl_slope = float(vy / vx)
        self.hl_intercept = float(y - self.hl_slope * x)
        
        # === 計算海平線端點 ===
        self.xs_hl = int(0)
        self.xe_hl = int(self.org_width - 1)
        self.ys_hl = int(self.hl_intercept)
        self.ye_hl = int((self.xe_hl * self.hl_slope) + self.hl_intercept)
        
        # === 計算最終參數 ===
        self.phi = (-atan(self.hl_slope)) * (180 / pi)
        self.Y = ((((self.org_width - 1) / 2) * self.hl_slope + self.hl_intercept))

    def get_inlier_edges(self):
        """獲取內點邊緣"""
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
        self.outlier_checker()      # 檢查異常值
        self.outlier_replacer()     # 替換異常值
        self.failure_state_handler() # 處理失敗狀態

    def outlier_checker(self):
        """检查最长的Hough峰值(假设为粗略的地平线)是否为异常值，如果是，则设置self.F_out为True"""
        self.F_out = False  # 重置为false
        
        # 计算对应于最长Hough线的(Y, phi)参数
        self.rho, self.theta = self.hough_lines[0][0]  # self.theta为弧度
        self.phi = ((np.pi / 2) - self.theta)
        self.phi = self.phi * (180 / np.pi)  # 转换为度数
        self.img_width = self.img_edges.shape[1]
        self.Y = (self.rho - 0.5 * self.img_width * np.cos(self.theta)) / (np.sin(self.theta))
        
        # 检查计算的参数(Y, phi)是否为异常值
        self.DY = abs(self.Y - self.Y_prv)
        self.Dphi = abs(self.phi - self.phi_prv)
        # 设置异常值标志
        self.F_out = (self.DY > self.DY_th) or (self.Dphi > self.Dphi_th)

    def outlier_replacer(self, M=2):
        """
        寻找异常地平线的有效替代品
        :param M: 考虑替换异常地平线的候选数量。如果M=-1，将考虑所有Hough候选
        """
        if not self.F_out:
            return  # 只有当检测为异常值时才执行
        
        self.img_width = self.img_edges.shape[1]
        self.hough_lines_nbr = self.hough_lines.shape[0]
        self.hough_lines = np.reshape(self.hough_lines, newshape=(self.hough_lines_nbr, 2))
        
        # 获取接下来M个候选的极坐标参数
        if M == -1:
            self.rho_cands = self.hough_lines[1::][:, 0]
            self.theta_cands = self.hough_lines[1::][:, 1]
        else:
            self.rho_cands = self.hough_lines[1:(M + 1)][:, 0]
            self.theta_cands = self.hough_lines[1:(M + 1)][:, 1]
        
        # 将(rho, theta)转换为(Y, phi)
        self.phi_cands = np.subtract((np.pi / 2), self.theta_cands) * (180 / np.pi)
        self.Y_cands = (self.rho_cands - 0.5 * self.img_width * np.cos(self.theta_cands)) / (np.sin(self.theta_cands))
        
        # 计算候选线的参数
        self.DY_cands = np.abs((np.subtract(self.Y_cands, self.Y_prv)))
        self.Dphi_cands = np.abs((np.subtract(self.phi_cands, self.phi_prv)))
        
        # 按DY从小到大排序
        self.Isrt_cands = np.argsort(self.DY_cands)
        self.DYsrt_cands = self.DY_cands[self.Isrt_cands]
        self.Dphisrt_cands = self.Dphi_cands[self.Isrt_cands]
        self.Ysrt_cands = self.Y_cands[self.Isrt_cands]
        self.phisrt_cands = self.phi_cands[self.Isrt_cands]
        
        # 寻找有效的替代品
        self.ISub = np.logical_and(self.DYsrt_cands < self.DY_th, self.Dphisrt_cands < self.Dphi_th)
        self.Isub = np.where(self.ISub == True)[0]
        self.sub_nbr = self.Isub.shape[0]
        
        if self.sub_nbr > 0:
            # 执行替换
            self.Isub = self.Isub[0]  # 选择第一个有效的替代品
            self.Y = self.Ysrt_cands[self.Isub]
            self.phi = self.phisrt_cands[self.Isub]
            
            # 计算对应的极坐标参数
            self.rhosrt_cands = self.rho_cands[self.Isrt_cands]
            self.thetasrt_cands = self.theta_cands[self.Isrt_cands]
            self.rho = self.rhosrt_cands[self.Isub]
            self.theta = self.thetasrt_cands[self.Isub]
            self.theta = (90 - self.phi) * (pi / 180)  # 转换为弧度
            self.rho = (self.Y * np.sin(self.theta)) + (0.5 * self.img_width * np.cos(self.theta))
        else:
            self.F_det = False  # 没有替代品意味着没有检测到地平线

    def failure_state_handler(self):
        """
        处理失败状态；锁定检测连续异常值的情况
        通过将之前的检测设置为np.nan来解决问题
        """
        if self.F_out:
            self.N_F_out += 1
            if self.N_F_out > self.Nth_F_out:
                # 检测到连续异常值的次数超过阈值
                self.Y_prv = np.nan
                self.phi_prv = np.nan
                self.N_F_out = 0
        else:
            self.N_F_out = 0