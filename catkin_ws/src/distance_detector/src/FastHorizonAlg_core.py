"""
FastHorizon - 快速海平線檢測算法
"""
import math
import cv2
import numpy as np
import os
from time import time
from warnings import warn
from math import pi, atan, sin, cos
import tkinter as tk


class FastHorizon:
    """快速海平線檢測器 - 基於邊緣檢測和霍夫變換的海平線檢測算法"""
    
    def __init__(self, init_all=True, canny_th1=25, canny_th2=45, Th_ROI=2, Th_slope=0.57, 
                 N_c=15, N_d=200, D_Y_hl_th=50, D_alpha_hl_th=2, max_outliers_th=4, 
                 hough_D_rho=2, hough_D_theta=pi/180, resize_factor=0.6):
        """
        初始化海平線檢測器
        
        主要參數:
        - canny_th1/th2: Canny邊緣檢測閾值
        - Th_ROI: ROI濾波器寬度控制參數
        - Th_slope: 線段斜率過濾閾值
        - N_c/N_d: 候選線段數量控制
        - D_Y/alpha_hl_th: 異常值檢測閾值
        - resize_factor: 圖像縮放因子
        """
        if init_all:
            # === 基本參數設置 ===
            self.resize_factor = resize_factor
            self.canny_th1, self.canny_th2 = canny_th1, canny_th2
            
            # === 核心算法參數 ===
            self.Th_ROI = Th_ROI * self.resize_factor
            self.Th_slope = Th_slope
            self.N_c, self.N_d, self.N_d_org = N_c, N_d, N_d
            
            # === 異常值處理參數 ===
            self.DY_th = D_Y_hl_th        # Y位置差異閾值
            self.Dphi_th = D_alpha_hl_th  # 角度差異閾值
            self.Nth_F_out = max_outliers_th  # 最大連續異常值數量
            
            # === 霍夫變換參數 ===
            self.hough_D_rho = Th_ROI
            self.hough_D_theta = hough_D_theta
            
            # === 創建快速線段檢測器 ===
            self.fsd = cv2.ximgproc.createFastLineDetector(_canny_th1=self.canny_th1, _canny_th2=self.canny_th2)
            
            # === 異常值追蹤變量 ===
            self.Y_prv = self.phi_prv = np.nan  # 前一幀的位置和角度
            self.DY = self.Dphi = np.nan        # 當前差異值
            self.N_F_out = 0                    # 連續異常值計數
            
            # === 顏色常數 ===
            self.color_red = (0, 0, 255)
            self.color_blue = (255, 0, 0)
            self.color_green = (0, 255, 0)
            self.color_yellow = (0, 255, 255)
            self.color_aqua = (255, 255, 0)
            self.D_rho, self.D_theta = 1, 1 * (pi/180)
        
        # === 重置所有處理變量 ===
        self._reset_processing_variables()

    def _reset_processing_variables(self):
        """重置所有處理過程中的變量"""
        # 線段相關變量
        for attr in ['Segs_a', 'Segs_b', 'Segs_c', 'Segs_d', 'Segs_e', 'Segs_f']:
            setattr(self, attr, None)
        
        # 長度和索引變量
        for attr in ['Len_a', 'Len_b', 'Len_c', 'Len_d', 'Len_e', 'Len_f', 'Len_b_sort_idxs']:
            setattr(self, attr, None)
        
        # 座標變量
        coord_attrs = ['xs_a', 'ys_a', 'xe_a', 'ye_a', 'xs_b', 'ys_b', 'xe_b', 'ye_b',
                      'xs_c', 'ys_c', 'xe_c', 'ye_c', 'xs_d', 'ys_d', 'xe_d', 'ye_d',
                      'xs_f', 'ys_f', 'xe_f', 'ye_f', 'xs_hl', 'ys_hl', 'xe_hl', 'ye_hl']
        for attr in coord_attrs:
            setattr(self, attr, None)
        
        # 處理標志和輸出
        self.F_continue = self.F_det = True
        self.F_out = False
        self.Y = self.phi = self.theta = self.rho = self.latency = np.nan
    
    def get_distance_scale_points(
        self,
        base_point=(320, 480),
        D_horizon=5000,
        camera_height=3.0,
        front_distance=0.0,               # <--- 對應 base_point 的距離
        mark_distances=None,
        alpha_distance_intervals=None,          # 距離區間（決定每區間的 alpha）
        alpha_intervals=None,             # 各區間對應的 alpha
    ):
        """
        仰角法多區間權重：不同距離區間可指定不同 alpha 權重。
        會將 mark_distances 投影到畫面 y 座標，回傳 [(x, y, D), ...]
        """
        H = float(camera_height)
        base_x, base_y = base_point

        # --- 海平線參數 ---
        if abs(self.xe_hl - self.xs_hl) < 1e-6:
            return []
        a = (self.ye_hl - self.ys_hl) / (self.xe_hl - self.xs_hl)
        b = self.ys_hl - a * self.xs_hl
        y_horizon = a * base_x + b

        # --- 標註距離 ---
        if mark_distances is None:
            mark_distances = [0, 100, 500, 1000, 2000, 5000, D_horizon]
        if alpha_distance_intervals is None:
            # 預設區間根據 mark_distances
            alpha_distance_intervals = mark_distances
        if alpha_intervals is None:
            alpha_intervals = [2.5] * (len(alpha_distance_intervals) - 1)
        assert len(alpha_intervals) == len(alpha_distance_intervals) - 1, "alpha_intervals 必須比 alpha_distance_intervals 少一格"

        # --- 仰角與畫面高度 ---
        theta_0 = math.atan2(front_distance, H)
        theta_max = math.atan2(alpha_distance_intervals[-1], H)
        dy = base_y - y_horizon
        if abs(dy) < 1:
            return []

        # --- 計算每個 D 的畫面 y 座標 ---
        points = []
        for D in mark_distances:
            # 找出 D 屬於哪個區間
            seg_idx = None
            for i in range(len(alpha_distance_intervals) - 1):
                if alpha_distance_intervals[i] <= D <= alpha_distance_intervals[i + 1]:
                    seg_idx = i
                    break
            if seg_idx is None:
                # 超出範圍就跳過
                continue
            alpha = alpha_intervals[seg_idx]
            # 仰角
            theta = math.atan2(D, H)
            # 仰角比
            t = (theta - theta_0) / (theta_max - theta_0 + 1e-8)
            t = min(max(t, 0.0), 1.0)
            t_alpha = t ** alpha
            y_proj = int(round(base_y - t_alpha * dy))
            points.append((int(base_x), int(y_proj), D))
        return points


    def draw_distance_scale_points(
        self,
        points,
        color=(0,255,255),
        base_point=None,
        Fd=9.2,
        angle_label=None,
        angle_label_offset=30
    ):
        """
        根據 points 列表標記到 self.img_with_hl
        points: [(x, y, D), ...]
        """
        h, w = self.img_with_hl.shape[:2]
        for idx, (x_proj, y_proj, D) in enumerate(points):
            if D == 0 and base_point is not None:
                cv2.circle(self.img_with_hl, (base_point[0], base_point[1]), 5, (0, 165, 255), -1)
                cv2.putText(self.img_with_hl, f"{Fd:.1f}m", (base_point[0] + 8, base_point[1] + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1, cv2.LINE_AA)
            elif 0 <= x_proj < w and 0 <= y_proj < h:
                cv2.circle(self.img_with_hl, (x_proj, y_proj), 3, color, -1)
                if D >= 1000:
                    label = f"{int(D//1000)}km"
                else:
                    label = f"{int(D)}m"
                cv2.putText(self.img_with_hl, label, (x_proj + 5, y_proj + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        # 角度標註
        if angle_label is not None:
            x_hl_mid = int((self.xs_hl + self.xe_hl) / 2)
            y_hl_mid = int((self.ys_hl + self.ye_hl) / 2)
            x_label = x_hl_mid
            y_label = max(0, y_hl_mid - angle_label_offset)
            text = f"{angle_label:.1f}deg"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.65
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            x_text = x_label - text_width // 2
            cv2.putText(self.img_with_hl, text, (x_text, y_label),
                    font, font_scale, (255, 0, 255), thickness, cv2.LINE_AA)

    def get_horizon(self, img, get_image=False):
        """
        主要處理函數 - 檢測圖像中的海平線
        
        Args:
            img: 輸入圖像
            get_image: 是否繪製結果圖像
            
        Returns:
            tuple: (Y位置, 角度, 延遲時間, 檢測成功標志)
        """
        self.start_time = time()
        self.__init__(init_all=False)  # 重置處理變量
        self.N_d = self.N_d_org
        self.F_det = True
        
        # === 主要處理流程 ===
        self.get_horizon_edges(img=img)     # 獲取海平線邊緣點
        self.hough_transform()              # 霍夫變換檢測直線
        self.outlier_handler_module()       # 異常值處理
        self.linear_least_square_fitting()  # 線性最小二乘擬合
        
        if self.F_det:
            print(f"Y = {self.Y}, phi = {self.phi}")
            self.Y_prv, self.phi_prv = self.Y, self.phi
            self.end_time = time()
            self.latency = round((self.end_time - self.start_time), 4)
        else:
            self.img_edges = np.zeros(shape=self.in_img_red.shape, dtype=np.uint8)
            self.Y = self.phi = self.latency = np.nan
            self.img_with_hl = img
            
        print(f"執行時間: {self.latency} 秒")
        return self.Y, self.phi, self.latency, self.F_det

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
        self.alpha_a = np.divide(np.subtract(self.ye_a, self.ys_a), 
                                np.subtract(self.xe_a, self.xs_a))
        
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
            self.rho = (self.Y * sin(self.theta)) + (0.5 * self.img_width * cos(self.theta))
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

    def draw_hl(self):
        """在属性'self.img_with_hl'上绘制地平线"""
        if self.F_det:
            thickness = 2
            cv2.line(self.img_with_hl, (self.xs_hl, self.ys_hl), (self.xe_hl, self.ye_hl), 
                    (0, 0, 255), thickness=thickness)

    def y_from_xpolar(self, x, rho, theta):
        """根据极坐标rho和theta定义的直线，返回对应x的y坐标"""
        return int((1 / np.sin(theta)) * (rho - x * np.cos(theta)))

    def reset_for_new_video(self):
        """重置与上一个视频相关的属性，避免使用其他视频的结果处理新视频"""
        self.Y_prv = np.nan      # 前一帧地平线位置
        self.phi_prv = np.nan    # 前一帧地平线倾斜
        self.DY = np.nan         # 当前和前一位置的绝对差值
        self.Dphi = np.nan       # 当前和前一倾斜的绝对差值
        self.N_F_out = 0         # 连续异常值检测计数

    def evaluate(self, src_video_folder, src_gt_folder, dst_video_folder="", 
                dst_quantitative_results_folder="", draw_and_save=True):
        """
        产生包含地平线边缘滤波器算法定量结果的.npy文件
        包含每张图像的信息：|Y_gt - Y_det|, |alpha_gt - alpha_det|和延迟(秒)
        """
        src_video_names = sorted(os.listdir(src_video_folder))
        srt_gt_names = sorted(os.listdir(src_gt_folder))
        
        # 打印视频文件和对应的gt文件
        for src_video_name, src_gt_name in zip(src_video_names, srt_gt_names):
            print("{} will correspond to {}".format(src_video_name, src_gt_name))
        
        # 用户确认对应关系
        while True:
            yn = 'y'  # 自动确认
            if yn == 'y':
                break
            elif yn == 'n':
                print("\n定量评估中止：一个或多个GT文件与正确的视频文件不对应")
                return
        
        self.det_horizons_all_files = np.empty(shape=[0, 5])
        nbr_of_vids = len(src_video_names)
        vid_indx = 0
        
        # 处理每个视频文件
        for src_video_name, src_gt_name in zip(src_video_names, srt_gt_names):
            self.reset_for_new_video()
            vid_indx += 1
            print("loaded video/loaded gt: {}/{}".format(src_video_name, src_gt_name))
            
            src_video_path = os.path.join(src_video_folder, src_video_name)
            src_gt_path = os.path.join(src_gt_folder, src_gt_name)
            
            cap = cv2.VideoCapture(src_video_path)
            
            # 创建视频写入器
            fps = cap.get(propId=cv2.CAP_PROP_FPS)
            self.org_width = int(cap.get(propId=cv2.CAP_PROP_FRAME_WIDTH))
            self.org_height = int(cap.get(propId=cv2.CAP_PROP_FRAME_HEIGHT))
            self.res_width = int(self.org_width * self.resize_factor)
            self.res_height = int(self.org_height * self.resize_factor)
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            
            if draw_and_save:
                dst_vid_path = os.path.join(dst_video_folder, "Author1_" + src_video_name)
                video_writer = cv2.VideoWriter(dst_vid_path, fourcc, fps, 
                                            (self.org_width, self.org_height), True)
            
            self.gt_horizons = np.load(src_gt_path)
            nbr_of_annotations = self.gt_horizons.shape[0]
            nbr_of_frames = int(cap.get(propId=cv2.CAP_PROP_FRAME_COUNT))
            
            if nbr_of_frames != nbr_of_annotations:
                warning_text = "注释数量({})不等于帧数({})".format(nbr_of_annotations, nbr_of_frames)
                print("----------警告---------")
                print(warning_text)
                print("--------------------------")
            
            self.det_horizons_per_file = np.zeros((nbr_of_annotations, 5))
            self.__init__(init_all=True)  # 重新初始化所有属性
            
            # 处理每一帧
            for idx, gt_horizon in enumerate(self.gt_horizons):
                no_error_flag, frame = cap.read()
                if not no_error_flag:
                    break
                
                self.input_img = frame
                self.get_horizon(img=self.input_img)  # 获取地平线位置和倾斜
                self.gt_position_hl, self.gt_tilt_hl = gt_horizon[0], gt_horizon[1]
                
                print("Frame {}/{}. Video {}/{}".format(idx, nbr_of_frames, vid_indx, nbr_of_vids))
                
                self.det_horizons_per_file[idx] = [
                    self.Y, self.phi,
                    round(abs(self.Y - self.gt_position_hl), 4),
                    round(abs(self.phi - self.gt_tilt_hl), 4),
                    self.latency
                ]
                
                if draw_and_save:
                    self.draw_hl()
                    video_writer.write(self.img_with_hl)
            
            cap.release()
            if draw_and_save:
                video_writer.release()
            print("视频文件 {} 处理完成。".format(src_video_name))
            
            # 保存当前视频文件的定量结果
            src_video_name_no_ext = os.path.splitext(src_video_name)[0]
            det_horizons_per_file_dst_path = os.path.join(dst_quantitative_results_folder,
                                                        src_video_name_no_ext + ".npy")
            np.save(det_horizons_per_file_dst_path, self.det_horizons_per_file)
            
            self.det_horizons_all_files = np.append(self.det_horizons_all_files,
                                                self.det_horizons_per_file, axis=0)
        
        # 处理完所有视频文件后，保存定量结果
        src_video_folder_name = os.path.basename(src_video_folder)
        dst_detected_path = os.path.join(dst_quantitative_results_folder,
                                    "all_det_hl_" + src_video_folder_name + ".npy")
        np.save(dst_detected_path, self.det_horizons_all_files)
        
        self.Y_hl_all = self.det_horizons_all_files[:, 2]
        self.alpha_hl_all = self.det_horizons_all_files[:, 3]
        self.latency_all = self.det_horizons_all_files[:, 4]
        self.false_positive_nbr = np.size(np.argwhere(np.isnan(self.Y_hl_all)))

    def video_demo(self, video_path, display=True):
        """视频演示功能"""
        demo_cap = cv2.VideoCapture(video_path)
        self.org_width = int(demo_cap.get(propId=cv2.CAP_PROP_FRAME_WIDTH))
        self.org_height = int(demo_cap.get(propId=cv2.CAP_PROP_FRAME_HEIGHT))
        self.res_width = int(self.org_width * self.resize_factor)
        self.res_height = int(self.org_height * self.resize_factor)
        
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        
        # 调整显示窗口大小
        if (self.org_height >= screen_height) or (self.org_width >= screen_width):
            dst_width = int(screen_width - 0.2 * screen_width)
            dst_height = int(screen_height - 0.2 * screen_height)
        else:
            dst_width = self.org_width
            dst_height = self.org_height
        
        wait = 30
        if not demo_cap.isOpened():
            print("错误：无法打开视频文件。")
            return
        
        try:
            while True:
                ret, frame = demo_cap.read()
                if not ret:
                    break
                
                self.input_img = frame
                self.get_horizon(img=self.input_img)  # 获取地平线位置和倾斜
                self.draw_hl()  # 在self.img_with_hl上绘制地平线
                
                if display:
                    cv2.imshow("Horizon Detection", cv2.resize(self.img_with_hl, (dst_width, dst_height)))
                    if cv2.waitKey(wait) & 0xFF == ord('q'):
                        break
        finally:
            demo_cap.release()
            root.destroy()
            if display:
                cv2.destroyAllWindows()