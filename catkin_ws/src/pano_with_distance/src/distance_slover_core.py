# -*- coding: utf-8 -*-
"""
HorizonDistanceSolver 類別
----------------------------

用途：
  根據 pano 圖上特定 y 座標位置（通常為畫面中線或標記點），推算該點對應的真實水平距離（以公尺為單位）。

應用場景：
  通常搭配已知標記點的距離比例尺與相機高度，將 pano 圖上距離標記點的 y 像素差轉換為真實距離。

初始化參數：
  alpha_intervals: List[List[float]]，每個相機的 alpha 區段（控制非線性插值）
  camera_height: 相機安裝高度（單位：公尺）
  front_distance: 畫面最下方基準點對應距離（公尺）
  cam_num: 總相機數（通常為 3：左中右）

方法說明：
  get_cam_points(polygon_points, cam_id)
    從全景距離標記點列表中，依相機 ID 取出該相機的點序列

  invert_distance_from_y(y, cam_id, polygon_points)
    根據 pano 上的 y 像素座標與相機 ID，從標記點列表反推實際距離（單位：公尺）

輸入：
  - y: pano 上的 y 座標（例如滑鼠點擊位置）
  - cam_id: 相機 ID（0: 左, 1: 中, 2: 右）
  - polygon_points: pano 上的所有距離標記點，每 cam 有 N 點，共 cam_num × N 點

輸出：
  - D: 推算得到的實際距離（公尺），失敗則為 None
"""

import math

class HorizonDistanceSolver:
    def __init__(
        self, 
        alpha_intervals=None,   # [[2.5,2.5,2.5], ...]，共 cam_num 組
        camera_height=3.0, 
        front_distance=9.2,
        cam_num=3
    ):
        self.cam_num = cam_num
        self.camera_height = camera_height
        self.front_distance = front_distance
        self.alpha_intervals = alpha_intervals

    def get_cam_points(self, polygon_points, cam_id):
        """
        根據 cam_id 取得對應的標記點列表

        輸入：
            polygon_points: 所有相機的距離標記點，依序排列
            cam_id: 第幾台相機（左中右為 0~2）

        回傳：
            對應相機的點列表、每台相機的點數
        """
        points_per_cam = len(polygon_points) // self.cam_num
        assert 0 <= cam_id < self.cam_num, f"cam_id 應為 0~{self.cam_num-1}"
        start = cam_id * points_per_cam
        end = start + points_per_cam
        return polygon_points[start:end], points_per_cam

    def invert_distance_from_y(self, y, cam_id, polygon_points):
        """
        根據 y, cam_id, polygon_points 反推出距離 D（單位：公尺）

        輸入：
            y: pano 圖上的 y 座標（像素）
            cam_id: 第幾台相機（左中右）
            polygon_points: 所有距離標記點（x, y, D），依相機分段依序排列

        回傳：
            D: 推算出來的距離（公尺），無法對應則為 None
        """
        scale_points, points_per_cam = self.get_cam_points(polygon_points, cam_id)
        # 若未指定 alpha_intervals，預設每段 alpha=2.5
        if self.alpha_intervals is None:
            alpha_list = [2.5] * (points_per_cam - 1)
        else:
            alpha_list = self.alpha_intervals[cam_id]
            assert len(alpha_list) == points_per_cam - 1, \
                f"alpha_intervals[{cam_id}] 長度應為 {points_per_cam - 1}"

        H = float(self.camera_height)

        # 按 y 由大到小排（y越小，距離越遠）
        scale_points = sorted(scale_points, key=lambda p: -p[1])

        for i in range(points_per_cam - 1):
            x0, y0, D0 = scale_points[i]
            x1, y1, D1 = scale_points[i+1]

            if (y0 >= y >= y1) or (y1 >= y >= y0):
                alpha = alpha_list[i]
                if abs(y0 - y1) < 1e-3:
                    continue
                t_alpha = (y0 - y) / (y0 - y1)
                t_alpha = min(max(t_alpha, 0.0), 1.0)
                t = t_alpha ** (1.0 / alpha)

                theta_0 = math.atan2(self.front_distance, H)
                theta_max = math.atan2(D1, H)
                theta = theta_0 + t * (theta_max - theta_0)
                D = math.tan(theta) * H
                return D
        return None
