# -*- coding: utf-8 -*-
"""
BBHeight2Distance 類別
------------------------

用途：
  根據物體在影像中所呈現的「邊界框高度（像素）」推估實際距離，假設已知物體真實高度與相機焦距。

應用場景：
  - 單目測距：從攝影機拍到的影像中偵測到物體（如人、船、車等）並量測其 bbox 高度後，推算該物體與相機的距離。
  - 不需校正地面或使用深度相機，即可快速得到估算值（適合先驗物體高度已知的場景）

核心公式：
  D = H * f / h  
  - D：與物體的實際距離（公尺）
  - H：物體真實高度（公尺）
  - f：相機焦距（像素）
  - h：影像中物體高度（像素）

初始化參數：
  - camera_height_m: 相機高度（選填，用於延伸應用）
  - focal_length_px: 相機焦距（像素），可由相機內參 `fx` 推得
  - object_real_height_m: 物體真實高度（公尺）

主要方法：
  - estimate_distance(h_px): 根據單一 bbox 高度估算距離
  - batch_estimate([h1, h2, ...]): 批次估算多個 bbox 的距離

範例：
  bbox_height = 200px，焦距 800px，物體實際高度 3 公尺
  ➜ 距離 = 3 × 800 / 200 = 12 公尺
"""

import numpy as np

class BBHeight2Distance:
    def __init__(self, camera_height_m=3.0, focal_length_px=800.0, object_real_height_m=3.0):
        """
        初始化距離估算器

        參數:
        - camera_height_m: 相機高度（公尺）
        - focal_length_px: 焦距（像素），可由相機內參矩陣 fx 取得
        - object_real_height_m: 偵測物體真實高度，預設為 3 公尺
        """
        self.camera_height = camera_height_m
        self.focal_length = focal_length_px
        self.object_real_height = object_real_height_m

    def estimate_distance(self, bbox_height_px):
        """
        根據 bounding box 高度估算距離

        參數:
        - bbox_height_px: 邊界框的像素高度

        回傳:
        - 與物體的距離（公尺），無法估算則為 None
        """
        if bbox_height_px <= 0:
            return None
        # 公式: D = H * f / h
        distance_m = (self.object_real_height * self.focal_length) / bbox_height_px
        return distance_m

    def batch_estimate(self, bbox_heights_px):
        """
        批次估算多個邊界框高度對應的距離

        參數:
        - bbox_heights_px: list 或 numpy array

        回傳:
        - list of distance (公尺)
        """
        return [self.estimate_distance(h) for h in bbox_heights_px]


# --------- 範例測試 ---------
# if __name__ == "__main__":
#     # 初始化測距器（可調整參數）
#     estimator = BBHeight2Distance(
#         camera_height_m=3.0,         # 可選擇是否使用於進階應用
#         focal_length_px=800.0,       # fx 可從內參矩陣獲得
#         object_real_height_m=3.0     # 例如船隻/人物高度為 3 公尺
#     )

#     # 假設有幾個物體的 bbox 高度為：
#     test_heights = [300, 200, 100, 50]  # 單位：像素
#     for h in test_heights:
#         dist = estimator.estimate_distance(h)
#         print(f"Bounding box height: {h}px → Estimated distance: {dist:.2f} m")
