# -*- coding: utf-8 -*-
"""
bbox2distance_core.py

根據物體類別與 bbox 高度，搭配已知的相機焦距與物體真實高度，
估算距離，並回傳帶有距離資訊的 detection 字典。
"""

class BBoxDistanceEstimator:
    def __init__(self,focal_length_px):
        self.focal_length_px = focal_length_px
        self.object_heights = {
            "Buoy": 1.0,
            "GuardBoat": 2.5,
            "RedBall": 3.0,
            "YellowBall": 3.0,
            "GreenBall": 3.0,
        }

    def estimate_distance(self, detection):
        """
        支援單一或多個 detection 輸入，並將距離加到 detection 字典中。

        :param detection: dict 或 list[dict]
        :return: dict 或 list[dict]，每個 dict 都會增加 "distance" 欄位
        """
        if isinstance(detection, list):
            return [self._estimate_single_and_add(d) for d in detection]
        else:
            return self._estimate_single_and_add(detection)

    def _estimate_single_and_add(self, det):
        obj_class = det.get("class")
        bbox = det.get("bbox")

        if obj_class not in self.object_heights:
            raise ValueError(f"未知的物體類別: {obj_class}")
        if not bbox or len(bbox) != 4:
            raise ValueError("bbox 格式錯誤，應為 [x_min, y_min, x_max, y_max]")

        bbox_pixel_height = bbox[3] - bbox[1]
        if bbox_pixel_height <= 0:
            raise ValueError("bbox 高度必須大於 0")

        real_height = self.object_heights[obj_class]
        distance = (real_height * self.focal_length_px) / bbox_pixel_height

        # # 加入 distance 欄位
        # det["distance"] = distance
        # 加入 distance 欄位
        det["distance_bbox"] = distance
        return det
