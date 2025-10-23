import numpy as np
from typing import List, Tuple, Dict, Optional

class HorizonX2AngleMapper:
    def __init__(
        self,
        crop_rect: Tuple[int, int, int, int],
        camera_angle: List[float],
        left_bound_angle: float = -105.0,
        right_bound_angle: float = 105.0,
    ):
        """
        :param crop_rect: (crop_x, crop_y, pano_width, pano_height)
        :param camera_angle: 每個相機對應的水平角度中心（例如 [-60, 0, 60]）
        :param left_bound_angle: pano 最左側對應的角度（預設 -105）
        :param right_bound_angle: pano 最右側對應的角度（預設 +105）
        """
        self.crop_x, self.crop_y, self.pano_w, _ = crop_rect
        self.camera_angle = camera_angle
        self.left_bound_angle = left_bound_angle
        self.right_bound_angle = right_bound_angle

    def _warp_point(self, pt: Tuple[float, float], M: Optional[np.ndarray]) -> Optional[Tuple[int, int]]:
        if M is None or pt is None or np.any(np.isnan(pt)):
            return None
        px, py = pt
        p = np.array([px, py, 1.0], dtype=np.float32)
        if M.shape == (2, 3):
            x = M[0, 0] * px + M[0, 1] * py + M[0, 2]
            y = M[1, 0] * px + M[1, 1] * py + M[1, 2]
        elif M.shape == (3, 3):
            p_warp = M @ p
            if abs(p_warp[2]) < 1e-6:
                return None
            x = p_warp[0] / p_warp[2]
            y = p_warp[1] / p_warp[2]
        else:
            return None
        return int(round(x)), int(round(y))

    def _build_mapping_from_horizon_slope(
            self,
            horizon_slope_pts: List[Dict],  # 每個 Dict: {'slope': float, 'center': (x, y)}
            M_L: Optional[np.ndarray],
            M_R: Optional[np.ndarray],
            offset: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        根據 horizon_slope_pts 與相機角度建立 pano_x ↔ angle 對應關係
        """
        x_list = [0.0]
        angle_list = [self.left_bound_angle]

        for cam_idx, (slope_pt, angle) in enumerate(zip(horizon_slope_pts, self.camera_angle)):
            center = slope_pt.get("center")
            if center is None or np.any(np.isnan(center)):
                continue
            x, y = center
            if cam_idx == 0 and M_L is not None:
                mapped = self._warp_point((x, y), M_L)
                if mapped is not None:
                    mapped_x = mapped[0] + offset
                else:
                    continue
            elif cam_idx == 1:
                mapped_x = int(round(x)) + offset
            elif cam_idx == 2 and M_R is not None:
                mapped = self._warp_point((x, y), M_R)
                if mapped is not None:
                    mapped_x = mapped[0] + offset
                else:
                    continue
            else:
                continue

            pano_x = mapped_x - self.crop_x
            x_list.append(pano_x)
            angle_list.append(angle)

        x_list.append(self.pano_w)
        angle_list.append(self.right_bound_angle)

        return np.array(x_list, dtype=np.float32), np.array(angle_list, dtype=np.float32)

    def estimate_angle(self, pano_x: float, x_list: np.ndarray, angle_list: np.ndarray) -> float:
        """
        根據 pano_x 推估對應的 angle
        """
        x_clamped = np.clip(pano_x, x_list[0], x_list[-1])
        return float(np.interp(x_clamped, x_list, angle_list))

    # def process_with_horizon_slope(
    #         self,
    #         horizon_slope_pts: List[Dict],  # List[{'slope': float, 'center': (x, y)}]
    #         mapped_detections: List[Dict],
    #         M_L: Optional[np.ndarray],
    #         M_R: Optional[np.ndarray],
    #         offset: int
    #     ) -> List[Dict]:
    #     """
    #     輸入 horizon 與 detection，為每個目標估算角度並加入 angle 欄位
    #     """
    #     x_list, angle_list = self._build_mapping_from_horizon_slope(horizon_slope_pts, M_L, M_R, offset)
    #     results = []
    #     for det in mapped_detections:
    #         bbox = det.get("bbox")
    #         if not bbox or len(bbox) != 4:
    #             continue
    #         x1, _, x2, _ = bbox
    #         pano_x = (x1 + x2) / 2.0  # bbox 中心點 x
    #         angle = self.estimate_angle(pano_x, x_list, angle_list)
    #         new_det = det.copy()
    #         new_det["angle"] = angle
    #         results.append(new_det)
    #     return results

    def process_with_horizon_slope(
            self,
            horizon_slope_pts: List[Dict],  # List[{'slope': float, 'center': (x, y)}]
            mapped_detections: List[Dict],
            M_L: Optional[np.ndarray],
            M_R: Optional[np.ndarray],
            offset: int
        ) -> List[Dict]:
        """
        輸入 horizon 與 detection，為每個目標估算角度，並加入 horizon point & slope
        """
        x_list, angle_list = self._build_mapping_from_horizon_slope(horizon_slope_pts, M_L, M_R, offset)
        # 取出 horizon 的所有 x 位置 (以 pano 上的 x 為主)
        horizon_xs = []
        horizon_ys = []
        horizon_slopes = []
        for cam_idx, slope_pt in enumerate(horizon_slope_pts):
            center = slope_pt.get("center")
            if center is None or np.any(np.isnan(center)):
                continue
            x, y = center
            # 對應各自投影
            if cam_idx == 0 and M_L is not None:
                mapped = self._warp_point((x, y), M_L)
                if mapped is not None:
                    mapped_x = mapped[0] + offset
                    mapped_y = mapped[1]
                else:
                    continue
            elif cam_idx == 1:
                mapped_x = int(round(x)) + offset
                mapped_y = int(round(y))
            elif cam_idx == 2 and M_R is not None:
                mapped = self._warp_point((x, y), M_R)
                if mapped is not None:
                    mapped_x = mapped[0] + offset
                    mapped_y = mapped[1]
                else:
                    continue
            else:
                continue
            pano_x = mapped_x - self.crop_x
            pano_y = mapped_y - self.crop_y
            horizon_xs.append(pano_x)
            horizon_ys.append(pano_y)
            horizon_slopes.append(slope_pt.get("slope"))

        results = []
        for det in mapped_detections:
            bbox = det.get("bbox")
            src = det.get("source")   # 來源相機 index
            if not bbox or len(bbox) != 4 or src is None:
                continue
            x1, _, x2, _ = bbox
            pano_x = (x1 + x2) / 2.0
            angle = self.estimate_angle(pano_x, x_list, angle_list)

            # ==== 根據 source 直接抓 horizon info ====
            if 0 <= src < len(horizon_slope_pts):
                horizon_info = horizon_slope_pts[src]
                center = horizon_info.get("center")
                slope = horizon_info.get("slope")
                if center is not None and not np.any(np.isnan(center)):
                    x_horizon_point, y_horizon_point = center
                else:
                    x_horizon_point = None
                    y_horizon_point = None
            else:
                x_horizon_point = None
                y_horizon_point = None
                slope = None

            new_det = det.copy()
            new_det["angle"] = angle
            new_det["horizon_point"] = {"x": x_horizon_point, "y": y_horizon_point}
            new_det["horizon_slope"] = slope
            results.append(new_det)
        return results

# import numpy as np
# from typing import List, Tuple, Dict, Optional

# class HorizonX2AngleMapper:
#     def __init__(self, crop_rect: Tuple[int, int, int, int], camera_angle: List[float]):
#         self.crop_x, self.crop_y, self.pano_w, _ = crop_rect
#         self.camera_angle = camera_angle

#     def _warp_point(self, pt: Tuple[float, float], M: Optional[np.ndarray]) -> Optional[Tuple[int, int]]:
#         if M is None or pt is None or np.any(np.isnan(pt)):
#             return None
#         px, py = pt
#         p = np.array([px, py, 1.0], dtype=np.float32)
#         if M.shape == (2, 3):
#             x = M[0, 0] * px + M[0, 1] * py + M[0, 2]
#             y = M[1, 0] * px + M[1, 1] * py + M[1, 2]
#         elif M.shape == (3, 3):
#             p_warp = M @ p
#             if abs(p_warp[2]) < 1e-6:
#                 return None
#             x = p_warp[0] / p_warp[2]
#             y = p_warp[1] / p_warp[2]
#         else:
#             return None
#         return int(round(x)), int(round(y))

#     def _build_mapping_from_horizon_slope(
#             self,
#             horizon_slope_pts: List[Dict],  # 每個 Dict: {'slope': float, 'center': (x, y)}
#             M_L: Optional[np.ndarray],
#             M_R: Optional[np.ndarray],
#             offset: int
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         用 horizon_slope_pts（每相機1個center點）+角度，做 pano x ↔ angle 對應
#         """
#         x_list = [0.0]
#         angle_list = [-105.0]  # or -180~180, 視你的 pano

#         for cam_idx, (slope_pt, angle) in enumerate(zip(horizon_slope_pts, self.camera_angle)):
#             center = slope_pt.get("center")
#             if center is None or np.any(np.isnan(center)):
#                 continue
#             x, y = center
#             if cam_idx == 0 and M_L is not None:
#                 mapped = self._warp_point((x, y), M_L)
#                 if mapped is not None:
#                     mapped_x = mapped[0] + offset
#                 else:
#                     continue
#             elif cam_idx == 1:
#                 mapped_x = int(round(x)) + offset
#             elif cam_idx == 2 and M_R is not None:
#                 mapped = self._warp_point((x, y), M_R)
#                 if mapped is not None:
#                     mapped_x = mapped[0] + offset
#                 else:
#                     continue
#             else:
#                 continue

#             pano_x = mapped_x - self.crop_x
#             x_list.append(pano_x)
#             angle_list.append(angle)

#         x_list.append(self.pano_w)
#         angle_list.append(105.0)

#         return np.array(x_list, dtype=np.float32), np.array(angle_list, dtype=np.float32)

#     def estimate_angle(self, pano_x: float, x_list: np.ndarray, angle_list: np.ndarray) -> float:
#         x_clamped = np.clip(pano_x, x_list[0], x_list[-1])
#         return float(np.interp(x_clamped, x_list, angle_list))

#     def process_with_horizon_slope(
#             self,
#             horizon_slope_pts: List[Dict],  # List[{'slope': float, 'center': (x, y)}]
#             mapped_detections: List[Dict],
#             M_L: Optional[np.ndarray],
#             M_R: Optional[np.ndarray],
#             offset: int
#     ) -> List[Dict]:
#         """
#         horizon_slope_pts: List[{'slope': float, 'center': (x, y)}]
#         mapped_detections 每筆格式同前
#         """
#         x_list, angle_list = self._build_mapping_from_horizon_slope(horizon_slope_pts, M_L, M_R, offset)
#         results = []
#         for det in mapped_detections:
#             bbox = det.get("bbox")
#             if not bbox or len(bbox) != 4:
#                 continue
#             x1, _, x2, _ = bbox
#             pano_x = (x1 + x2) / 2.0  # bbox 中心點
#             angle = self.estimate_angle(pano_x, x_list, angle_list)
#             new_det = det.copy()
#             new_det["angle"] = angle
#             results.append(new_det)
#         return results