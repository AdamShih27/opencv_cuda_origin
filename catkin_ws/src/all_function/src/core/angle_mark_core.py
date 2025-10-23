import numpy as np
import cv2

class PanoHorizonAngleMarker:
    def __init__(
        self,
        angle_colors=[(255,0,255), (255,0,255), (255,0,255)],
        radius=1,
        thickness=1,
        font_scale=0.4,
        crop_rect=None,
        camera_angle=[-60,0,60],
        angle_label_offset=16,
    ):
        self.angle_colors = angle_colors
        self.radius = radius
        self.thickness = thickness
        self.font_scale = font_scale
        self.crop_rect = crop_rect
        self.camera_angle = camera_angle
        self.angle_label_offset = angle_label_offset

    def warp_point(self, pt, M):
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

    def mark(self, pano, horizon_slope_pts, M_L, M_R, offset):
        """
        pano:            pano 圖 (np.array)
        horizon_slope_pts: List of dicts, 每台相機 {'slope':..., 'center':(x, y)}
        M_L/M_R:         左右影像 pano ← 原圖 的仿射或單應矩陣
        offset:          單張寬度
        """
        pano_mask = pano.copy()
        h, w_img = pano.shape[:2]
        crop_x = crop_y = 0
        if self.crop_rect is not None:
            crop_x, crop_y = self.crop_rect[:2]
        for cam_idx, (slope_pt, angle) in enumerate(zip(horizon_slope_pts, self.camera_angle)):
            center = slope_pt.get("center", None)
            if center is None or np.any(np.isnan(center)):
                continue
            x, y = center
            # 做座標轉換
            if cam_idx == 0:
                mapped = self.warp_point((x, y), M_L)
                if mapped is not None:
                    mapped = (mapped[0] + offset, mapped[1])
            elif cam_idx == 1:
                mapped = (int(round(x)) + offset, int(round(y)))
            else:
                mapped = self.warp_point((x, y), M_R)
                if mapped is not None:
                    mapped = (mapped[0] + offset, mapped[1])
            if mapped is None or np.any(np.isnan(mapped)):
                continue
            pt_shifted = (mapped[0] - crop_x, mapped[1] - crop_y)
            if not (0 <= pt_shifted[0] < w_img and 0 <= pt_shifted[1] < h):
                continue
            angle_text = f"{angle}deg"
            angle_pos = (pt_shifted[0], pt_shifted[1] - self.radius - self.angle_label_offset)
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