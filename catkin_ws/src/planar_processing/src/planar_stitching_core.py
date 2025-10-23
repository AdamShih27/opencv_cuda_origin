# planar_stitching_core.py

import cv2
import numpy as np
import os

class Stitcher:
    def __init__(self, use_blending=True, fixed_width=250):
        self.use_blending = use_blending
        self.fixed_width = fixed_width

    def remove_black_border(self, img):
        """移除影像周圍黑邊（取得最大非零rect）"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        best_rect = (0, 0, img.shape[1], img.shape[0])
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area > max_area:
                max_area = area
                best_rect = (x, y, w, h)
        x, y, w, h = best_rect
        return img[y:y+h, x:x+w]

    def linear_blending(self, img_left, img_right):
        """Alpha blending，僅於重疊區"""
        h1, w1 = img_left.shape[:2]
        h2, w2 = img_right.shape[:2]
        height = max(h1, h2)
        width = max(w1, w2)

        img_left_large = np.zeros((height, width, 3), dtype=np.uint8)
        img_right_large = np.zeros((height, width, 3), dtype=np.uint8)
        img_left_large[:h1, :w1] = img_left
        img_right_large[:h2, :w2] = img_right

        overlap_mask = np.logical_and(
            np.any(img_left_large != 0, axis=2),
            np.any(img_right_large != 0, axis=2)
        )
        overlap_indices = np.where(np.any(overlap_mask, axis=0))[0]
        if len(overlap_indices) < 2:
            return img_left_large + img_right_large

        min_x, max_x = overlap_indices[0], overlap_indices[-1]
        overlap_width = max_x - min_x + 1
        alpha_mask = np.zeros((height, width), dtype=np.float32)
        alpha_line = np.empty(overlap_width, dtype=np.float32)
        if overlap_width <= self.fixed_width:
            alpha_line.fill(1.0)
        else:
            alpha_line[:self.fixed_width] = 1.0
            blend_width = overlap_width - self.fixed_width
            if blend_width > 1:
                alpha_line[self.fixed_width:] = np.linspace(1.0, 0.0, blend_width)
            else:
                alpha_line[self.fixed_width:] = 0.0
        alpha_mask[:, min_x:max_x+1] = np.tile(alpha_line, (height, 1))
        alpha_3c = cv2.merge([alpha_mask] * 3)
        blended = img_left_large.astype(np.float32) * alpha_3c + \
                  img_right_large.astype(np.float32) * (1.0 - alpha_3c)
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        blended[~overlap_mask] = img_left_large[~overlap_mask] + img_right_large[~overlap_mask]
        return blended

    def compute_homography(self, img_left, img_right):
        """用SIFT+BF+RANSAC求Homography"""
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img_left, None)
        kp2, des2 = sift.detectAndCompute(img_right, None)
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return None
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        src_pts, dst_pts = [], []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
                src_pts.append(kp1[m.queryIdx].pt)
                dst_pts.append(kp2[m.trainIdx].pt)
        if len(src_pts) < 4 or len(dst_pts) < 4:
            return None
        src_pts = np.float32(src_pts)
        dst_pts = np.float32(dst_pts)
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        return H

    def stitching(self, img_left, img_right, flip=False, H=None, save_H_path=None):
        """單一拼接流程"""
        if H is None:
            H = self.compute_homography(img_left, img_right)
            if save_H_path is not None and H is not None:
                np.save(save_H_path, H)
        if H is None:
            return None
        # (2) warpPerspective
        gpu_right = cv2.cuda_GpuMat()
        gpu_right.upload(img_right)
        warped_size = (img_left.shape[1] + img_right.shape[1], img_left.shape[0])
        gpu_right_warped = cv2.cuda.warpPerspective(gpu_right, H, warped_size)
        img_right_warped = gpu_right_warped.download()
        # (3) blending
        result = self.linear_blending(img_left, img_right_warped) if self.use_blending else img_left + img_right_warped
        # (4) remove black border
        result = self.remove_black_border(result)
        if flip:
            result = cv2.flip(result, 1)
        return result
