# -*- coding: utf-8 -*-
"""
cylindrical_stitching_core.py
-----------------------------

此模組實作 GPU 加速的圓柱投影影像拼接流程，適用於三相機全景拼接系統。

涵蓋模組：

1. CylindricalProjector：將影像以圓柱模型投影（避免透視變形）
2. AffineMatcher：使用 CUDA ORB 特徵點進行特徵比對並估算仿射矩陣
3. FeatherBlender：針對重疊區進行 alpha feathering 平滑融合
4. TripletStitcher：主流程，用於三張相機影像拼接為一張全景圖
5. crop_by_fixed_rect：簡單裁切工具

核心架構：
    左相機 ─仿射變換→ 中相機（基準） ←仿射變換─ 右相機
    並對重疊區進行 per-row alpha blending，實現平滑接縫。

輸出包含：
    - 拼接後 pano 影像
    - 左右仿射矩陣（若非固定輸入）
    - 左右影像在 pano 中的偏移（offset）

依賴：
    - OpenCV + CUDA 模組（cv2.cuda）
"""

import cv2
import numpy as np

# -----------------------------
# 1. Cylindrical 投影
# -----------------------------
class CylindricalProjector:
    def project(self, img: np.ndarray, f: float):
        """
        將影像以圓柱模型投影，消除側向透視變形

        參數:
        - img: 輸入 BGR 影像
        - f: 焦距（單位：像素）

        回傳:
        - warped: 投影後影像
        - mask: 有效區域遮罩
        """
        h, w = img.shape[:2]
        yy, xx = np.indices((h, w), np.float32)
        x_c, y_c = xx - w / 2.0, yy - h / 2.0
        theta, h_ = x_c / f, y_c / f
        X, Z = np.sin(theta), np.cos(theta)
        map_x = (f * X / Z + w / 2.0).astype(np.float32)
        map_y = (f * h_ / Z + h / 2.0).astype(np.float32)
        g_img = cv2.cuda_GpuMat(); g_mx = cv2.cuda_GpuMat(); g_my = cv2.cuda_GpuMat()
        g_img.upload(img); g_mx.upload(map_x); g_my.upload(map_y)
        warped = cv2.cuda.remap(
            g_img, g_mx, g_my,
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT
        ).download()
        mask = (warped.sum(axis=2) > 0).astype(np.uint8) * 255
        return warped, mask

# -----------------------------
# 2. ORB 特徵比對 + 仿射矩陣估算
# -----------------------------
class AffineMatcher:
    def __init__(self, max_features=2000, match_limit=500):
        self.orb = cv2.cuda.ORB_create(max_features)
        self.matcher = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_HAMMING)
        self.match_limit = match_limit

    def estimate(self, base_roi: np.ndarray, new_img: np.ndarray):
        """
        對比 base_roi 與 new_img，估算仿射矩陣 new_img → base_roi

        回傳：
        - 仿射矩陣 (2x3) 或 None
        """
        gb = cv2.cuda_GpuMat(); gn = cv2.cuda_GpuMat()
        gb.upload(cv2.cvtColor(base_roi, cv2.COLOR_BGR2GRAY))
        gn.upload(cv2.cvtColor(new_img,  cv2.COLOR_BGR2GRAY))
        kp1_g, des1_g = self.orb.detectAndComputeAsync(gb, None)
        kp2_g, des2_g = self.orb.detectAndComputeAsync(gn, None)
        kp1, kp2 = self.orb.convert(kp1_g), self.orb.convert(kp2_g)
        if des1_g.empty() or des2_g.empty():
            return None
        matches = self.matcher.match(des2_g, des1_g)
        matches = sorted(matches, key=lambda m: m.distance)[:self.match_limit]
        if len(matches) < 4:
            return None
        src = np.float32([kp2[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst = np.float32([kp1[m.trainIdx].pt  for m in matches]).reshape(-1,1,2)
        M, _ = cv2.estimateAffinePartial2D(src, dst, cv2.RANSAC)
        return M

# -----------------------------
# 3. 羽化混合器（重疊區平滑融合）
# -----------------------------
class FeatherBlender:
    def __init__(self, blend_width=20):
        self.blend_width = blend_width

    def blend(self, pano: np.ndarray, pano_msk: np.ndarray,
              warped: np.ndarray, warped_msk: np.ndarray,
              is_left: bool):
        """
        將 warped 圖像融合進 pano 當中，於重疊區進行羽化處理

        - pano: 全景圖（累積結果）
        - pano_msk: pano 的有效區遮罩
        - warped: 新投影圖像
        - warped_msk: 新圖遮罩
        - is_left: 表示融合方向（True: 左圖，False: 右圖）
        """
        h, W = pano.shape[:2]
        overlap = pano_msk & warped_msk
        if not overlap.any():
            pano[warped_msk] = warped[warped_msk]
            pano_msk[warped_msk] = True
            return pano, pano_msk

        alpha = np.zeros((h, W), np.float32)
        for y in np.where(overlap.any(axis=1))[0]:
            cols = np.where(overlap[y])[0]
            c0, c1 = int(cols[0]), int(cols[-1])
            center = (c0 + c1) // 2
            bw = self.blend_width // 2
            ls = max(center - bw, c0)
            re = min(center + bw, c1)
            width = re - ls + 1
            if is_left:
                alpha[y, c0:ls]      = 1.0
                alpha[y, re+1:c1+1]  = 0.0
                alpha[y, ls:re+1]    = np.linspace(1, 0, width) if width > 1 else 0.5
            else:
                alpha[y, c0:ls]      = 0.0
                alpha[y, re+1:c1+1]  = 1.0
                alpha[y, ls:re+1]    = np.linspace(0, 1, width) if width > 1 else 0.5
        alpha *= overlap.astype(np.float32)

        for c in range(3):
            pano[...,c] = np.where(overlap,
                (pano[...,c]*(1-alpha) + warped[...,c]*alpha).astype(np.uint8),
                pano[...,c])
        edge = warped_msk & ~pano_msk
        pano[edge] = warped[edge]
        pano_msk[edge] = True
        return pano, pano_msk

# -----------------------------
# 4. 圖像裁切工具
# -----------------------------
def crop_by_fixed_rect(img, rect):
    x, y, w, h = rect
    return img[y:y+h, x:x+w]

# -----------------------------
# 5. 三圖拼接主流程（Triplet Stitcher）
# -----------------------------
class TripletStitcher:
    def __init__(self, focals, blend_width=30, matcher_cfg=None, crop_rect=None):
        """
        初始化三圖拼接器

        - focals: [f_L, f_M, f_R] 三個相機焦距（像素）
        - blend_width: 羽化混合寬度（像素）
        - matcher_cfg: 傳給 AffineMatcher 的初始化參數
        - crop_rect: 拼接完成後的裁切區域
        """
        self.focals    = focals
        self.projector = CylindricalProjector()
        self.matcher   = AffineMatcher(**(matcher_cfg or {}))
        self.blender   = FeatherBlender(blend_width)
        self.crop_rect = crop_rect

    def stitch(self, imgs, frame_idx, fixed_M_L=None, fixed_M_R=None):
        """
        執行拼接流程

        - imgs: [left, mid, right] 三張原始影像
        - frame_idx: 當前幀編號（可做 debug）
        - fixed_M_L / M_R: 若已知左右仿射矩陣，可傳入固定值

        回傳：
        - pano_cropped: 拼接後並裁切的 pano 影像
        - new_M_L, new_M_R: 若自動估算仿射，則傳回計算結果
        - off: 中圖在 pano 中的水平起始位置
        """
        projs, masks = zip(*(self.projector.project(im, f) for im, f in zip(imgs, self.focals)))
        h, w = projs[0].shape[:2]; W = w * 3
        pano     = np.zeros((h, W, 3), np.uint8)
        pano_msk = np.zeros((h, W), bool)
        off = w
        pano[:, off:off+w]     = projs[1]; pano_msk[:, off:off+w] = masks[1].astype(bool)
        base = pano[:, off:off+w]
        new_M_L, new_M_R = None, None

        # === 拼左圖 ===
        if fixed_M_L is not None:
            M_L = fixed_M_L
        else:
            M_L = self.matcher.estimate(base, projs[0])
            new_M_L = M_L
        if M_L is not None:
            M_adj = M_L.copy().astype(np.float32); M_adj[0,2] += off
            g_img = cv2.cuda_GpuMat(); g_img.upload(projs[0])
            g_msk = cv2.cuda_GpuMat(); g_msk.upload(masks[0].astype(np.uint8))
            g_warped   = cv2.cuda.warpAffine(g_img, M_adj, (W, h))
            g_warped_m = cv2.cuda.warpAffine(g_msk, M_adj, (W, h))
            warped     = g_warped.download()
            warped_msk = g_warped_m.download().astype(bool)
            pano, pano_msk = self.blender.blend(pano, pano_msk, warped, warped_msk, True)

        # === 拼右圖 ===
        if fixed_M_R is not None:
            M_R = fixed_M_R
        else:
            M_R = self.matcher.estimate(base, projs[2])
            new_M_R = M_R
        if M_R is not None:
            M_adj = M_R.copy().astype(np.float32); M_adj[0,2] += off
            g_img = cv2.cuda_GpuMat(); g_img.upload(projs[2])
            g_msk = cv2.cuda_GpuMat(); g_msk.upload(masks[2].astype(np.uint8))
            g_warped   = cv2.cuda.warpAffine(g_img, M_adj, (W, h))
            g_warped_m = cv2.cuda.warpAffine(g_msk, M_adj, (W, h))
            warped     = g_warped.download()
            warped_msk = g_warped_m.download().astype(bool)
            pano, pano_msk = self.blender.blend(pano, pano_msk, warped, warped_msk, False)

        # 裁切 pano 區域（若有指定）
        if self.crop_rect is not None:
            x, y, w, h = self.crop_rect
            pano_cropped = pano[y:y+h, x:x+w]
        else:
            pano_cropped = pano
            
        return pano_cropped, new_M_L, new_M_R, off
