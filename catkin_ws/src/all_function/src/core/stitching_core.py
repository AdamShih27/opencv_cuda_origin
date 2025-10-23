import cv2
import numpy as np
import cupy as cp
import time

class CylindricalProjector:
    def __init__(self):
        self.cache = {}

    def project_gpu(self, img: np.ndarray, f: float):
        h, w = img.shape[:2]
        cache_key = (h, w, f)
        if cache_key not in self.cache:
            yy, xx = np.indices((h, w), np.float32)
            x_c, y_c = xx - w / 2.0, yy - h / 2.0
            theta, h_ = x_c / f, y_c / f
            X, Z = np.sin(theta), np.cos(theta)
            map_x = (f * X / Z + w / 2.0).astype(np.float32)
            map_y = (f * h_ / Z + h / 2.0).astype(np.float32)
            self.cache[cache_key] = (map_x, map_y)
        else:
            map_x, map_y = self.cache[cache_key]
        g_img = cv2.cuda_GpuMat(); g_mx = cv2.cuda_GpuMat(); g_my = cv2.cuda_GpuMat()
        g_img.upload(img); g_mx.upload(map_x); g_my.upload(map_y)
        warped_gpu = cv2.cuda.remap(
            g_img, g_mx, g_my,
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT
        )
        warped = warped_gpu.download()
        mask = (warped.sum(axis=2) > 0).astype(np.uint8) * 255
        return warped_gpu, mask

class AffineMatcher:
    def __init__(self, max_features=2000, match_limit=500):
        self.orb = cv2.cuda.ORB_create(max_features)
        self.matcher = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_HAMMING)
        self.match_limit = match_limit

    def estimate(self, base_roi: np.ndarray, new_img: np.ndarray):
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

class FeatherBlender:
    def __init__(self, blend_width=30):
        self.blend_width = blend_width

    def blend(self, pano, pano_msk, warped, warped_msk, is_left: bool):
        h, W = pano.shape[:2]
        pano_gpu    = cp.asarray(pano)
        warped_gpu  = cp.asarray(warped)
        pano_msk_gpu   = cp.asarray(pano_msk)
        warped_msk_gpu = cp.asarray(warped_msk)

        overlap_gpu = pano_msk_gpu & warped_msk_gpu
        edge_gpu    = warped_msk_gpu & (~pano_msk_gpu)

        row_any = overlap_gpu.any(axis=1)
        rows = cp.where(row_any)[0]
        if rows.size == 0:
            pano_gpu[edge_gpu] = warped_gpu[edge_gpu]
            pano_msk_gpu[edge_gpu] = True
            pano = cp.asnumpy(pano_gpu)
            pano_msk = cp.asnumpy(pano_msk_gpu)
            return pano, pano_msk

        idx_mat = cp.arange(W)
        c0 = cp.full(h, -1, dtype=cp.int32)
        c1 = cp.full(h, -1, dtype=cp.int32)
        c0[rows] = cp.argmax(overlap_gpu[rows], axis=1)
        c1[rows] = W - 1 - cp.argmax(overlap_gpu[rows][:, ::-1], axis=1)
        alpha_gpu = cp.zeros((h, W), dtype=cp.float32)
        bw = self.blend_width // 2

        for side in [True, False]:
            select = rows if is_left == side else None
            if select is None or len(select) == 0:
                continue
            s_c0 = c0[select]
            s_c1 = c1[select]
            s_center = (s_c0 + s_c1) // 2
            s_ls = cp.maximum(s_center - bw, s_c0)
            s_re = cp.minimum(s_center + bw, s_c1)
            all_cols = cp.broadcast_to(idx_mat, (len(select), W))
            m = overlap_gpu[select]
            mask_l = (all_cols >= s_ls[:, None]) & (all_cols <= s_re[:, None]) & m
            width = (s_re - s_ls + 1)
            slope = cp.where(width[:, None] > 1,
                (s_re[:, None] - all_cols + 0.0) / cp.clip(width[:, None] - 1, 1, None),
                0.5)
            if is_left:
                alpha_gpu[select] += slope * mask_l
                alpha_gpu[select] += ((all_cols >= s_c0[:, None]) & (all_cols < s_ls[:, None]) & m) * 1.0
            else:
                alpha_gpu[select] += (1.0 - slope) * mask_l
                alpha_gpu[select] += ((all_cols > s_re[:, None]) & (all_cols <= s_c1[:, None]) & m) * 1.0
        alpha_gpu *= overlap_gpu.astype(cp.float32)
        alpha_gpu = alpha_gpu[..., None]

        pano_gpu[overlap_gpu] = (
            pano_gpu[overlap_gpu] * (1 - alpha_gpu[overlap_gpu]) +
            warped_gpu[overlap_gpu] * alpha_gpu[overlap_gpu]
        ).astype(cp.uint8)
        pano_gpu[edge_gpu] = warped_gpu[edge_gpu]
        pano_msk_gpu[edge_gpu] = True
        pano = cp.asnumpy(pano_gpu)
        pano_msk = cp.asnumpy(pano_msk_gpu)
        return pano, pano_msk

class TripletStitcher:
    def __init__(self, focals, blend_width=30, matcher_cfg=None, crop_rect=None):
        self.focals    = focals
        self.projector = CylindricalProjector()
        self.matcher   = AffineMatcher(**(matcher_cfg or {}))
        self.blender   = FeatherBlender(blend_width)
        self.crop_rect = crop_rect

    def stitch(self, imgs, fixed_M_L=None, fixed_M_R=None):
        t0 = time.time()

        # Cylindrical projection
        projs_gpu, masks = [], []
        for im, f in zip(imgs, self.focals):
            g_proj, msk = self.projector.project_gpu(im, f)
            projs_gpu.append(g_proj)
            masks.append(msk)
        t1 = time.time()

        h, w = masks[0].shape[:2]; W = w * 3
        pano     = cp.zeros((h, W, 3), dtype=cp.uint8)
        pano_msk = cp.zeros((h, W), dtype=bool)
        off = w

        # Middle assignment
        mid_img = projs_gpu[1].download()
        pano[:, off:off+w, :] = cp.asarray(mid_img)
        pano_msk[:, off:off+w] = cp.asarray(masks[1].astype(bool))
        base = mid_img
        t2 = time.time()

        new_M_L, new_M_R = None, None

        # Left image
        t_l1 = time.time()
        if fixed_M_L is not None:
            M_L = fixed_M_L
        else:
            M_L = self.matcher.estimate(base, projs_gpu[0].download())
            new_M_L = M_L
        t_l2 = time.time()
        if M_L is not None:
            M_adj = M_L.copy().astype(np.float32); M_adj[0,2] += off
            g_msk = cv2.cuda_GpuMat(); g_msk.upload(masks[0].astype(np.uint8))
            g_warped   = cv2.cuda.warpAffine(projs_gpu[0], M_adj, (W, h))
            g_warped_m = cv2.cuda.warpAffine(g_msk, M_adj, (W, h))
            warped     = g_warped.download()
            warped_msk = g_warped_m.download().astype(bool)
            t_l3 = time.time()
            pano, pano_msk = self.blender.blend(pano, pano_msk, warped, warped_msk, True)
            t_l4 = time.time()
        else:
            t_l3 = t_l4 = time.time()

        # Right image
        t_r1 = time.time()
        if fixed_M_R is not None:
            M_R = fixed_M_R
        else:
            M_R = self.matcher.estimate(base, projs_gpu[2].download())
            new_M_R = M_R
        t_r2 = time.time()
        if M_R is not None:
            M_adj = M_R.copy().astype(np.float32); M_adj[0,2] += off
            g_msk = cv2.cuda_GpuMat(); g_msk.upload(masks[2].astype(np.uint8))
            g_warped   = cv2.cuda.warpAffine(projs_gpu[2], M_adj, (W, h))
            g_warped_m = cv2.cuda.warpAffine(g_msk, M_adj, (W, h))
            warped     = g_warped.download()
            warped_msk = g_warped_m.download().astype(bool)
            t_r3 = time.time()
            pano, pano_msk = self.blender.blend(pano, pano_msk, warped, warped_msk, False)
            t_r4 = time.time()
        else:
            t_r3 = t_r4 = time.time()

        # Cropping
        if self.crop_rect is not None:
            x, y, w, h = self.crop_rect
            pano_cropped = pano[y:y+h, x:x+w, :]
        else:
            pano_cropped = pano
        t3 = time.time()

        # print(
        #     f"[PROFILE] Cylindrical={t1-t0:.3f}s | "
        #     f"Mid-copy={t2-t1:.3f}s | "
        #     f"L-matcher={t_l2-t_l1:.3f}s, L-warp={t_l3-t_l2:.3f}s, L-blend={t_l4-t_l3:.3f}s | "
        #     f"R-matcher={t_r2-t_r1:.3f}s, R-warp={t_r3-t_r2:.3f}s, R-blend={t_r4-t_r3:.3f}s | "
        #     f"Crop={t3-t_r4:.3f}s, TOTAL={t3-t0:.3f}s"
        # )

        pano_cropped = cp.asnumpy(pano_cropped)
        return pano_cropped, new_M_L, new_M_R