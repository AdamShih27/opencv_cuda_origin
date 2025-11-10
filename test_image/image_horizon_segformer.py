#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal")

import os
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision
torchvision.disable_beta_transforms_warning()

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


def draw_horizon_on_image(image_bgr, slope, center_xy, color=(0, 0, 255), thickness=2):
    """
    依照 (slope, center) 在原圖上畫海平線：
      line: y = a*x + b；b = center_y - a*center_x
    """
    h, w = image_bgr.shape[:2]
    cx, cy = center_xy
    a = float(slope)
    b = float(cy - a * cx)

    # 找兩個位於圖內的端點
    pts = [(0, int(round(b))),
           (w - 1, int(round(a * (w - 1) + b)))]
    if abs(a) > 1e-9:
        pts += [(int(round(-b / a)), 0),
                (int(round((h - 1 - b) / a)), h - 1)]
    in_pts = [(x, y) for (x, y) in pts if 0 <= x < w and 0 <= y < h]

    if len(in_pts) >= 2:
        p0, p1 = in_pts[0], None
        for q in in_pts[1:]:
            if q != p0:
                p1 = q
                break
        if p1 is None:
            p0 = (0, max(0, min(h - 1, int(round(b)))))
            p1 = (w - 1, max(0, min(h - 1, int(round(a * (w - 1) + b)))))
    else:
        p0 = (0, max(0, min(h - 1, int(round(b)))))
        p1 = (w - 1, max(0, min(h - 1, int(round(a * (w - 1) + b)))))

    out = image_bgr.copy()
    cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)
    cv2.circle(out, (int(cx), int(cy)), 4, (0, 255, 0), -1, cv2.LINE_AA)
    return out


class SegFormerPredictor:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.feature_extractor = SegformerImageProcessor.from_pretrained(model_path)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_path).to(self.device)
        if self.device == "cuda":
            self.model = self.model.half()
        self.model.eval()
        self.label_map = self.model.config.id2label

    def infer_batch(self, images):
        pil_images, original_sizes = [], []
        for img in images:
            if isinstance(img, np.ndarray):
                original_sizes.append(img.shape[:2])  # (H, W)
                pil_images.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            elif isinstance(img, Image.Image):
                original_sizes.append(img.size[::-1])  # (H, W)
                pil_images.append(img)
            else:
                raise ValueError("Each image must be a numpy array (BGR) or PIL Image")

        inputs = self.feature_extractor(images=pil_images, return_tensors="pt")
        if self.model.dtype == torch.float16:
            inputs = {k: v.half().to(self.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        masks = torch.argmax(logits, dim=1)

        masks_np = []
        for i in range(masks.shape[0]):
            resized = F.interpolate(
                masks[i].unsqueeze(0).unsqueeze(0).float(),
                size=original_sizes[i],
                mode="nearest"
            )[0, 0].cpu().numpy().astype(np.uint8)
            masks_np.append(resized)
        return masks_np

    @staticmethod
    def estimate_horizon_from_mask(mask, water_id=1, window_ratio=0.5,
                                   roi_ratio=(0.05, 0.95, 0.20, 0.75)):
        """
        在 mask（原圖大小）上估算海平線。
        只在 ROI 內搜尋並擬合（避免雜訊/邊界干擾）。
        回傳: {'slope': m, 'center': (cx, cy)} 或 None

        roi_ratio: (x0, x1, y0, y1)，各為 0~1 比例（含邊界），例如
                   (0.05, 0.95, 0.20, 0.75) 表示中間 90% 寬度、20%~75% 高度。
        """
        h, w = mask.shape
        water = (mask == water_id).astype(np.uint8)

        # 解析 ROI（轉成像素座標並裁切到有效範圍）
        x0r, x1r, y0r, y1r = roi_ratio
        x0 = max(0, min(w - 1, int(round(w * float(x0r)))))
        x1 = max(0, min(w - 1, int(round(w * float(x1r))) - 1))
        y0 = max(0, min(h - 1, int(round(h * float(y0r)))))
        y1 = max(0, min(h - 1, int(round(h * float(y1r))) - 1))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None

        # 只在 ROI 內由上往下找每個 x 的第一個水域像素
        horizon_y = np.full(w, np.nan, dtype=np.float32)
        for x in range(x0, x1 + 1):
            col = water[y0:y1 + 1, x]
            ys = np.where(col > 0)[0]
            if len(ys) > 0:
                horizon_y[x] = y0 + ys[0]

        # 在 ROI 中央的水平視窗做擬合（window_ratio）
        mid_x = (x0 + x1) // 2
        win_half = max(2, int((x1 - x0 + 1) * float(window_ratio) * 0.5))
        x_range = np.arange(mid_x - win_half, mid_x + win_half + 1)
        x_range = x_range[(x_range >= x0) & (x_range <= x1)]
        y_range = horizon_y[x_range]

        valid = ~np.isnan(y_range)
        if valid.sum() < 2:
            return None

        x_fit = x_range[valid]
        y_fit = y_range[valid]
        A = np.vstack([x_fit, np.ones_like(x_fit)]).T
        m, b = np.linalg.lstsq(A, y_fit, rcond=None)[0]

        center_y = int(round(m * mid_x + b))
        center_y = int(np.clip(center_y, 0, h - 1))
        return {"slope": float(m), "center": (int(mid_x), center_y)}


def main():
    # ====== 手動設定參數 ======
    model_path = "models/Segformer/segformer_model"  # 模型路徑
    image_dir  = "images/test"                       # 輸入資料夾
    out_dir    = "images/output"                     # 輸出資料夾
    water_id   = 1                                   # 水域 label id
    window_ratio = 0.5                               # 擬合視窗比例（0~1）
    roi_ratio = (0.0, 1.0, 0.25, 0.75)             # ★ 只用中間 90% 寬度、25%~75% 高度
    # =========================================

    os.makedirs(out_dir, exist_ok=True)

    # 初始化 SegFormer 模型
    predictor = SegFormerPredictor(model_path)

    # 過濾資料夾內的圖片檔
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    image_list = [f for f in os.listdir(image_dir) if f.lower().endswith(exts)]
    if not image_list:
        raise FileNotFoundError(f"資料夾 {image_dir} 沒有圖片！")

    for fname in image_list:
        image_path = os.path.join(image_dir, fname)
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[警告] 讀不到圖片：{image_path}，跳過。")
            continue

        # 分割
        mask = predictor.infer_batch([img_bgr])[0]  # 全圖 mask

        # 1) 輸出二值水域 mask
        water_mask = (mask == water_id).astype(np.uint8) * 255
        stem = os.path.splitext(fname)[0]
        out_mask_path = os.path.join(out_dir, f"{stem}_watermask_binary.png")
        cv2.imwrite(out_mask_path, water_mask)

        # 2) 在原圖畫海平線（使用 ROI 限制）
        horizon_info = predictor.estimate_horizon_from_mask(
            mask, water_id=water_id, window_ratio=window_ratio, roi_ratio=roi_ratio
        )

        vis_img = img_bgr.copy()
        if horizon_info is not None:
            m = horizon_info["slope"]
            cx, cy = horizon_info["center"]
            vis_img = draw_horizon_on_image(
                vis_img, slope=m, center_xy=(cx, cy),
                color=(0, 0, 255), thickness=2
            )
            cv2.putText(
                vis_img, f"slope={m:.4f}, center=({cx},{cy})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 240, 60), 2, cv2.LINE_AA
            )
        else:
            cv2.putText(
                vis_img, "無法估算海平線（有效水域像素不足或 ROI 太窄）",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA
            )

        out_horizon_path = os.path.join(out_dir, f"{stem}_horizon_full.png")
        cv2.imwrite(out_horizon_path, vis_img)

        print(f"[完成] {fname}")
        print(f"  - 水域 mask：{out_mask_path}")
        print(f"  - 原圖+海平線：{out_horizon_path}")


if __name__ == "__main__":
    main()
