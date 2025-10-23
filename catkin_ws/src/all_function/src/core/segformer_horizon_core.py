import warnings
import torchvision

# 忽略 NumPy 浮點數 subnormal 警告
warnings.filterwarnings("ignore", message="The value of the smallest subnormal")

# 關閉 torchvision beta API 警告
torchvision.disable_beta_transforms_warning()

import torch
import numpy as np
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import torch.nn.functional as F


class SegFormerPredictor:
    def __init__(self, model_path, roi_ratio=None):
        """
        Args:
            model_path (str): SegFormer 模型路徑
            roi_ratio (list[float]): [x0, x1, y0, y1]，相對於影像寬高 (0~1)
                                     None 時預設 [0.3, 0.7, 0.3, 0.7]
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.feature_extractor = SegformerImageProcessor.from_pretrained(model_path)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_path).to(self.device)
        if self.device == "cuda":
            self.model = self.model.half()
        self.model.eval()
        self.label_map = self.model.config.id2label

        # ROI 設定
        self.roi_ratio = roi_ratio if roi_ratio is not None else [0.3, 0.7, 0.3, 0.7]

    def infer_batch(self, images):
        pil_images = []
        original_sizes = []
        for img in images:
            if isinstance(img, np.ndarray):
                original_sizes.append(img.shape[:2])  # (H, W)
                pil_images.append(Image.fromarray(img))
            elif isinstance(img, Image.Image):
                original_sizes.append(img.size[::-1])  # PIL (W, H) → (H, W)
                pil_images.append(img)
            else:
                raise ValueError("Each image must be a numpy array or PIL Image")

        inputs = self.feature_extractor(images=pil_images, return_tensors="pt")
        if self.model.dtype == torch.float16:
            inputs = {k: v.half().to(self.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits  # (B, C, H, W)
        masks = torch.argmax(logits, dim=1)  # (B, H, W)

        masks_np = []
        for i in range(masks.shape[0]):
            resized = F.interpolate(
                masks[i].unsqueeze(0).unsqueeze(0).float(),
                size=original_sizes[i],
                mode="nearest"
            )[0, 0].cpu().numpy().astype(np.uint8)
            masks_np.append(resized)

        return masks_np

    def estimate_horizon_slope(self, mask, water_id=1, window_ratio=0.5):
        """
        在 ROI 範圍內估算海平線斜率
        """
        h, w = mask.shape
        x0, x1, y0, y1 = self.roi_ratio
        x0, x1 = int(w * x0), int(w * x1)
        y0, y1 = int(h * y0), int(h * y1)

        roi_mask = mask[y0:y1, x0:x1]
        water_mask = np.uint8(roi_mask == water_id)
        roi_w = x1 - x0

        horizon_y = np.full(roi_w, np.nan)
        for x in range(roi_w):
            ys = np.where(water_mask[:, x] > 0)[0]
            if len(ys) > 0:
                horizon_y[x] = ys[0]

        mid_x = roi_w // 2
        window_px = max(2, int(roi_w * window_ratio))
        x_range = np.arange(mid_x - window_px, mid_x + window_px + 1)
        x_range = x_range[(x_range >= 0) & (x_range < roi_w)]
        y_range = horizon_y[x_range]

        valid = ~np.isnan(y_range)
        x_fit = x_range[valid]
        y_fit = y_range[valid]

        if len(x_fit) < 2:
            return None

        A = np.vstack([x_fit, np.ones_like(x_fit)]).T
        m, b = np.linalg.lstsq(A, y_fit, rcond=None)[0]

        # 轉回全圖座標系
        center_y = int(m * mid_x + b) + y0
        mid_x_global = mid_x + x0

        return {
            "slope": float(m),
            "center": (int(mid_x_global), center_y)
        }

    def infer_horizon_slope_batch(self, images, water_id=1, window_ratio=0.5):
        """
        批次推論 + ROI 內估算海平線
        """
        masks = self.infer_batch(images)
        horizon_info_list = []
        for mask in masks:
            result = self.estimate_horizon_slope(mask, water_id=water_id, window_ratio=window_ratio)
            horizon_info_list.append(result)
        return horizon_info_list
