import torch
import numpy as np
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import cv2
import os
from pathlib import Path
import torch.nn.functional as F
import time

class SegFormerPredictor:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained(model_path)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_path).to(self.device)
        if self.device == "cuda":
            self.model = self.model.half()
        self.model.eval()
        self.label_map = self.model.config.id2label
        print(f"✅ SegFormer loaded from: {model_path}")
        print(f"🎯 Classes: {self.label_map}")

    def infer(self, image):
        return self.infer_batch([image])[0]

    def infer_batch(self, images):
        start_total = time.time()

        # ➤ 前處理：標準化輸入為 PIL，並記錄原圖尺寸
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

        # ➤ 預處理
        start_pre = time.time()
        inputs = self.feature_extractor(images=pil_images, return_tensors="pt")
        if self.model.dtype == torch.float16:
            inputs = {k: v.half().to(self.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        torch.cuda.synchronize() if self.device == "cuda" else None
        end_pre = time.time()

        # ➤ 推論
        start_infer = time.time()
        with torch.no_grad():
            outputs = self.model(**inputs)
        torch.cuda.synchronize() if self.device == "cuda" else None
        end_infer = time.time()

        # ➤ 後處理
        start_post = time.time()
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
        torch.cuda.synchronize() if self.device == "cuda" else None
        end_post = time.time()

        # ➤ 時間
        print(f"⏱️ Preprocessing: {(end_pre - start_pre)*1000:.2f} ms")
        print(f"⏱️ Inference:    {(end_infer - start_infer)*1000:.2f} ms")
        print(f"⏱️ Postprocess:  {(end_post - start_post)*1000:.2f} ms")
        print(f"⏱️ Total time:   {(end_post - start_total)*1000:.2f} ms")

        return masks_np

    def estimate_horizon_slope(self, mask, water_id=1, window_ratio=0.5, image=None, debug=False):
        h, w = mask.shape
        water_mask = np.uint8(mask == water_id)
        horizon_y = np.full(w, np.nan)

        for x in range(w):
            ys = np.where(water_mask[:, x] > 0)[0]
            if len(ys) > 0:
                horizon_y[x] = ys[0]

        mid_x = w // 2
        window_px = max(2, int(w * window_ratio))
        x_range = np.arange(mid_x - window_px, mid_x + window_px + 1)
        x_range = x_range[(x_range >= 0) & (x_range < w)]
        y_range = horizon_y[x_range]

        valid = ~np.isnan(y_range)
        x_fit = x_range[valid]
        y_fit = y_range[valid]

        if len(x_fit) < 2:
            print("❌ Not enough points for linear fit.")
            return None

        A = np.vstack([x_fit, np.ones_like(x_fit)]).T
        m, b = np.linalg.lstsq(A, y_fit, rcond=None)[0]
        center_y = int(m * mid_x + b)

        if debug and image is not None:
            if len(image.shape) == 2:
                vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                vis_img = image.copy()
            x1 = mid_x - window_px
            x2 = mid_x + window_px
            y1 = int(m * x1 + b)
            y2 = int(m * x2 + b)
            # cv2.line(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # cv2.circle(vis_img, (mid_x, center_y), 4, (0, 255, 255), -1)
            # cv2.putText(vis_img, f"Center Y: {center_y}px", (mid_x + 10, center_y - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            # cv2.imshow("Horizon with Slope and Center Height", vis_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        print(f"📐 Slope at center: {m:.4f}")
        print(f"📏 Center Y position: {center_y}px")
        return m


# === ✅ 測試區 ===
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "../catkin_ws/src/all_function/models/Segformer/segformer_model")

    # 資料夾名稱
    folder_names = ["_2025-06-10-12-41-45_0_left", "_2025-06-10-12-41-45_0_mid", "_2025-06-10-12-41-45_0_right"]
    folder_paths = [os.path.join(script_dir, "images", name) for name in folder_names]

    # 確認模型存在
    if not Path(model_path).exists():
        print(f"❌ 模型資料夾不存在：{model_path}")
        exit(1)

    # 檢查每個資料夾是否存在
    for folder_path in folder_paths:
        if not Path(folder_path).exists():
            print(f"❌ 缺少資料夾：{folder_path}")
            exit(1)

    # 讀取三個資料夾的圖檔名稱並排序（確保對應編號）
    filenames = sorted([
        f for f in os.listdir(folder_paths[0]) if f.lower().endswith(".png")
        and all(os.path.exists(os.path.join(p, f)) for p in folder_paths)
    ])

    if not filenames:
        print("⚠️ 三個資料夾中沒有對應的圖片")
        exit(1)

    # 初始化模型
    predictor = SegFormerPredictor(model_path=model_path)
    print(f"🚀 推論使用裝置：{predictor.device}")

    # 逐組推論（每次三張）
    for idx, fname in enumerate(filenames):
        image_paths = [os.path.join(p, fname) for p in folder_paths]
        images = [Image.open(p).convert("RGB") for p in image_paths]

        print(f"\n📂 第 {idx+1} 組圖片：{fname}")
        masks = predictor.infer_batch(images)

        for i, (img, mask, cam_name) in enumerate(zip(images, masks, folder_names)):
            print(f"🖼️ {cam_name} →", end=" ")
            predictor.estimate_horizon_slope(mask, image=np.array(img), debug=False)