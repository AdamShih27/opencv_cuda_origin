from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import os

# ✅ 設定目標資料夾（與程式碼同層）
save_dir = os.path.join(os.path.dirname(__file__), "segformer_model")

# ✅ 建立資料夾（若不存在）
os.makedirs(save_dir, exist_ok=True)

# ✅ 指定模型名稱
model_name = "Wilbur1240/segformer-b0-finetuned-ade-512-512-finetune-mastr1325-v2"

# ✅ 下載 FeatureExtractor 與 Model 至本地資料夾
feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
feature_extractor.save_pretrained(save_dir)

model = SegformerForSemanticSegmentation.from_pretrained(model_name)
model.save_pretrained(save_dir)

print(f"✅ 模型已下載至: {save_dir}")
