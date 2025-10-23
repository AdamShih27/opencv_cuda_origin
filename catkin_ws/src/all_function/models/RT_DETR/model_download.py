from typing import List
from huggingface_hub import hf_hub_download
import shutil
import os

def download_models(repo_id: str, filenames: List[str], dest_dir: str) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    for filename in filenames:
        target = os.path.join(dest_dir, filename)
        if os.path.exists(target):
            print(f"⚠️ 已存在：{target}，跳過")
            continue
        print(f"⬇️ 下載模型：{repo_id}/{filename}")
        try:
            src = hf_hub_download(repo_id=repo_id, filename=filename)
            shutil.copy(src, target)
            print(f"✅ 複製到：{target}")
        except Exception as e:
            print(f"❌ 下載/複製失敗：{filename}，錯誤：{e}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo = "leowang707/RT_DETR_JS5"
    files = ["200_6_last.pt", "200_6_best.pt",
             "100_6_last.pt", "100_6_best.pt",
             "200_4_last.pt", "200_4_best.pt",
             "100_4_last.pt", "100_4_best.pt",
             "80_4_last.pt", "80_4_best.pt"]
    download_models(repo, files, script_dir)
