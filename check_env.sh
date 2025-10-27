#!/usr/bin/env bash

echo "================= 基本資訊 ================="
echo "使用者: $(whoami)"
echo "主機名稱: $(hostname)"
lsb_release -d 2>/dev/null | awk -F'\t' '{print "作業系統: "$2}'
echo "內核版本: $(uname -r)"
echo ""

echo "================= Python 環境 ================="
python3 --version
pip3 --version
echo ""
echo "已安裝 Python 套件:"
pip3 list | grep -E "pynvjpeg|nvidia-nvjpeg" || echo "❌ pynvjpeg 或 runtime 未安装"
echo ""

echo "================= CUDA & GPU 驗證 ================="
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
else
  echo "❌ 無法找到 nvidia-smi"
fi
nvcc --version | grep release || echo "nvcc 未安装"
echo ""

echo "================= nvJPEG 功能測試 ================="
python3 - << 'PYCODE'
import sys, time, numpy as np
try:
    from nvjpeg import NvJpeg
    print("✅ NvJpeg 模組可用")
except Exception as e:
    print("❌ import nvjpeg 失敗:", e)
    sys.exit(1)

try:
    nv = NvJpeg()
    print("✅ NvJpeg 初始化成功")
except Exception as e:
    print("❌ NvJpeg 初始化錯誤:", e)
    sys.exit(1)

h, w = 1080, 1920
frame = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

t0 = time.time()
try:
    jpeg = nv.encode(frame, 90)
    t1 = time.time()
    print(f"✅ Encode 成功，JPEG bytes: {len(jpeg)}")
    print(f"Encode 時間: {(t1-t0)*1000:.1f} ms")
except Exception as e:
    print("❌ Encode 錯誤:", e)
    sys.exit(1)

t2 = time.time()
try:
    decoded = nv.decode(jpeg)
    t3 = time.time()
    print(f"✅ Decode 成功，shape: {decoded.shape}")
    print(f"Decode 時間: {(t3-t2)*1000:.1f} ms")
except Exception as e:
    print("❌ Decode 錯誤:", e)
    sys.exit(1)
PYCODE

echo ""
echo "================= 結論 ================="
echo "- 若 Encode/Decode 時間在幾毫秒內，且 GPU 使用率高，代表 nvJPEG 已通过 GPU 加速 :contentReference[oaicite:0]{index=0}"
echo "- 若执行成功但时间偏長，表示可能退回 CPU fallback 模式"
echo "- 若你使用 CUDA 11.7，你可再安装对应 runtime：pip3 install nvidia-nvjpeg-cu11"
echo ""

echo "✅ 檢查完成"
