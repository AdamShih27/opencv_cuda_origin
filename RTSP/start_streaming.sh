#!/bin/bash

# 記錄執行腳本前的資料夾
ORIGINAL_DIR=$(pwd)

# 設定 MediaMTX 路徑
MTX_BIN="/opt/mediamtx/mediamtx"
MTX_YML="$HOME/opencv-cuda-plus/RTSP/mediamtx.yml"

# 設定 RTSP 推流腳本與網頁根目錄
RTSP_SCRIPT="$HOME/opencv-cuda-plus/RTSP/RTSP_stream_Relay.py"
WEB_ROOT="$HOME/opencv-cuda-plus/RTSP"

# 啟動 MediaMTX
echo "🟢 啟動 MediaMTX..."
"$MTX_BIN" "$MTX_YML" &
MTX_PID=$!
echo "MediaMTX 啟動，PID: $MTX_PID"

# 啟動 HTTP 伺服器 (提供網頁)
# echo "🟢 啟動 HTTP 伺服器 (port 8080)..."
# python3 -m http.server 8080 --directory "$WEB_ROOT" --bind 0.0.0.0 &
# HTTP_SERVER_PID=$!
# echo "HTTP 伺服器啟動，PID: $HTTP_SERVER_PID"

# 啟動 RTSP 推流腳本
echo "🟢 啟動 RTSP 推流腳本..."
python3 "$RTSP_SCRIPT" &
RTSP_PID=$!
echo "RTSP_stream_Relay.py 啟動，PID: $RTSP_PID"

# 提示
echo "✅ 所有服務已啟動！"
echo " - HTTP 網頁: http://localhost:8080"
echo " - RTSP 串流 via MediaMTX"
echo "按下 [CTRL+C] 可停止所有服務..."

# trap：當收到 Ctrl+C 或腳本結束時，自動關閉所有服務
trap 'echo "⚠️ 偵測到 Ctrl+C，正在關閉服務..."; \
      kill $HTTP_SERVER_PID $RTSP_PID $MTX_PID; \
      echo "✅ 所有服務已關閉"; \
      cd "$ORIGINAL_DIR"; \
      echo "返回原本目錄：$ORIGINAL_DIR";' SIGINT EXIT

# 等待所有背景程式結束
wait

# 返回原本資料夾
cd "$ORIGINAL_DIR"
echo "返回原本目錄：$ORIGINAL_DIR"

# 啟動互動式 shell
exec bash
