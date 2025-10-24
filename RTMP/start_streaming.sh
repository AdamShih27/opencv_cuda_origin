#!/bin/bash

# 記錄執行腳本前的資料夾
ORIGINAL_DIR=$(pwd)

# 設定 `ROS-to-RTMP.py` 的絕對路徑
RTMP_SCRIPT="$HOME/opencv-cuda-plus/RTMP/FLV_stream_Relay.py"
WEB_ROOT="$HOME/opencv-cuda-plus/RTMP"

# 啟動 NGINX 服務
echo " 正在重啟 NGINX..."
service nginx restart
sleep 2  # 等待 NGINX 啟動

# 啟動 HTTP 伺服器 (提供 HLS)
echo " 啟動 HTTP 伺服器 (port 8000)..."
python3 -m http.server 8080 --directory "$WEB_ROOT" --bind 0.0.0.0 &
HTTP_SERVER_PID=$!
echo "Python HTTP server started with PID $HTTP_SERVER_PID"

# 啟動 ROS-to-RTMP
echo " 啟動 RTMP 推流..."
python3 "$RTMP_SCRIPT" &
RTMP_PID=$!
echo "FLV_stream_Relay.py started with PID $RTMP_PID"

# 提示所有服務已啟動
echo " 所有服務已啟動！"
echo " - HTTP Server: http://0.0.0.0:8080"
echo " - RTMP Streaming: Processing via ROS-to-RTMP.py"
echo "按 [CTRL+C] 來停止所有服務..."

# 設置 `trap`，當收到 `CTRL+C` (`SIGINT`) 或腳本退出 (`EXIT`) 時，停止所有服務
trap 'echo " 接收到 Ctrl+C，正在關閉所有服務..."; \
      kill $HTTP_SERVER_PID $RTMP_PID; \
      service nginx stop; \
      echo " 所有服務已關閉！"; \
      cd "$ORIGINAL_DIR"; \
      echo "Returned to original directory: $ORIGINAL_DIR";' SIGINT EXIT

# 主程序阻塞，等待所有後台程式結束
wait

# 所有服務結束後的提示
echo " 所有服務已完全結束！"

# 回到原本的資料夾
cd "$ORIGINAL_DIR"
echo "Returned to original directory: $ORIGINAL_DIR"

# 啟動交互式 Shell，不退出容器或終端
exec bash
