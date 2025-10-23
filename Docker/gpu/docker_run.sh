#!/usr/bin/env bash

# 取得命令列參數
ARGS=("$@")

# 設定 Docker 映像名稱與標籤
USER_NAME="leowang707"  # 與 Dockerfile 內的 `USER` 一致
REPOSITORY="ros_noetic_cuda"  # Dockerfile 構建的映像名稱
TAG="plus"
IMG="${USER_NAME}/${REPOSITORY}:${TAG}"

# 設定 Docker 容器名稱
REPO_NAME="opencv-cuda-plus"
CONTAINER_NAME="ros_noetic_cuda_container"

# 檢查是否有已運行的容器
CONTAINER_ID=$(docker ps -aqf "name=^/${CONTAINER_NAME}$")
if [ -n "$CONTAINER_ID" ]; then
  echo "✅ 附加到已運行的 Docker 容器: $CONTAINER_NAME"
  xhost +
  docker exec --privileged -e DISPLAY=${DISPLAY} -e LINES="$(tput lines)" -it ${CONTAINER_ID} bash
  xhost -
  exit 0
fi

# 設定 X11 顯示變數，讓 GUI (如 RViz) 能夠顯示
XAUTH=/tmp/.docker.xauth
if [ ! -f $XAUTH ]; then
  xauth_list=$(xauth nlist $DISPLAY)
  xauth_list=$(sed -e 's/^..../ffff/' <<<"$xauth_list")
  if [ ! -z "$xauth_list" ]; then
    echo "$xauth_list" | xauth -f $XAUTH nmerge -
  else
    touch $XAUTH
  fi
  chmod a+r $XAUTH
fi

# 確保 XAUTH 成功建立，否則退出
if [ ! -f $XAUTH ]; then
  echo "⚠️  XAUTH ($XAUTH) 未能成功建立，退出..."
  exit 1
fi

# 啟動新的 Docker 容器
echo "🚀 啟動新的 Docker 容器: $CONTAINER_NAME"
docker run \
  -it \
  --runtime=nvidia \
  --gpus all \
  --ipc=host \
  -e DISPLAY=$DISPLAY \
  -e XAUTHORITY=$XAUTH \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e HOME=/home/${USER_NAME} \
  -e USER=${USER_NAME} \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -v "$XAUTH:$XAUTH" \
  -v "/tmp/.X11-unix:/tmp/.X11-unix" \
  -v "/etc/localtime:/etc/localtime:ro" \
  -v "/dev:/dev" \
  -v "/var/run/docker.sock:/var/run/docker.sock" \
  -v "/home/${USER}/${REPO_NAME}:/home/${USER_NAME}/${REPO_NAME}" \
  --user "root:root" \
  --workdir "/home/${USER_NAME}/${REPO_NAME}" \
  --name "${CONTAINER_NAME}" \
  --network host \
  --privileged \
  --rm \
  --security-opt seccomp=unconfined \
  "${IMG}" \
  bash
