#!/usr/bin/env bash
USER_NAME="leowang707"
REPOSITORY="ros_noetic_cuda"  # Dockerfile 構建的映像名稱
TAG="plus"
IMG="${USER_NAME}/${REPOSITORY}:${TAG}"
docker pull "${IMG}"