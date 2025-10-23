FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV USER=leowang707
ENV HOME=/home/${USER}
ENV OPENCV_VERSION=4.5.2
ARG MTX_VERSION=1.11.3

# --------------------- 基礎系統套件 ---------------------
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && apt-get -y dist-upgrade && \
    apt-get install -y --no-install-recommends \
    tzdata ca-certificates sudo locales curl gnupg2 wget git \
    build-essential cmake unzip pkg-config make \
    vim gedit net-tools xdotool x11-utils unclutter wmctrl \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev libgtk-3-dev \
    libatlas-base-dev gfortran python3-dev python3-pip python3-numpy \
    python3-venv python3-setuptools pciutils software-properties-common \
    lsb-release dirmngr mercurial libpcap-dev \
    libcanberra-gtk-module libcanberra-gtk3-module \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# --------------------- 建立非 root 使用者 ---------------------
RUN useradd -m -s /bin/bash ${USER} && \
    usermod -aG sudo ${USER} && \
    echo "root ALL=(ALL) ALL" >> /etc/sudoers && \
    echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# # --------------------- 安裝 GStreamer ---------------------
RUN add-apt-repository universe && apt-get update && \
    apt-get install -y --no-install-recommends \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-ugly \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-tools gstreamer1.0-libav gstreamer1.0-gl \
    gstreamer1.0-gtk3 gstreamer1.0-x libgstreamer1.0-0 \
    libgstrtspserver-1.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# --------------------- 下載 & 編譯 OpenCV ---------------------
RUN git clone -b ${OPENCV_VERSION} --depth 1 https://github.com/opencv/opencv.git /opencv && \
    git clone -b ${OPENCV_VERSION} --depth 1 https://github.com/opencv/opencv_contrib.git /opencv_contrib && \
    mkdir -p /opencv/build

WORKDIR /opencv/build

RUN export CUDA_ARCH_BIN=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | awk '{printf "%.1f ", $1}') && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
          -D WITH_CUDA=ON \
          -D ENABLE_FAST_MATH=1 \
          -D CUDA_FAST_MATH=1 \
          -D WITH_CUBLAS=1 \
          -D WITH_CUDNN=ON \
          -D OPENCV_DNN_CUDA=ON \
          -D CUDA_ARCH_BIN="${CUDA_ARCH_BIN}" \
          -D BUILD_EXAMPLES=OFF \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          -D PYTHON_EXECUTABLE=/usr/bin/python3 \
          -D OPENCV_GENERATE_PKGCONFIG=YES \
          -D BUILD_opencv_python2=OFF \
          -D BUILD_opencv_python3=ON \
          -D WITH_GSTREAMER=ON \
          .. && \
    make -j4 && make install && ldconfig

RUN python3 -c "import cv2; print('OpenCV:', cv2.__version__, 'CUDA:', cv2.cuda.getCudaEnabledDeviceCount() > 0)"

# --------------------- 安裝 ROS Noetic ---------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg2 lsb-release apt-transport-https ca-certificates && \
    echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" | tee /etc/apt/sources.list.d/ros-latest.list && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add - && \
    apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-base \
    ros-noetic-vision-opencv \
    python3-rosdep && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN rosdep init && rosdep update --rosdistro noetic

# --------------------- 安裝 ROS 擴充套件 ---------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-image-transport \
    ros-noetic-image-transport-plugins \
    ros-noetic-compressed-image-transport \
    ros-noetic-theora-image-transport \
    ros-noetic-rviz \
    ros-noetic-rviz-visual-tools \
    ros-noetic-rqt-rviz \
    ros-noetic-image-view \
    ros-noetic-robot-state-publisher \
    ros-noetic-joint-state-publisher \
    ros-noetic-xacro \
    ros-noetic-urdf \
    ros-noetic-urdf-parser-plugin \
    ros-noetic-tf2-tools \
    ros-noetic-rqt \
    ros-noetic-rqt-common-plugins \
    ros-noetic-rqt-image-view \
    ros-noetic-rqt-robot-plugins \
    ros-noetic-rqt-tf-tree && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# --------------------- 安裝 catkin 工具包 ---------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-catkin-tools python3-osrf-pycommon && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# --------------------- 安裝 Python 擴充 ---------------------
RUN pip3 install tqdm websocket-server websocket-client ffmpeg-python

# ✅ 安裝 Tkinter：移到後面避免干擾 OpenCV 編譯
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-tk && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# --------------------- 安裝 FFmpeg 播放支援 ---------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libavcodec-dev libavformat-dev libavutil-dev \
    libswscale-dev libavfilter-dev libavdevice-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# --------------------- 安裝 nginx + RTMP 模組 ---------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx vim libnginx-mod-rtmp && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# --------------------- 安裝 MediaMTX ---------------------
RUN mkdir -p /opt/mediamtx && \
    curl -L https://github.com/bluenviron/mediamtx/releases/download/v${MTX_VERSION}/mediamtx_v${MTX_VERSION}_linux_amd64.tar.gz \
    | tar xz -C /opt/mediamtx && \
    ln -s /opt/mediamtx/mediamtx /usr/local/bin/mediamtx

# --------------------- 安裝額外套件 ---------------------
RUN pip3 install mss

# --------------------- 安裝 pynvjpeg ---------------------
RUN pip3 install --upgrade pip setuptools wheel && \
    pip3 install "importlib_metadata<5" && \
    pip3 install pynvjpeg

# --------------------- 安裝 RT-DETR PyTorch 所需套件 ---------------------
RUN pip3 install \
    torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --extra-index-url https://download.pytorch.org/whl/cu117 && \
    pip3 install \
    onnx==1.14.0 \
    onnxruntime-gpu==1.15.1 \
    onnxsim \
    tensorrt==8.5.2.2 && \
    pip3 install \
    faster-coco-eval>=1.6.5 \
    PyYAML \
    tensorboard \
    scipy \
    pycocotools
    
# 1. 刪除 /usr/local/bin 的 cmake，讓系統優先使用 apt 的 cmake
RUN rm -f /usr/local/bin/cmake* /usr/local/share/cmake* || true

# 2. 重新安裝系統版 cmake
RUN apt-get update && apt-get install --reinstall -y cmake

# 3. 鎖定版本，防止被升級
RUN apt-mark hold cmake

# 檢查 CMake 版本是否 > 3.5，若是則報錯中止
RUN cmake_ver=$(cmake --version | head -n1 | awk '{print $3}') && \
    printf "✅ 檢測到 CMake 版本：%s\n" "$cmake_ver" && \
    if python3 -c "import sys; sys.exit(0) if tuple(map(int, '$cmake_ver'.split('.'))) <= (3,16,3) else sys.exit(1)"; then \
        echo "✅ CMake 版本符合 ≤ 3.16.3 要求"; \
    else \
        echo "❌ CMake 版本過新（$cmake_ver），請降版本到 3.16.3 " && exit 1; \
    fi

# --------------------- 安裝 RT-DETR PyTorch 相關套件 ---------------------
RUN pip3 install ultralytics --no-deps

# --------------------- 安裝 tf-conversions 與其他常見依賴（可擴充） ---------------------
RUN apt update && apt install -y \
    ros-noetic-tf-conversions \
    && rm -rf /var/lib/apt/lists/*

# --------------------- 安裝 transformers 與 accelerate ---------------------
RUN pip install transformers accelerate \
    cupy

# --------------------- 設定 ROS + bashrc 環境變數 ---------------------
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "source ${HOME}/opencv-cuda-plus/catkin_ws/devel/setup.bash" >> /root/.bashrc && \
    echo "export ROS_PACKAGE_PATH=${HOME}/opencv-cuda-plus/catkin_ws/src:\$ROS_PACKAGE_PATH" >> /root/.bashrc && \
    echo "export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:\$PYTHONPATH" >> /root/.bashrc && \
    echo "source /opt/ros/noetic/setup.bash" >> ${HOME}/.bashrc && \
    echo "source ${HOME}/opencv-cuda-plus/catkin_ws/devel/setup.bash" >> ${HOME}/.bashrc && \
    echo "export ROS_PACKAGE_PATH=${HOME}/opencv-cuda-plus/catkin_ws/src:\$ROS_PACKAGE_PATH" >> ${HOME}/.bashrc && \
    echo "export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:\$PYTHONPATH" >> ${HOME}/.bashrc && \
    echo "cp ${HOME}/opencv-cuda-plus/RTMP/nginx.conf /etc/nginx/nginx.conf" >> ~/.bashrc

# 再次設定用戶權限（確保 ${USER} 擁有 sudo 權限）
RUN echo "root ALL=(ALL) ALL" >> /etc/sudoers && \
    echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

CMD ["bash"]