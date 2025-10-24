# 🧵 Image Stitching & Streaming Usage Guide

## 📂 Bag File Preparation

Before running any launch files, **ensure that `bags/camera.bag` exists**, if not:

```bash
source download_bags_sftp.sh
```

---

## 🚀 Usage Guide for Stitching

### ▶️ ROS Bag Playback

```bash
roscore
```

```bash
rosbag play bags/camera.bag --loop
```

### 🔧 Start Image Stitching

#### ➤ For Going JetSea - planar
```bash
roslaunch planar_processing planar_stitching_GJS.launch
```

#### ➤ For Going JetSea - cylindrical
```bash
roslaunch cylindrical_processing cylindrical_stitching.launch
```

#### ➤ For JS No.5 - planar
```bash
roslaunch planar_processing planar_stitching_JS5.launch
```

---

### ⚙️ Optimize FPS Performance

To improve frame rate:

- Edit `image_stitcher_GPU.launch`
- Adjust:
  - `stitch_timer_hz` (rostimer frequency)
  - Number of `ThreadPoolExecutor` workers

---

## 📺 Streaming Features

> Make sure to **update the `BASE_IP`** in both `XX_Relay.py` and `index.html` to match your local machine IP.

### ▶️ Start RSMP Streaming
```bash
source RSMP/start_streaming.sh
```

### ▶️ Start RTMP Streaming

> Update the base IP in related config files before launching.

```bash
source RTMP/start_streaming.sh
```

---

### 🛠 MediaMTX Configuration

To customize RTSP relay settings, mount your own configuration file:

```bash
RTSP/mediamtx.yml
```

---

### ✅ Check the Result

You can view the streaming result using the IP address set as `BASE_IP`.

#### 📡 RTMP

Make sure to update `BASE_IP` in both:

- `RTMP/FLV_stream_Relay.py`
- `RTMP/start_streaming.sh`

```bash
http://<BASE_IP>:8080
```

#### 📡 RTSP

Make sure to update `BASE_IP` in both:

- `RTSP/RTSP_stream_Relay.py`
- `RSMP/start_streaming.sh`

```bash
http://<BASE_IP>:8080
```
