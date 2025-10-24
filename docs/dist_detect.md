# 🧵 Image Stitching & Streaming Usage Guide

## 📂 Bag File Preparation

Before running any launch files, **ensure that `bags/js5_triangle_path.bag` exists**, if not:

```bash
wget -nv -r -nd -np -nc -P bags \
     ftp://140.113.148.83/arg-projectfile-download/opencv_cuda/bags/
```

---

## 🚀 Usage Guide for Stitching

### ▶️ ROS Bag Playback

```bash
roscore
```

```bash
rosbag play bags/js5_triangle_path.bag --loop
```

### 🔧 Start Image Stitching for JS No.5

```bash
roslaunch FastHorizonAlg.launch
```

---

## 📊 Check the Result

Launch RViz and check the following topics:

```bash
rviz
```

Watch these topics to view the horizon outputs:

```
/cam1/horizon/compressed
/cam2/horizon/compressed
/cam3/horizon/compressed
/cam4/horizon/compressed
```