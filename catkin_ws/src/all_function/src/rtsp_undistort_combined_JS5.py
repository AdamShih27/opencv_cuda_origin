#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import rospy
import time
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from nvjpeg import NvJpeg

class RTSPCameraPublisher:
    def __init__(self, name, rtsp_url, topic_name):
        self.name = name
        self.rtsp_url = rtsp_url
        self.topic_name = topic_name
        self.publisher = rospy.Publisher(topic_name, CompressedImage, queue_size=1)

        self.nvjpeg = NvJpeg()

        # H.264 解碼 pipeline
        self.pipeline = (
            f"rtspsrc location={rtsp_url} latency=0 protocols=udp drop-on-latency=true ! "
            "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! "
            "videoconvert n-threads=2 ! video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )

        # H.265 解碼 pipeline
        # self.pipeline = (
        #     f"rtspsrc location={rtsp_url} latency=0 protocols=udp drop-on-latency=true ! "
        #     "rtph265depay ! h265parse ! avdec_h265 ! videoconvert ! "
        #     "videoconvert n-threads=2 ! video/x-raw,format=BGR ! "
        #     "appsink drop=true max-buffers=1 sync=false"
        # )


        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            rospy.logerr(f"[{self.name}] ❌ 無法開啟 RTSP 串流：{rtsp_url}")
            self.cap = None

        self.thread = Thread(target=self.stream_loop)
        self.thread.daemon = True
        self.running = True

    def start(self):
        if self.cap:
            self.thread.start()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

    def stream_loop(self):
        while not rospy.is_shutdown() and self.running:
            if self.cap is None:
                time.sleep(1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                rospy.logwarn(f"[{self.name}] ⚠️ 讀取 RTSP 影格失敗")
                time.sleep(0.1)
                continue

            try:
                jpeg_bytes = self.nvjpeg.encode(frame, 100)
                msg = CompressedImage()
                msg.header.stamp = rospy.Time.now()
                msg.format = "jpeg"
                msg.data = jpeg_bytes
                self.publisher.publish(msg)
            except Exception as e:
                rospy.logwarn(f"[{self.name}] ❌ nvJPEG 編碼失敗: {e}")
                time.sleep(0.05)

class CameraUndistorter:
    def __init__(self, input_topic, output_topic, dim, K, D,
                 balance=1.0, crop_fov_deg=110.0, full_hfov=150.0):
        self.dim = dim
        self.K = K
        self.D = D
        self.crop_fov_deg = crop_fov_deg
        self.full_hfov = full_hfov
        self.bridge = CvBridge()
        self.nj = NvJpeg()
        self.executor = ThreadPoolExecutor(max_workers=1)

        self.new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.K, self.D, self.dim, np.eye(3), balance=balance)

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, np.eye(3), self.new_K, self.dim, cv2.CV_32FC1)
        self.gpu_map1 = cv2.cuda_GpuMat(); self.gpu_map2 = cv2.cuda_GpuMat()
        self.gpu_map1.upload(map1); self.gpu_map2.upload(map2)

        self.sub = rospy.Subscriber(
            input_topic, CompressedImage, self.callback,
            queue_size=1, tcp_nodelay=True)
        self.pub = rospy.Publisher(output_topic, CompressedImage, queue_size=1)

        rospy.loginfo(f"[{input_topic}] nvJPEG + CUDA Undistorter initialized, "
                      f"cropping to ~{crop_fov_deg}° FOV")
        rospy.loginfo(f"[{input_topic}] New camera matrix:\n{self.new_K}")

    def callback(self, msg):
        self.executor.submit(self.process_image, msg)

    def crop_to_fov(self, img):
        crop_ratio = self.crop_fov_deg / self.full_hfov
        h, w = img.shape[:2]
        margin = int((1 - crop_ratio) / 2 * w)
        return img[:, margin:w - margin]

    def process_image(self, msg):
        try:
            img_cpu = self.nj.decode(msg.data)
            if img_cpu is None:
                rospy.logwarn("[nvJPEG] decode returned None")
                return

            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img_cpu)
            gpu_und = cv2.cuda.remap(
                gpu_img, self.gpu_map1, self.gpu_map2,
                interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            img_undistorted = gpu_und.download()
            img_cropped = self.crop_to_fov(img_undistorted)
            jpeg_bytes = self.nj.encode(img_cropped, 100)

            out = CompressedImage()
            out.header = msg.header
            out.format = 'jpeg'
            out.data = jpeg_bytes
            self.pub.publish(out)
        except Exception as e:
            rospy.logerr(f"Error processing frame: {e}")

def main():
    rospy.init_node("rtsp_stream_with_undistort", anonymous=True)

    factor = 1.25
    DIM = (int(640 * factor), int(480 * factor))

    # 相機配置：name, url, 原圖 topic, 校正圖 topic, K, D
    camera_configs = [
        {
            "name": "cam1",
            "url": "rtsp://admin:admin@192.168.10.101:554/video",
            "raw_topic": "/camera1/color/image_raw/compressed",
            "fix_topic": "/camera1_fix/color/image_raw/compressed",
            "K": np.array([[294.104211, 0.1005012, 317.351129],
                           [0.0, 390.626759, 244.645859],
                           [0.0, 0.0, 1.0]]) * factor,
            "D": np.array([[-0.15439177], [0.45612835], [-0.79521684], [0.46727377]])
        },
        {
            "name": "cam2",
            "url": "rtsp://admin:admin@192.168.10.102:554/video",
            "raw_topic": "/camera2/color/image_raw/compressed",
            "fix_topic": "/camera2_fix/color/image_raw/compressed",
            "K": np.array([[294.104211, 0.1005012, 317.351129],
                           [0.0, 390.626759, 244.645859],
                           [0.0, 0.0, 1.0]]) * factor,
            "D": np.array([[-0.15439177], [0.45612835], [-0.79521684], [0.46727377]])
        },
        {
            "name": "cam3",
            "url": "rtsp://admin:admin@192.168.10.103:554/video",
            "raw_topic": "/camera3/color/image_raw/compressed",
            "fix_topic": "/camera3_fix/color/image_raw/compressed",
            "K": np.array([[294.104211, 0.1005012, 317.351129],
                           [0.0, 390.626759, 244.645859],
                           [0.0, 0.0, 1.0]]) * factor,
            "D": np.array([[-0.15439177], [0.45612835], [-0.79521684], [0.46727377]])
        },
        {
            "name": "cam4",
            "url": "rtsp://admin:admin@192.168.10.104:554/video",
            "raw_topic": "/camera4/color/image_raw/compressed",
            "fix_topic": None,
            "K": None,
            "D": None
        }
    ]

    publishers = []
    undistorters = []

    for cfg in camera_configs:
        pub = RTSPCameraPublisher(cfg["name"], cfg["url"], cfg["raw_topic"])
        pub.start()
        publishers.append(pub)

        if cfg["fix_topic"] and cfg["K"] is not None and cfg["D"] is not None:
            und = CameraUndistorter(cfg["raw_topic"], cfg["fix_topic"],
                                    DIM, cfg["K"], cfg["D"],
                                    balance=1.0, crop_fov_deg=90.0, full_hfov=150.0)
            undistorters.append(und)

    rospy.loginfo("✅ 所有 RTSP 串流與去畸變模組啟動完成")
    rospy.spin()

    for pub in publishers:
        pub.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()