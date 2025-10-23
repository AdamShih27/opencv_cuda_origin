#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import time
from sensor_msgs.msg import CompressedImage
from threading import Thread
from nvjpeg import NvJpeg  # ✅ 引入 NVIDIA JPEG 編碼器


class RTSPCameraPublisher:
    def __init__(self, name, rtsp_url, topic_name):
        self.name = name
        self.rtsp_url = rtsp_url
        self.topic_name = topic_name
        self.publisher = rospy.Publisher(topic_name, CompressedImage, queue_size=1)
        
        self.nvjpeg = NvJpeg()  # ✅ 初始化 GPU JPEG 編碼器
        
        # H.265 解碼 pipeline
        self.pipeline = (
            f"rtspsrc location={rtsp_url} latency=0 protocols=udp drop-on-latency=true ! "
            "rtph265depay ! h265parse ! avdec_h265 ! videoconvert ! "
            "videoconvert n-threads=2 ! video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )

        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            rospy.logerr(f"[{self.name}] ❌ 無法開啟 RTSP H.265 串流：{rtsp_url}")
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
                msg = CompressedImage()
                msg.header.stamp = rospy.Time.now()
                msg.format = "jpeg"

                # ✅ 使用 nvJPEG 進行 GPU 編碼
                try:
                    jpeg_bytes = self.nvjpeg.encode(frame, 100)
                    msg.data = jpeg_bytes
                    self.publisher.publish(msg)
                except Exception as e:
                    rospy.logwarn(f"[{self.name}] ❌ nvJPEG 編碼失敗: {e}")
                    time.sleep(0.05)

            except Exception as e:
                rospy.logerr(f"[{self.name}] 發布過程出錯：{e}")
                time.sleep(0.1)

def main():
    rospy.init_node("multi_rtsp_to_compressed_node", anonymous=True)

    cameras = [
        {"name": "cam1", "url": "rtsp://192.168.10.92:8554/s2_1234", "topic": "/camera/color/image_raw/compressed"},
    ]

    nodes = []
    for cam in cameras:
        node = RTSPCameraPublisher(cam["name"], cam["url"], cam["topic"])
        node.start()
        nodes.append(node)

    rospy.loginfo("✅ 所有 RTSP H.265 相機串流節點啟動完成。")
    rospy.spin()

    for node in nodes:
        node.stop()

if __name__ == "__main__":
    main()
