
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from threading import Thread
from nvjpeg import NvJpeg

class RTSPCameraHandler:
    def __init__(self, name, rtsp_url, topic_raw, topic_fixed=None, dim=None, K=None, D=None, balance=0.9):
        self.name = name
        self.rtsp_url = rtsp_url
        self.topic_raw = topic_raw
        self.topic_fixed = topic_fixed
        self.publish_fixed = topic_fixed is not None and K is not None and D is not None

        self.pub_raw = rospy.Publisher(topic_raw, CompressedImage, queue_size=1)
        self.pub_fixed = rospy.Publisher(topic_fixed, CompressedImage, queue_size=1) if self.publish_fixed else None
        self.nvjpeg = NvJpeg()
        self.stream = cv2.cuda_Stream()  # ✅ 每路建立專屬 CUDA Stream
        self.dim = dim
        self.K = K
        self.D = D

        if self.publish_fixed:
            self.new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                self.K, self.D, self.dim, np.eye(3), balance=balance)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                self.K, self.D, np.eye(3), self.new_K, self.dim, cv2.CV_32FC1)
            self.gpu_map1 = cv2.cuda_GpuMat(); self.gpu_map2 = cv2.cuda_GpuMat()
            self.gpu_map1.upload(map1); self.gpu_map2.upload(map2)

        # H.264 解碼 pipeline
        # self.pipeline = (
        #     f"rtspsrc location={rtsp_url} latency=0 protocols=udp drop-on-latency=true ! "
        #     "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! "
        #     "videoconvert n-threads=2 ! video/x-raw,format=BGR ! "
        #     "appsink drop=true max-buffers=1 sync=false"
        # )

        # H.265 解碼 pipeline
        self.pipeline = (
            f"rtspsrc location={rtsp_url} latency=0 protocols=udp drop-on-latency=true ! "
            "rtph265depay ! h265parse ! avdec_h265 ! videoconvert ! "
            "videoconvert n-threads=2 ! video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )
        
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
            ret, frame = self.cap.read()
            if not ret or frame is None:
                rospy.logwarn(f"[{self.name}] ⚠️ 無法讀取影像")
                continue

            try:
                # 發布原圖
                jpeg_raw = self.nvjpeg.encode(frame, 100)
                msg_raw = CompressedImage()
                msg_raw.header.stamp = rospy.Time.now()
                msg_raw.format = "jpeg"
                msg_raw.data = jpeg_raw
                self.pub_raw.publish(msg_raw)

                # 發布校正圖
                if self.publish_fixed:
                    gpu_img = cv2.cuda_GpuMat(); gpu_img.upload(frame,stream=self.stream)
                    gpu_undist = cv2.cuda.remap(gpu_img, self.gpu_map1, self.gpu_map2,
                                                interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,stream=self.stream)
                    img_out = gpu_undist.download(stream=self.stream)
                    self.stream.waitForCompletion() 

                    jpeg_fixed = self.nvjpeg.encode(img_out, 100)
                    msg_fixed = CompressedImage()
                    msg_fixed.header.stamp = rospy.Time.now()
                    msg_fixed.format = "jpeg"
                    msg_fixed.data = jpeg_fixed
                    self.pub_fixed.publish(msg_fixed)

            except Exception as e:
                rospy.logerr(f"[{self.name}] ❌ 發布影像失敗: {e}")
                continue

def main():
    rospy.init_node("rtsp_dual_output_node", anonymous=True)
    factor = 4.05
    DIM = (int(640 * factor), int(480 * factor))

    camera_configs = [
        {
            "name": "cam1",
            "rtsp_url": "rtsp://admin:admin@192.168.1.108:554/video",
            "topic_raw": "/camera1/color/image_raw/compressed",
            "topic_fixed": "/camera1_fix/color/image_raw/compressed",
            "K": np.array([[268.06554187*factor, 1.51263281*factor, 320*factor],
                           [0., 356.57309093*factor, 231.71146684*factor],
                           [0., 0., 1.]]),
            "D": np.array([[0.05184647], [0.01756823], [-0.02638422], [0.00762106]])
        }
    ]

    nodes = []
    for cfg in camera_configs:
        node = RTSPCameraHandler(
            name=cfg["name"],
            rtsp_url=cfg["rtsp_url"],
            topic_raw=cfg["topic_raw"],
            topic_fixed=cfg["topic_fixed"],
            dim=DIM,
            K=cfg["K"],
            D=cfg["D"]
        )
        node.start()
        nodes.append(node)

    rospy.loginfo("✅ 所有相機雙輸出模組（原圖 + 去畸變）已啟動")
    rospy.spin()

    for node in nodes:
        node.stop()

if __name__ == "__main__":
    main()
