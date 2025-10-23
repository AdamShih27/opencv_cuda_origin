#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import ast
import time
import threading
from nvjpeg import NvJpeg

def robust_to_array(arr):
    if isinstance(arr, str):
        arr = ast.literal_eval(arr)
    return np.asarray(arr, dtype=np.float64)

def scale_intrinsic(K, factor):
    K = np.array(K, dtype=np.float64).copy()
    K[0, 0] *= factor  # fx
    K[1, 1] *= factor  # fy
    K[0, 1] *= factor  # skew
    K[0, 2] *= factor  # cx
    K[1, 2] *= factor  # cy
    return K

class CameraUndistorter:
    def __init__(self, input_topic, output_topic, dim, K, D,
                 balance=1.0, crop_fov_deg=110.0, full_hfov=150.0):
        self.dim = dim
        self.K = robust_to_array(K)
        self.D = robust_to_array(D)
        self.crop_fov_deg = crop_fov_deg
        self.full_hfov = full_hfov
        self.bridge = CvBridge()
        self.nj = NvJpeg()
        self.stream = cv2.cuda_Stream()

        self.new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.K, self.D, self.dim, np.eye(3), balance=balance)

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, np.eye(3), self.new_K, self.dim, cv2.CV_32FC1)
        self.gpu_map1 = cv2.cuda_GpuMat(); self.gpu_map2 = cv2.cuda_GpuMat()
        self.gpu_map1.upload(map1); self.gpu_map2.upload(map2)

        self.latest_msg = None
        self.lock = threading.Lock()
        self.worker_thread = threading.Thread(target=self.worker_loop)
        self.worker_thread.daemon = True
        self.worker_thread.start()

        self.sub = rospy.Subscriber(
            input_topic, CompressedImage, self.callback,
            queue_size=1, tcp_nodelay=True)
        self.pub = rospy.Publisher(output_topic, CompressedImage, queue_size=1)

        rospy.loginfo(f"[{input_topic}] nvJPEG + CUDA Undistorter initialized (latest only), cropping to ~{crop_fov_deg}° FOV")
        rospy.loginfo(f"[{input_topic}] New camera matrix:\n{self.new_K}")

    def callback(self, msg):
        with self.lock:
            self.latest_msg = msg  # always keep only the latest message

    def worker_loop(self):
        while not rospy.is_shutdown():
            msg = None
            with self.lock:
                if self.latest_msg is not None:
                    msg = self.latest_msg
                    self.latest_msg = None
            if msg is not None:
                self.process_image(msg)
            else:
                time.sleep(0.001)  # avoid busy wait

    def crop_to_fov(self, img):
        crop_ratio = self.crop_fov_deg / self.full_hfov
        h, w = img.shape[:2]
        margin = int((1 - crop_ratio) / 2 * w)
        return img[:, margin:w - margin] if margin > 0 else img

    def process_image(self, msg):
        t0 = time.perf_counter()
        try:
            img_cpu = self.nj.decode(msg.data)
            t1 = time.perf_counter()
            if img_cpu is None:
                rospy.logwarn("[nvJPEG] decode returned None")
                return

            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img_cpu, stream=self.stream)
            t2 = time.perf_counter()

            gpu_und = cv2.cuda.remap(
                gpu_img, self.gpu_map1, self.gpu_map2,
                interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, stream=self.stream)
            t3 = time.perf_counter()

            img_undistorted = gpu_und.download(stream=self.stream)
            self.stream.waitForCompletion()
            t4 = time.perf_counter()

            img_cropped = self.crop_to_fov(img_undistorted)
            jpeg_bytes = self.nj.encode(img_cropped, 100)
            t5 = time.perf_counter()

            out = CompressedImage()
            out.header = msg.header
            out.format = 'jpeg'
            out.data = jpeg_bytes
            self.pub.publish(out)
            t6 = time.perf_counter()

            # print(
            #     f"[UndistortTimer] "
            #     f"decode={1000*(t1-t0):.1f}ms | "
            #     f"upload={1000*(t2-t1):.1f}ms | "
            #     f"remap={1000*(t3-t2):.1f}ms | "
            #     f"download={1000*(t4-t3):.1f}ms | "
            #     f"encode+crop={1000*(t5-t4):.1f}ms | "
            #     f"publish={1000*(t6-t5):.1f}ms | "
            #     f"total={1000*(t6-t0):.1f}ms"
            # )

        except Exception as e:
            rospy.logerr(f"Error processing frame ({self.K[0,2]:.1f}): {e}")

if __name__ == '__main__':
    rospy.init_node('fisheye_undistorter', anonymous=True)

    input_topic = rospy.get_param("~input_topic")
    output_topic = rospy.get_param("~output_topic")
    factor = rospy.get_param("~factor", 1.0)
    balance = rospy.get_param("~balance", 1.0)
    crop_fov_deg = rospy.get_param("~crop_fov_deg", 90.0)
    full_hfov = rospy.get_param("~full_hfov", 150.0)

    K = rospy.get_param("~K")
    D = rospy.get_param("~D")

    DIM = (int(640 * factor), int(480 * factor))
    K_scaled = scale_intrinsic(robust_to_array(K), factor)

    CameraUndistorter(
        input_topic, output_topic, DIM, K_scaled, D,
        balance=balance, crop_fov_deg=crop_fov_deg, full_hfov=full_hfov
    )

    rospy.loginfo("✅ 單一魚眼相機去畸變已啟動")
    rospy.spin()

# #test for raw 
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import cv2, numpy as np, rospy, time, threading, ast
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge

# def robust_to_array(arr):
#     if isinstance(arr, str): arr = ast.literal_eval(arr)
#     return np.asarray(arr, dtype=np.float64)

# def scale_intrinsic(K, factor):
#     K = np.array(K, dtype=np.float64).copy()
#     K[0,0] *= factor; K[1,1] *= factor
#     K[0,1] *= factor; K[0,2] *= factor; K[1,2] *= factor
#     return K

# class CameraUndistorterRaw:
#     def __init__(self, input_topic, output_topic, dim, K, D,
#                  balance=1.0, crop_fov=110.0, full_hfov=150.0):
#         self.dim = dim
#         self.crop_fov = crop_fov
#         self.full_hfov = full_hfov
#         self.K = robust_to_array(K)
#         self.D = robust_to_array(D)
#         self.bridge = CvBridge()
#         self.stream = cv2.cuda_Stream()

#         self.new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
#             self.K, self.D, self.dim, np.eye(3), balance=balance
#         )
#         map1, map2 = cv2.fisheye.initUndistortRectifyMap(
#             self.K, self.D, np.eye(3), self.new_K, self.dim, cv2.CV_32FC1
#         )
#         self.gpu_map1 = cv2.cuda_GpuMat(); self.gpu_map2 = cv2.cuda_GpuMat()
#         self.gpu_map1.upload(map1); self.gpu_map2.upload(map2)

#         self.latest_msg = None
#         self.lock = threading.Lock()
#         threading.Thread(target=self.worker_loop, daemon=True).start()

#         self.sub = rospy.Subscriber(
#             input_topic, Image, self.callback,
#             queue_size=1, tcp_nodelay=True
#         )
#         self.pub = rospy.Publisher(output_topic, Image, queue_size=1)

#         rospy.loginfo(f"[{input_topic}] RAW Undistorter initialized -> {output_topic}")

#     def callback(self, msg):
#         with self.lock:
#             self.latest_msg = msg

#     def worker_loop(self):
#         while not rospy.is_shutdown():
#             with self.lock:
#                 msg = self.latest_msg
#                 self.latest_msg = None
#             if msg:
#                 self.process_image(msg)
#             else:
#                 time.sleep(0.001)

#     def crop_to_fov(self, img):
#         h, w = img.shape[:2]
#         margin = int((1 - self.crop_fov / self.full_hfov) / 2 * w)
#         return img[:, margin:w-margin] if margin > 0 else img

#     def process_image(self, msg):
#         t0 = time.perf_counter()
#         try:
#             # Step 1: 将 ROS Image 转为 NumPy
#             img_cpu = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
#             t1 = time.perf_counter()

#             # Step 2: 上传至 GPU
#             gpu = cv2.cuda_GpuMat()
#             gpu.upload(img_cpu, stream=self.stream)
#             t2 = time.perf_counter()

#             # Step 3: GPU fisheye remap 去畸变
#             und = cv2.cuda.remap(
#                 gpu, self.gpu_map1, self.gpu_map2,
#                 interpolation=cv2.INTER_LINEAR,
#                 borderMode=cv2.BORDER_CONSTANT,
#                 stream=self.stream
#             )
#             img2 = und.download(stream=self.stream)
#             self.stream.waitForCompletion()
#             t3 = time.perf_counter()

#             # Step 4: 裁切 FOV
#             cropped = self.crop_to_fov(img2)
#             t4 = time.perf_counter()

#             # Step 5: 转为 ROS Image 并发布
#             out = self.bridge.cv2_to_imgmsg(cropped, 'bgr8')
#             out.header = msg.header
#             self.pub.publish(out)
#             t5 = time.perf_counter()

#             # 打印各阶段耗时
#             print(
#                 f"[Timing] "
#                 f"decode/bridge={1000*(t1-t0):.1f}ms | "
#                 f"upload={1000*(t2-t1):.1f}ms | "
#                 f"remap+download={1000*(t3-t2):.1f}ms | "
#                 f"crop={1000*(t4-t3):.1f}ms | "
#                 f"publish={1000*(t5-t4):.1f}ms | "
#                 f"total={1000*(t5-t0):.1f}ms"
#             )
#         except Exception as e:
#             rospy.logerr(f"RAW undistort error: {e}")


# if __name__ == "__main__":
#     rospy.init_node('fisheye_undistorter_raw_latest', anonymous=True)

#     it = rospy.get_param('~input_topic')
#     ot = rospy.get_param('~output_topic')
#     factor = rospy.get_param('~factor', 1.0)
#     balance = rospy.get_param('~balance', 1.0)
#     crop = rospy.get_param('~crop_fov_deg', 90.0)
#     full_hfov = rospy.get_param('~full_hfov', 150.0)
#     K = rospy.get_param('~K'); D = rospy.get_param('~D')

#     DIM = (int(640*factor), int(480*factor))
#     K_scaled = scale_intrinsic(robust_to_array(K), factor)

#     CameraUndistorterRaw(
#         it, ot, DIM, K_scaled, D,
#         balance=balance,
#         crop_fov=crop,
#         full_hfov=full_hfov
#     )

#     rospy.spin()