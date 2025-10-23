#!/usr/bin/env python3
# File: planar_stitching_node.py

import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import rospkg
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import threading
import numpy as np
import cv2
from planar_stitching_core import Stitcher

rospack = rospkg.RosPack()

class ROSImageStitcher:
    def __init__(self):
        rospy.init_node('image_stitcher', anonymous=True)
        self.left_topic  = rospy.get_param('~sub_camera_topic_left',  '/camera3/color/image_raw/compressed')
        self.mid_topic   = rospy.get_param('~sub_camera_topic_mid',   '/camera2/color/image_raw/compressed')
        self.right_topic = rospy.get_param('~sub_camera_topic_right', '/camera1/color/image_raw/compressed')
        self.output_topic= rospy.get_param('~pub_camera_topic',       '/camera_stitched/color/image_raw/compressed')
        self.hz          = rospy.get_param('~stitch_timer_hz',        30)
        # === 圖片/矩陣儲存路徑（由 launch 傳入） ===
        self.output_image_dir = rospy.get_param('~output_image_dir', '$(find planar_processing)/src/planar_homo/result')
        self.output_homo_dir_h1 = rospy.get_param('~output_homo_dir_h1', '$(find planar_processing)/src/planar_homo/per_H1')
        self.output_homo_dir_h2 = rospy.get_param('~output_homo_dir_h2', '$(find planar_processing)/src/planar_homo/per_H2')
        self.h1_path = rospy.get_param('~h1_path', "")
        self.h2_path = rospy.get_param('~h2_path', "")

        self.output_image_dir = os.path.expandvars(self.output_image_dir)
        self.output_homo_dir_h1 = os.path.expandvars(self.output_homo_dir_h1)
        self.output_homo_dir_h2 = os.path.expandvars(self.output_homo_dir_h2)
        os.makedirs(self.output_image_dir, exist_ok=True)
        os.makedirs(self.output_homo_dir_h1, exist_ok=True)
        os.makedirs(self.output_homo_dir_h2, exist_ok=True)
        self.h1_path = os.path.expandvars(self.h1_path)
        self.h2_path = os.path.expandvars(self.h2_path)
        self.H1 = np.load(self.h1_path) if self.h1_path and os.path.exists(self.h1_path) else None
        self.H2 = np.load(self.h2_path) if self.h2_path and os.path.exists(self.h2_path) else None

        self.bridge = CvBridge()
        self.image_index = 1
        self.lock = threading.Lock() 
        self.left_queue = deque(maxlen=6)
        self.mid_queue  = deque(maxlen=6)
        self.right_queue= deque(maxlen=6)
        self.publisher = rospy.Publisher(self.output_topic, CompressedImage, queue_size=1)
        rospy.Subscriber(self.left_topic,  CompressedImage, self.left_callback)
        rospy.Subscriber(self.mid_topic,   CompressedImage, self.mid_callback)
        rospy.Subscriber(self.right_topic, CompressedImage, self.right_callback)
        self.stitcher = Stitcher()
        self.executor = ThreadPoolExecutor(max_workers=3)

    def left_callback(self, msg):
        with self.lock:
            self.left_queue.append((msg.header.stamp, msg))
    def mid_callback(self, msg):
        with self.lock:
            self.mid_queue.append((msg.header.stamp, msg))
    def right_callback(self, msg):
        with self.lock:
            self.right_queue.append((msg.header.stamp, msg))
    def find_closest_match(self, timestamp, queue):
        closest_msg = None
        min_diff = float('inf')
        for ts, msg in queue:
            diff = abs((timestamp - ts).to_sec())
            if diff < min_diff:
                closest_msg = msg
                min_diff = diff
        return closest_msg

    def stitch_images(self):
        with self.lock:
            if not self.left_queue or not self.mid_queue or not self.right_queue:
                rospy.loginfo("Waiting for all images to arrive.")
                return None
            mid_timestamp, mid_msg = self.mid_queue.popleft()
        left_msg  = self.find_closest_match(mid_timestamp, self.left_queue)
        right_msg = self.find_closest_match(mid_timestamp, self.right_queue)
        if not left_msg or not right_msg:
            rospy.loginfo("Missing matching images for stitching.")
            return None
        left_image  = self.bridge.compressed_imgmsg_to_cv2(left_msg,  desired_encoding='bgr8')
        mid_image   = self.bridge.compressed_imgmsg_to_cv2(mid_msg,   desired_encoding='bgr8')
        right_image = self.bridge.compressed_imgmsg_to_cv2(right_msg, desired_encoding='bgr8')
        img_left  = cv2.flip(mid_image, 1)
        img_right = cv2.flip(left_image, 1)
        LM_img = self.stitcher.stitching(
            img_left=img_left,
            img_right=img_right,
            flip=True,
            H=self.H1,
            save_H_path=None if self.H1 is not None else os.path.join(self.output_homo_dir_h1, f"H1_{self.image_index}.npy")
        )
        if LM_img is None:
            rospy.loginfo(f"Skipping stitching for image set {self.image_index} (H1 invalid).")
            return None
        final_image = self.stitcher.stitching(
            img_left=LM_img,
            img_right=right_image,
            flip=False,
            H=self.H2,
            save_H_path=None if self.H2 is not None else os.path.join(self.output_homo_dir_h2, f"H2_{self.image_index}.npy")
        )
        if final_image is None:
            rospy.loginfo(f"Skipping final stitching for image set {self.image_index} (H2 invalid).")
            return None
        if self.H1 is None or self.H2 is None:
            outpath = os.path.join(self.output_image_dir, f"pano_{self.image_index:06d}.png")
            cv2.imwrite(outpath, final_image)
            rospy.loginfo(f"[Stitcher] Saved {outpath}")
        return final_image

    def process_images_task(self):
        stitched_image = self.stitch_images()
        if stitched_image is not None:
            compressed_msg = self.bridge.cv2_to_compressed_imgmsg(stitched_image, dst_format='jpeg')
            self.publisher.publish(compressed_msg)
            rospy.loginfo(f"Published stitched image {self.image_index}.")
            self.image_index += 1

    def run(self):
        rospy.Timer(rospy.Duration(1.0 / self.hz), self.timer_callback)
        rospy.loginfo(f"Main thread: start spin() for callbacks (stitching at {self.hz} Hz).")
        rospy.spin()
        rospy.loginfo("Main thread: spin() ended, shutting down executor.")
        self.executor.shutdown(wait=True)

    def timer_callback(self, event):
        self.executor.submit(self.process_images_task)

if __name__ == '__main__':
    try:
        stitcher = ROSImageStitcher()
        stitcher.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("GPU-based Image stitcher node terminated.")