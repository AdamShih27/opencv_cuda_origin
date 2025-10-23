#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS Node: 訂閱 CompressedImage，執行 FastHorizon 海平線偵測、畫海平線與距離刻度、標註角度，最後再發佈 CompressedImage。
"""
import ast, rospy, cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from FastHorizonAlg_core import FastHorizon

def parse_param(name, default):
    """ 解析 ROS 參數，允許自動支援 list 字串轉換 """
    val = rospy.get_param("~" + name, default)
    if isinstance(val, str):
        try:
            val = ast.literal_eval(val.strip())
        except Exception:
            pass
    return val

class HorizonAnnotator:
    def __init__(self):
        # 讀參數（每顆鏡頭都可以 launch file 分開設）
        self.sub_topic   = parse_param("image_topic", "/camera2/color/image_raw/compressed")
        self.pub_topic   = parse_param("pub_topic",   "/camera/horizon/compressed")
        self.resize_fac  = float(parse_param("resize_factor", 1.0))
        self.jpeg_qual   = int(parse_param("jpeg_quality", 100))
        bp               = parse_param("base_point", None)
        self.base_pt     = tuple(bp) if isinstance(bp, (list, tuple)) else None

        self.alpha_intervals           = parse_param("alpha_intervals", None)
        self.alpha_distance_intervals  = parse_param("alpha_distance_intervals", None)
        self.mark_distances            = parse_param("mark_distances", None)
        self.angle_label               = parse_param("angle_label", None)
        self.angle_label_offset        = int(parse_param("angle_label_offset", 32))
        self.D_horizon_param           = parse_param("D_horizon", 5000)
        # 新增相機高度、前方距離可直接分開傳
        self.camera_height_param       = float(parse_param("camera_height", 3.0))
        self.front_distance_param      = float(parse_param("front_distance", 9.2))

        self.detector = FastHorizon(resize_factor=self.resize_fac)
        self.bridge   = CvBridge()
        self.pub      = rospy.Publisher(self.pub_topic, CompressedImage, queue_size=3)
        rospy.Subscriber(self.sub_topic, CompressedImage, self.cb_image, queue_size=1, buff_size=2**24)
        rospy.loginfo(f"[HorizonAnnotator] {self.sub_topic} → {self.pub_topic}")

        self._init_done = False

    def cb_image(self, msg: CompressedImage):
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg)
        except Exception:
            rospy.logwarn_throttle(5.0, "JPEG decode failed")
            return

        if not self._init_done:
            h, w = frame.shape[:2]
            self.detector.org_width  = w
            self.detector.org_height = h
            self.detector.res_width  = int(w * self.detector.resize_factor)
            self.detector.res_height = int(h * self.detector.resize_factor)
            self._init_done = True

        _, _, _, ok = self.detector.get_horizon(frame)
        if ok:
            self.detector.draw_hl()
            h, w = frame.shape[:2]
            base_point = self.base_pt if self.base_pt is not None else (w // 2, h - 1)

            points = self.detector.get_distance_scale_points(
                base_point                = base_point,
                D_horizon                 = self.D_horizon_param,
                camera_height             = self.camera_height_param,
                front_distance            = self.front_distance_param,
                mark_distances            = self.mark_distances,
                alpha_intervals           = self.alpha_intervals,
                alpha_distance_intervals  = self.alpha_distance_intervals,
            )
            self.detector.draw_distance_scale_points(
                points,
                color              = (0, 255, 255),
                base_point         = base_point,
                Fd                 = self.front_distance_param,
                angle_label        = self.angle_label,
                angle_label_offset = self.angle_label_offset,
            )
            annotated = self.detector.img_with_hl
        else:
            annotated = frame

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_qual]
        success, enc = cv2.imencode('.jpg', annotated, encode_param)
        if not success:
            rospy.logwarn_throttle(5.0, "JPEG encode failed")
            return

        out = CompressedImage()
        out.header = msg.header
        out.format = 'jpeg'
        out.data   = enc.tobytes()
        self.pub.publish(out)

if __name__ == "__main__":
    rospy.init_node("horizon_annotator_node")
    HorizonAnnotator()
    rospy.spin()
