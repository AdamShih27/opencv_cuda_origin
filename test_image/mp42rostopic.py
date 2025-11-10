#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Header
import time
def publish_video_as_compressed(video_path, topic_name="/camera_k180/color/image_raw/compressed", rate_hz=20):
    rospy.init_node("video_publisher_compressed", anonymous=True)
    pub = rospy.Publisher(topic_name, CompressedImage, queue_size=1)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        rospy.logerr(f"Cannot open video file: {video_path}")
        return
    rate = rospy.Rate(rate_hz)
    frame_id = 0
    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            rospy.loginfo("Video file ended.")
            break
        # 編碼為JPEG格式
        _, jpeg_data = cv2.imencode('.jpg', frame)
        # 建立 CompressedImage 訊息
        msg = CompressedImage()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = f"frame_{frame_id}"
        msg.format = "jpeg"
        msg.data = jpeg_data.tobytes()
        pub.publish(msg)
        frame_id += 1
        rate.sleep()
    cap.release()
if __name__ == "__main__":
    video_file_path = "S1_ch1234_20250610_1255.mp4"  # 修改成你的mp4路徑
    publish_video_as_compressed(video_file_path)