#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import torch
from ultralytics import RTDETR
import os

class RTDETRVisualizerNode:
    def __init__(self):
        # ROS node 初始化
        rospy.init_node("rtdetr_draw_node", anonymous=True)

        # 參數設定
        self.model_path = rospy.get_param("~model_path", "models/rtdetr-l.pt")
        self.conf_thres = rospy.get_param("~conf", 0.5)
        self.imgsz = rospy.get_param("~imgsz", 640)
        self.input_topic = rospy.get_param("~input_topic", "/camera_pano_stitched/image_raw/compressed")
        self.output_topic = rospy.get_param("~output_topic", "/rtdetr/output/image/compressed")

        # 載入模型
        if not os.path.exists(self.model_path):
            rospy.logerr(f"❌ Model not found: {self.model_path}")
            exit(1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RTDETR(self.model_path)
        self.model.to(self.device)
        rospy.loginfo("✅ RT-DETR model loaded.")

        # ROS 訂閱與發布
        rospy.Subscriber(self.input_topic, CompressedImage, self.image_callback, queue_size=1, buff_size=2**24)
        self.pub_img = rospy.Publisher(self.output_topic, CompressedImage, queue_size=1)

    def image_callback(self, msg):
        # 解碼 CompressedImage -> OpenCV
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # 推論
        results = self.model.predict(
            source=img,
            imgsz=self.imgsz,
            conf=self.conf_thres,
            device=self.device,
            verbose=False
        )

        r = results[0]

        # 如果無偵測結果，顯示提示文字
        if len(r.boxes) == 0:
            cv2.putText(img, "No Detection", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            for box in r.boxes:
                cls_id = int(box.cls)
                cls_name = r.names[cls_id]
                conf = float(box.conf)
                xyxy = box.xyxy.cpu().numpy().astype(int).tolist()[0]
                x1, y1, x2, y2 = xyxy

                # 繪製 BBox 與類別
                label = f"{cls_name} {conf:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 轉回 CompressedImage 並發布
        _, img_encoded = cv2.imencode('.jpg', img)
        out_msg = CompressedImage()
        out_msg.header = msg.header
        out_msg.format = "jpeg"
        out_msg.data = np.array(img_encoded).tobytes()

        self.pub_img.publish(out_msg)

if __name__ == "__main__":
    try:
        node = RTDETRVisualizerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
