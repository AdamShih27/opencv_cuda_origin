import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp/yolo_config"

import torch
from ultralytics import RTDETR

class RTDETRPredictor:
    def __init__(self, model_path, conf_thres=0.5, imgsz=640, device=None):
        self.conf_thres = conf_thres
        self.imgsz = imgsz
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RTDETR(model_path)

    def infer(self, image):
        """
        僅做推論，回傳 bounding boxes、類別與信心分數
        """
        results = self.model.predict(
            source=image,
            imgsz=self.imgsz,
            conf=self.conf_thres,
            device=self.device,
            verbose=False
        )[0]

        boxes, classes, scores = [], [], []

        if len(results.boxes) > 0:
            for box in results.boxes:
                cls_id = int(box.cls)
                cls_name = results.names[cls_id]
                conf = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().flatten())

                boxes.append([x1, y1, x2, y2])
                classes.append(cls_name)
                scores.append(conf)

        return boxes, classes, scores
