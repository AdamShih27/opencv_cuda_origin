import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp/yolo_config"  # 可留可刪

import torch
from ultralytics import YOLO

class YOLOPredictor:
    def __init__(self, model_path, conf_thres=0.5, imgsz=640, device=None):
        self.conf_thres = conf_thres
        self.imgsz = imgsz
        # Ultralytics 的 device 參數可用 0 / 'cpu'
        self.device = 0 if (device is None and torch.cuda.is_available()) else (device or 'cpu')
        self.model = YOLO(model_path)  # 載入 YOLO 權重（.pt）

    def infer(self, image):
        """
        僅做推論，回傳 bounding boxes、類別與信心分數
        boxes: List[[x1,y1,x2,y2]]  (int)
        classes: List[str]          (類別名稱)
        scores: List[float]
        """
        results = self.model.predict(
            source=image,
            imgsz=self.imgsz,
            conf=self.conf_thres,
            device=self.device,
            verbose=False
        )[0]

        boxes, classes, scores = [], [], []
        names = getattr(results, "names", None)
        if names is None:
            # 後備：從模型讀
            names = getattr(self.model.model, "names", {})
        # 確保 key 是 int
        if isinstance(names, dict):
            names = {int(k): v for k, v in names.items()}

        for b in results.boxes:
            # xyxy
            xyxy = b.xyxy[0].detach().cpu().numpy().astype(int).tolist()
            # conf
            conf = float(b.conf[0].item() if hasattr(b.conf[0], "item") else b.conf[0])
            # class id -> name
            cls_idx = int(b.cls[0].item() if hasattr(b.cls[0], "item") else b.cls[0])
            cls_name = names.get(cls_idx, f"cls_{cls_idx}")

            boxes.append(xyxy)
            classes.append(cls_name)
            scores.append(conf)

        return boxes, classes, scores
