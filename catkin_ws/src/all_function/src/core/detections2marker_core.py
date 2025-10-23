def convert_detections_to_pose_info(mapped_detections):
    """
    保留距離、角度、顏色與物件大小（不做座標轉換、不做角度正規化，bbox 一定存在）
    """
    # 物件實際高度（公尺）
    object_heights = {
        "Buoy": 1.0,
        "GuardBoat": 2.0,
        "RedBall": 3.0,
        "YellowBall": 3.0,
        "GreenBall": 3.0,
    }

    # 物件顏色 (R,G,B) 0~1
    class_color_map = {
        "Buoy": (1.0, 0.5, 0.0),
        "GuardBoat": (1.0, 0.0, 0.0),
        "RedBall": (1.0, 0.0, 0.0),
        "YellowBall": (1.0, 1.0, 0.0),
        "GreenBall": (0.0, 1.0, 0.0),
    }

    output = []

    for det in mapped_detections:
        cls = det.get("class")
        bbox = det["bbox"]  # [x1, y1, x2, y2]
        distance = det.get("distance", None)
        angle = det.get("angle", None)

        # 基本檢查
        if cls not in object_heights or distance is None or angle is None:
            continue

        # 預設實際高度
        real_height = object_heights[cls]

        # 根據 bbox 像素比例換算寬深
        x1, y1, x2, y2 = bbox
        pixel_width = abs(x2 - x1)
        pixel_height = abs(y2 - y1)
        if pixel_height > 0:
            scale_ratio = real_height / pixel_height
            real_width = pixel_width * scale_ratio
            width = depth = real_width
        else:
            width = depth = real_height

        # 顏色
        color = class_color_map.get(cls, (1.0, 1.0, 1.0))

        output.append({
            "class": cls,
            "distance": float(distance),
            "angle": float(angle),      # 保留原角度
            "bbox": bbox,
            "scale": (width, depth, real_height),
            "color": color
        })

    return output