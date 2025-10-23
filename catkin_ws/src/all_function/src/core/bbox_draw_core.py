import cv2

def draw_bboxes_from_mapped(image, mapped_detections, color=(0, 255, 0),
                            thickness=2, font_scale=0.5, font_color=(0, 0, 255),
                            font=cv2.FONT_HERSHEY_SIMPLEX):
    """
    畫出 bbox 並標註類別與信心值。
    所有偵測到的物件、距離與角度會列在左上角。
    """
    img = image.copy()

    if not mapped_detections:
        cv2.putText(img, "no detection", (10, 20), font, font_scale, font_color,
                    thickness=1, lineType=cv2.LINE_AA)
        return img

    # # ➤ 左上角清單：類別 + 距離 + 角度
    # corner_text_lines = []
    # for det in mapped_detections:
    #     cls_name = det.get("class", "unknown")
    #     distance = det.get("distance", None)
    #     angle = det.get("angle", None)

    #     dist_str = f"{distance:.1f}m" if distance is not None else "N/A"
    #     angle_str = f"{angle:.1f}deg" if angle is not None else "N/A"

    #     corner_text_lines.append(f"{cls_name}: {dist_str} / {angle_str}")

    # # ➤ 顯示左上角物件距離/角度列表
    # for i, line in enumerate(corner_text_lines):
    #     y = 20 + i * 20  # 每行間隔 20 pixels
    #     cv2.putText(img, line, (10, y), font, font_scale, font_color,
    #                 thickness=1, lineType=cv2.LINE_AA)

    # ➤ 畫出 BBOX + 類別 + score（不含距離/角度）
    for det in mapped_detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cls_name = det.get("class", "unknown")
        score = det.get("score", 0.0)

        # 畫框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # 框上文字：類別 + 分數
        text = f"{cls_name} {score:.2f}"
        cv2.putText(img, text, (x1, y1 - 5), font, font_scale, font_color,
                    thickness=1, lineType=cv2.LINE_AA)

    return img
