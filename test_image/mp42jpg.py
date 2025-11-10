import cv2
import os
from tqdm import tqdm

def extract_frames(video_path, output_dir, step=1, resize=None):
    # 建立輸出資料夾
    os.makedirs(output_dir, exist_ok=True)

    # 開啟影片檔案
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 無法開啟影片：{video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📼 影片幀數：{total_frames}")

    frame_idx = 0
    saved_count = 0

    with tqdm(total=total_frames, desc="🔍 擷取中") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                if resize:
                    frame = cv2.resize(frame, resize)
                filename = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(filename, frame)
                saved_count += 1

            frame_idx += 1
            pbar.update(1)

    cap.release()
    print(f"✅ 完成：共儲存 {saved_count} 張圖片到 {output_dir}")

if __name__ == "__main__":
    # ✅ 手動指定參數
    video_path = "S1_ch1234_20250610_1255.mp4"
    output_dir = "video_frames"
    step = 5
    resize = None  # 或指定為 (640, 480)

    extract_frames(video_path, output_dir, step=step, resize=resize)
