#!/usr/bin/env python3
import subprocess
import mss
import shutil
import time
import argparse

def main():
    # 參數解析
    parser = argparse.ArgumentParser(description="Screen→RTSP with aspect-ratio scaling")
    parser.add_argument("--scale", type=float, default=0.8,
                        help="相對於原始(1920)的縮放係數，例如 0.5")
    parser.add_argument("--fps", type=int, default=30,
                        help="幀率 (fps)")
    parser.add_argument("--rtsp-url", type=str,
                        default="rtsp://192.168.0.205:8554/desktop_h265",
                        help="RTSP 推流 URL")
    args = parser.parse_args()

    # 原始解析度
    orig_w, orig_h = 1920, 1080

    # 根據 scale 或直接 width 決定輸出尺寸
    if args.scale:
        out_w = int(orig_w * args.scale)
        out_h = int(orig_h * args.scale)
    else:
        out_w = args.width
        out_h = int(orig_h * (out_w / orig_w))

    print(f"→ 目標解析度：{out_w}×{out_h}，幀率：{args.fps} fps")

    # 帶寬限制
    max_bitrate_kbps = 1500
    avg_bitrate_kbps = 1000
    bufsize_kbps = 3000

    # 確認 ffmpeg
    if not shutil.which("ffmpeg"):
        print("Error: ffmpeg not found")
        return

    # ffmpeg command：stdin raw→編碼→rtsp
    cmd = [
        "ffmpeg", "-re",
        "-f", "rawvideo", "-pix_fmt", "bgra",
        "-s", f"{orig_w}x{orig_h}",
        "-r", str(args.fps),
        "-i", "-", 
        # 軟體縮放到目標解析度
        "-vf", f"scale={out_w}:{out_h}",
        "-c:v", "hevc_nvenc", 
        "-preset", "ll", 
        "-tune", "zerolatency",
        "-rc", "cbr",
        "-g", str(args.fps*2),
        # "-x265-params", "bframes=0",
        "-b:v", f"{avg_bitrate_kbps}k",
        "-maxrate", f"{max_bitrate_kbps}k",
        "-bufsize", f"{bufsize_kbps}k",
        "-pix_fmt", "yuv420p",
        "-f", "rtsp", "-rtsp_transport", "tcp",
        args.rtsp_url
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    print("▶ FFmpeg CMD:", " ".join(cmd))

    frame_interval = 1.0 / args.fps
    with mss.mss() as sct:
        monitor = {"top":0, "left":0, "width":orig_w, "height":orig_h}
        try:
            while True:
                t0 = time.time()
                img = sct.grab(monitor)
                proc.stdin.write(img.bgra)
                dt = time.time() - t0
                if dt < frame_interval:
                    time.sleep(frame_interval - dt)
        except KeyboardInterrupt:
            pass
        finally:
            proc.stdin.close()
            proc.wait()

if __name__ == "__main__":
    main()
