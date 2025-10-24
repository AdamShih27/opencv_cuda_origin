#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy, cv2, numpy as np
import subprocess, os, fcntl, time
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

class FFmpegRTMPStreamer:
    def __init__(self, rtmp_url, fps=7, resolution=None):
        self.rtmp_url = rtmp_url
        self.fps = fps
        self.resolution = resolution
        self.ffmpeg_process = None
        self.width = self.height = None
        self.last_restart_time = 0
        self.retry_interval = 5

    def start_ffmpeg(self, w, h):
        now = time.time()
        if now - self.last_restart_time < self.retry_interval:
            return
        self.last_restart_time = now
        self.stop_ffmpeg()

        rospy.loginfo(f"🚀 Start FFmpeg → {self.rtmp_url} {w}x{h}@{self.fps}fps")

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-r", str(self.fps),
            "-i", "-",
            "-c:v", "h264_nvenc",
            "-preset", "llhp",
            "-tune", "ll",
            "-g", str(self.fps), "-keyint_min", str(self.fps),
            "-b:v", "1000k", "-rc-lookahead", "0",
            "-bf", "0",
            "-probesize", "32", "-analyzeduration", "0",
            "-vf", "format=yuv420p",
            "-f", "flv", self.rtmp_url
        ]

        try:
            self.ffmpeg_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            fcntl.fcntl(self.ffmpeg_process.stderr, fcntl.F_SETFL, os.O_NONBLOCK)
            self.width, self.height = w, h
        except Exception as e:
            rospy.logerr(f"❌ FFmpeg 啟動失敗：{e}")

    def stop_ffmpeg(self):
        if self.ffmpeg_process:
            rospy.logwarn(f"🔻 關閉 FFmpeg：{self.rtmp_url}")
            try:
                self.ffmpeg_process.stdin.close()
            except: pass
            try:
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait(timeout=2)
            except: pass
            self.ffmpeg_process = None

    def push_frame(self, image):
        if self.resolution:
            image = cv2.resize(image, self.resolution)
        h, w, _ = image.shape

        if self.ffmpeg_process is None or (w != self.width or h != self.height):
            self.start_ffmpeg(w, h)

        if self.ffmpeg_process is None or self.ffmpeg_process.poll() is not None:
            self.stop_ffmpeg()
            return

        try:
            self.ffmpeg_process.stdin.write(image.tobytes())
        except Exception as e:
            rospy.logerr(f"[FFmpeg] 寫入錯誤：{e}")
            self.stop_ffmpeg()
        try:
            err = self.ffmpeg_process.stderr.read()
            if err:
                rospy.logwarn(f"[FFmpeg] {err.decode(errors='ignore').strip()}")
        except:
            pass


class MultiRTMPRelay:
    def __init__(self):
        rospy.init_node("multi_rtmp_relay", anonymous=True)
        self.bridge = CvBridge()

        # ===== 共用 IP 管理 =====
        BASE_IP = "192.168.0.180"
        PORT = 1935
        APP = "live"

        # ===== topic 與流設定 =====
        self.streams = {
            "/camera_stitched/color/image_raw/compressed": [
                {
                    "url": f"rtmp://{BASE_IP}:{PORT}/{APP}/mystream",
                    "fps": 7,
                    "resolution": (1920, 480)
                }
            ],
            "/camera4/color/image_raw/compressed": [
                {
                    "url": f"rtmp://{BASE_IP}:{PORT}/{APP}/cameraback",
                    "fps": 7,
                    "resolution": (640, 480)
                }
            ],
            "/halo_radar/radar_image/compressed": [
                {
                    "url": f"rtmp://{BASE_IP}:{PORT}/{APP}/radar",
                    "fps": 5,
                    "resolution": (480, 480)
                }
            ],
            "/detection_result_img/camera_stitched/compressed": [
                {
                    "url": f"rtmp://{BASE_IP}:{PORT}/{APP}/detr",
                    "fps": 7,
                    "resolution": (1920, 480)
                }
            ]
        }

        self.stream_objs = {}
        for topic, cfgs in self.streams.items():
            self.stream_objs[topic] = []
            for cfg in cfgs:
                streamer = FFmpegRTMPStreamer(
                    rtmp_url=cfg["url"],
                    fps=cfg.get("fps", 7),
                    resolution=cfg.get("resolution", None)
                )
                self.stream_objs[topic].append(streamer)

            rospy.Subscriber(
                topic, CompressedImage,
                self.make_callback(self.stream_objs[topic]),
                queue_size=1
            )

    def make_callback(self, streamer_list):
        def cb(msg):
            try:
                np_arr = np.frombuffer(msg.data, np.uint8)
                image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if image is not None:
                    for streamer in streamer_list:
                        streamer.push_frame(image)
            except Exception as e:
                rospy.logerr(f"[Callback] {e}")
        return cb

    def run(self):
        rospy.spin()
        for streamers in self.stream_objs.values():
            for s in streamers:
                s.stop_ffmpeg()


if __name__ == "__main__":
    try:
        node = MultiRTMPRelay()
        node.run()
    except rospy.ROSInterruptException:
        pass
