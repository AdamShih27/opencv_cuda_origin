#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS CompressedImage → 同時推 RTSP (H.264 / H.265)
"""
import rospy, cv2, numpy as np
import subprocess, fcntl, os, time
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

# ------------------------------------------------------------
#  FFmpeg 推流器
# ------------------------------------------------------------
class FFmpegStreamer:
    def __init__(self, rtsp_url, fixed_fps=6, resolution=None, display=False, codec="h264", bitrate=None, maxrate=None):
        self.rtsp_url = rtsp_url
        self.fixed_fps = fixed_fps
        self.resolution = resolution
        self.display = display
        self.codec = codec.lower()
        self.bitrate = bitrate
        self.maxrate = maxrate
        self.ffmpeg_process = None
        self.player_process = None
        self.width = self.height = None
        self.last_restart_time = 0
        self.retry_interval = 5
        self._frame_cnt = 0
        self._fps_t0 = time.time()

    def _spawn_player(self):
        if self.display and self.player_process is None:
            self.player_process = subprocess.Popen(
                ['ffplay', '-fflags', 'nobuffer', '-loglevel', 'error',
                 '-rtsp_transport', 'tcp', self.rtsp_url])

    def _kill_player(self):
        if self.player_process:
            try:
                self.player_process.terminate()
                self.player_process.wait(timeout=2)
            except Exception:
                pass
            self.player_process = None

    def stop_ffmpeg(self):
        if self.ffmpeg_process:
            rospy.logwarn(f"🔻 關閉 FFmpeg：{self.rtsp_url}")
            try:
                self.ffmpeg_process.stdin.close()
            except Exception:
                pass
            try:
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait(timeout=2)
            except Exception:
                pass
            self.ffmpeg_process = None
        self._kill_player()
    
    def _build_codec_args(self):
        codec_args = []
        if self.codec in ("h265_nvenc", "hevc_nvenc"):
            codec_args = [
                "-c:v", "hevc_nvenc",
                "-preset", "ll",
                "-tune", "zerolatency",
                "-rc", "cbr",
                "-g", str(self.fixed_fps),
                "-forced-idr", "1",
                "-bf", "0",
                "-spatial_aq", "1",
                "-aq-strength", "8"
            ]
        elif self.codec == "h264_nvenc":
            codec_args = [
                "-c:v", "h264_nvenc",
                "-preset", "ll",
                "-tune", "zerolatency",
                "-rc", "cbr",
                "-g", str(self.fixed_fps),
                "-forced-idr", "1",
                "-bf", "0",
                "-spatial_aq", "1",
                "-aq-strength", "8"
            ]
        elif self.codec in ("libx265", "x265", "h265"):
            codec_args = [
                "-c:v", "libx265",
                "-preset", "fast",
                "-x265-params", f"repeat-headers=1:keyint={self.fixed_fps}:min-keyint={self.fixed_fps}:no-scenecut=1:bf=0"
            ]
        else:
            codec_args = [
                "-c:v", "libx264",
                "-profile:v", "baseline",
                "-level", "3.1",
                "-preset", "veryfast",
                "-tune", "zerolatency",
                "-x264-params", f"keyint={self.fixed_fps}:min-keyint={self.fixed_fps}:scenecut=0:bframes=0"
            ]

        # 添加比特率設定
        if hasattr(self, 'bitrate') and self.bitrate:
            codec_args += ["-b:v", self.bitrate]
        if hasattr(self, 'maxrate') and self.maxrate:
            codec_args += ["-maxrate", self.maxrate, "-bufsize", self.maxrate]

        return codec_args


    def start_ffmpeg(self, w, h):
        now = time.time()
        if now - self.last_restart_time < self.retry_interval:
            return
        self.last_restart_time = now
        self.stop_ffmpeg()

        rospy.loginfo(f"🚀 FFmpeg→{self.rtsp_url}  {w}x{h}@{self.fixed_fps}fps  {self.codec}")
        cmd = [
            "ffmpeg",
            "-f", "rawvideo",                  # 協定：原始未壓縮影格
            "-pix_fmt", "bgr24",               # 像素格式：OpenCV 默認 BGR24 :contentReference[oaicite:10]{index=10}
            "-s", f"{w}x{h}",                  # 解析度：寬×高
            "-r", str(self.fixed_fps),         # 輸入影格率
            "-i", "-",                         # 來源：stdin pipe
            *self._build_codec_args(),         # 前述編碼器參數
            "-pix_fmt", "yuv420p",             # 輸出像素格式，YUV420p 廣泛兼容 :contentReference[oaicite:11]{index=11}
            "-g", str(self.fixed_fps),         # GOP size，同一秒一 IDR :contentReference[oaicite:12]{index=12}
            "-keyint_min", str(self.fixed_fps),# 最小 GOP 間隔
            "-bf", "0",                        # 禁用 B-frame，降低延遲 :contentReference[oaicite:14]{index=14}
            "-f", "rtsp",                      # 輸出格式：RTSP
            "-rtsp_transport", "tcp",          # 使用 TCP 方式傳輸 RTSP (更穩定) :contentReference[oaicite:15]{index=15}
            self.rtsp_url
        ]

        try:
            self.ffmpeg_process = subprocess.Popen(
                cmd, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
            fcntl.fcntl(self.ffmpeg_process.stderr, fcntl.F_SETFL, os.O_NONBLOCK)
            self.width, self.height = w, h
            self._spawn_player()
        except Exception as e:
            rospy.logerr(f"❌ FFmpeg 啟動失敗：{e}")

    def push_frame(self, cv_image):
        if self.resolution:
            cv_image = cv2.resize(cv_image, self.resolution)
        h, w, _ = cv_image.shape

        if self.ffmpeg_process is None or (w != self.width or h != self.height):
            self.start_ffmpeg(w, h)

        if self.ffmpeg_process is None or self.ffmpeg_process.poll() is not None:
            self.stop_ffmpeg()
            return

        try:
            self.ffmpeg_process.stdin.write(cv_image.tobytes())
            self._frame_cnt += 1
            now = time.time()
            if now - self._fps_t0 >= 1.0:
                fps = self._frame_cnt / (now - self._fps_t0)
                rospy.loginfo(f"[FPS] {self.rtsp_url} → {fps:.1f} fps")
                self._frame_cnt = 0
                self._fps_t0 = now
        except BrokenPipeError:
            self.stop_ffmpeg()
        except Exception as e:
            rospy.logwarn(f"📛 FFmpeg 寫入錯誤：{e}")
            self.stop_ffmpeg()

        try:
            err = self.ffmpeg_process.stderr.read()
            if err:
                dec = err.decode(errors="ignore")
                if any(t in dec for t in ("Connection refused", "not found")):
                    rospy.logwarn(f"[FFmpeg] {dec.strip()}")
                    self.stop_ffmpeg()
        except Exception:
            pass


# ------------------------------------------------------------
#  ROS Multi‑RTSP Relay
# ------------------------------------------------------------
class AVCMultiRtspRelay:
    def __init__(self):
        rospy.init_node("avc_multi_rtsp_relay", anonymous=True)
        self.bridge = CvBridge()

        BASE_IP = "192.168.10.136"
        # BASE_IP = "172.20.10.2"
        PORT = 8554
        self.streams = {
            "/camera_pano_masked/image_raw/compressed": [
                {"url": f"rtsp://{BASE_IP}:{PORT}/mystream_h265", "fps": 20, "resolution": (750, 375), 
                 "codec": "h265_nvenc", "display": True, "bitrate": "750k", "maxrate": "1000k"}
            ],

            # "/detr/compressed": [
            #     {"url": f"rtsp://{BASE_IP}:{PORT}/mystream_h265", "fps": 20, "resolution": (700, 240), 
            #      "codec": "h265_nvenc", "display": False, "bitrate": "750k", "maxrate": "1000k"}
            # ],

            # "/camera_stitched/color/image_raw/compressed": [
            #     {"url": f"rtsp://{BASE_IP}:{PORT}/mystream_h264", "fps": 60, "resolution": (1920, 480), 
            #      "codec": "h264_nvenc", "display": False, "bitrate": "2000k", "maxrate": "2500k"},
            #     {"url": f"rtsp://{BASE_IP}:{PORT}/mystream_h265", "fps": 20, "resolution": (1920, 480), 
            #      "codec": "h265_nvenc", "display": False, "bitrate": "1125k", "maxrate": "1500k"}
            # ],
            "/camera4/color/image_raw/compressed": [
            #     {"url": f"rtsp://{BASE_IP}:{PORT}/cameraback_h264", "fps": 60, "resolution": (640, 480), 
            #      "codec": "h264_nvenc", "display": False, "bitrate": "2000k", "maxrate": "2500k"},
                {"url": f"rtsp://{BASE_IP}:{PORT}/cameraback_h265", "fps": 20, "resolution": (640, 480), 
                 "codec": "h265_nvenc", "display": False, "bitrate": "375k", "maxrate": "500k"}
            ]
            # "/halo_radar/radar_image/compressed": [ 
                # {"url": f"rtsp://{BASE_IP}:{PORT}/radar_h264", "fps": 5, "resolution": (480, 480), 
                #  "codec": "h264_nvenc", "display": False, "bitrate": "2000k", "maxrate": "2500k"},
            #     {"url": f"rtsp://{BASE_IP}:{PORT}/radar_h265", "fps": 5, "resolution": (480, 480), 
            #      "codec": "h265_nvenc", "display": False, "bitrate": "375k", "maxrate": "500k"}
            # ],
            # "/detection_result_img/camera_stitched/compressed": [
            #     {"url": f"rtsp://{BASE_IP}:{PORT}/detr_h264", "fps": 60, "resolution": (1920, 480), 
            #      "codec": "h264_nvenc", "display": False, "bitrate": "2000k", "maxrate": "2500k"},
            #     {"url": f"rtsp://{BASE_IP}:{PORT}/detr_h265", "fps": 60, "resolution": (1920, 480), 
            #      "codec": "h265_nvenc", "display": False, "bitrate": "1500k", "maxrate": "2000k"}
            # ]
        }

        self.stream_objs = {}
        for topic, cfg_list in self.streams.items():
            self.stream_objs[topic] = []
            for cfg in cfg_list:
                st = FFmpegStreamer(
                    cfg["url"],
                    cfg["fps"],
                    cfg["resolution"],
                    cfg["display"],
                    codec=cfg["codec"],
                    bitrate=cfg.get("bitrate"),
                    maxrate=cfg.get("maxrate"))
                self.stream_objs[topic].append(st)

            rospy.Subscriber(topic, CompressedImage, self.make_callback(self.stream_objs[topic]), queue_size=1)

    def make_callback(self, streamer_list):
        def cb(msg):
            try:
                cv_img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
                if cv_img is not None:
                    for st in streamer_list:
                        st.push_frame(cv_img)
            except Exception as e:
                rospy.logerr(f"[Callback] {e}")
        return cb

    def run(self):
        rospy.spin()
        for lst in self.stream_objs.values():
            for st in lst:
                st.stop_ffmpeg()


# ------------------------------------------------------------
if __name__ == "__main__":
    try:
        node = AVCMultiRtspRelay()
        node.run()
    except rospy.ROSInterruptException:
        pass