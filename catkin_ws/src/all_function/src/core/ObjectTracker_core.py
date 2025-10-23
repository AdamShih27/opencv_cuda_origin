# -*- coding: utf-8 -*-
import math
from scipy.optimize import linear_sum_assignment
import numpy as np
import cv2

def unwrap_deg(prev_deg, new_deg):
    """將 new_deg unwrap 到最接近 prev_deg 的等價角度（以 360° 週期）。"""
    diff = new_deg - prev_deg
    diff_wrapped = (diff + 180.0) % 360.0 - 180.0
    return prev_deg + diff_wrapped

class KalmanTrack:
    def __init__(self, init_distance, init_angle_deg, dt=1.0):
        self.dt = float(dt)
        self.kf = cv2.KalmanFilter(4, 2)

        # ====== 固定模型結構 ======
        self.vel_decay_per_sec_d = 0.8   # 距離速度衰減（原值）
        self.vel_decay_per_sec_a = 0.3   # 角速度衰減（更快，→ 更黏量測的 yaw）
        self.kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)

        # 角度量測噪聲（固定）
        self.R_a = (0.05 ** 2)

        # --- 近/遠模式參數（可以依實測微調）---
        # 滯回門檻：<near_enter → 進入近距；>far_enter → 進入遠距；中間維持原模式
        self.near_enter = 90.0
        self.far_enter  = 110.0

        # 近距：更信量測（小 R_d），反應快（Q 稍大），gate 寬一點
        self.near_R_d = (0.8 ** 2)         # 距離量測噪聲（近距）
        self.near_gate_sigma = 3.0         # 近距 gate（σ倍）
        self.near_Q_scale = 1.5            # 近距 Q 放大係數（>1 更靈敏）

        # 遠距：更信預測（大 R_d），更抗跳（Q 稍小），gate 緊一些
        self.far_R_d  = (2.5 ** 2)         # 距離量測噪聲（遠距）
        self.far_gate_sigma = 1.6          # 遠距 gate（σ倍）
        self.far_Q_scale = 0.6             # 遠距 Q 縮小係數（<1 更平滑）

        # Process noise 的基準（你原來的對角 Q），後面依模式乘上 scale
        self._Q_base_diag = np.array([5e-4, 8e-4, 2e-3, 5e-3], dtype=np.float32)

        # ✅ 暖機設定（前幾幀快速收斂）
        self.warmup_frames = 3
        self.warmup_gate_sigma = 1e9  # 幾乎不裁
        self.Rd_warmup = (0.8 ** 2)   # 更信距離

        # 內部狀態
        self._mode = None  # 'near' 或 'far'
        self.gate_sigma_d = None
        self.R_d = None

        # 初始狀態：用第一筆量測
        self.kf.statePost = np.array([[init_distance],
                                      [init_angle_deg],
                                      [0.0],
                                      [0.0]], dtype=np.float32)
        # 放大初始不確定度，讓早期更相信量測
        P0 = np.diag([30.0**2, 10.0**2, 5.0**2, 5.0**2]).astype(np.float32)
        self.kf.errorCovPost = P0

        self.last_angle_deg = float(init_angle_deg)
        self.age = 0

        # 依初始距離決定模式（在 [80,90) 內預設用「近距」）
        init_d = float(init_distance)
        if   init_d > self.far_enter:  self._mode = 'far'
        else:                          self._mode = 'near'

        self._update_transition()
        self._apply_mode_params()  # 套用對應模式的 Q / R / gate

    # --------- 內部：根據 dt 與速度衰減更新 F ---------
    def _decay_factor(self, base, dt):
        return float(base) ** max(dt, 1e-6)

    def _update_transition(self):
        dt = self.dt
        kd = self._decay_factor(self.vel_decay_per_sec_d, dt)
        ka = self._decay_factor(self.vel_decay_per_sec_a, dt)
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, kd,  0],
            [0, 0, 0,  ka],
        ], dtype=np.float32)

    def _apply_mode_params(self):
        """依目前模式設定 Q 與距離量測/門檻。"""
        if self._mode == 'near':
            self.R_d = self.near_R_d
            self.gate_sigma_d = self.near_gate_sigma
            scale = self.near_Q_scale
        else:
            self.R_d = self.far_R_d
            self.gate_sigma_d = self.far_gate_sigma
            scale = self.far_Q_scale

        # 直接用你原本的對角 Q 乘縮放（簡單有效）
        self.kf.processNoiseCov = np.diag(self._Q_base_diag * scale).astype(np.float32)

        # 量測噪聲矩陣（角度固定、距離依模式，真正使用時會在 correct() 裡再針對「暖機」覆寫距離那一格）
        self.kf.measurementNoiseCov = np.diag([self.R_d, self.R_a]).astype(np.float32)

    def _maybe_switch_mode(self):
        """用上一次的後驗距離做滯回切換：<80 近距、>90 遠距；中間維持。"""
        d_last = float(self.kf.statePost[0,0])
        if self._mode == 'near':
            if d_last > self.far_enter:
                self._mode = 'far'
                self._apply_mode_params()
        else:  # far
            if d_last < self.near_enter:
                self._mode = 'near'
                self._apply_mode_params()

    def set_dt(self, dt):
        dt = max(1.0/60.0, min(float(dt), 0.2))  # [~16.7ms, 200ms]
        self.dt = dt
        self._update_transition()

    def predict(self):
        pred = self.kf.predict()
        return float(pred[0,0]), float(pred[1,0])

    def correct(self, distance, angle_deg):
        angle_unwrapped = unwrap_deg(self.last_angle_deg, float(angle_deg))
        z_pred_d = float(self.kf.statePre[0,0])

        # 暖機：不裁切且更信距離；否則依模式 gate/R_d
        if self.age < self.warmup_frames:
            gate_sigma = self.warmup_gate_sigma
            R_d_use = min(self.R_d, self.Rd_warmup)
        else:
            gate_sigma = self.gate_sigma_d
            R_d_use = self.R_d

        innov_d = float(distance) - z_pred_d
        sigma_d = math.sqrt(R_d_use)
        max_innov = gate_sigma * sigma_d
        innov_d = max(-max_innov, min(max_innov, innov_d))  # clip
        distance_adj = z_pred_d + innov_d

        # 臨時覆寫量測噪聲（距離用 R_d_use、角度固定）
        R_backup = self.kf.measurementNoiseCov.copy()
        self.kf.measurementNoiseCov[:] = np.diag([R_d_use, self.R_a]).astype(np.float32)

        meas = np.array([[distance_adj], [angle_unwrapped]], dtype=np.float32)
        corr = self.kf.correct(meas)

        self.kf.measurementNoiseCov[:] = R_backup

        d = float(corr[0,0]); a = float(corr[1,0])
        self.last_angle_deg = a
        self.age += 1
        return d, a

    def update(self, distance, angle_deg, dt=None):
        if dt is not None:
            self.set_dt(dt)

        # 先依上一次後驗距離做滯回切換，再做 predict/correct
        self._maybe_switch_mode()
        _ = self.predict()
        return self.correct(distance, angle_deg)

class ObjectTracker:
    def __init__(self, auto_reset=True, empty_frames_to_reset=30):
        """
        :param auto_reset: 是否啟用自動歸零 ID
        :param empty_frames_to_reset: 連續多少幀完全無偵測物件時重置
        """
        self.auto_reset = bool(auto_reset)
        self.empty_frames_to_reset = int(empty_frames_to_reset)
        self.track_id_counter = 0
        self.previous_objects = []
        self.kalman_tracks = {}  # track_id: KalmanTrack
        self.empty_streak = 0

    def reset(self):
        self.track_id_counter = 0
        self.previous_objects = []
        self.kalman_tracks.clear()
        self.empty_streak = 0

    # --- IoU 輔助 ---
    def expand_bbox(self, bbox, scale):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        new_w = w * scale
        new_h = h * scale
        new_x1 = cx - new_w / 2
        new_y1 = cy - new_h / 2
        new_x2 = cx + new_w / 2
        new_y2 = cy + new_h / 2
        return [new_x1, new_y1, new_x2, new_y2]

    def get_expand_factor(self, bbox, min_size=30):
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        if area <= 0:
            return 1.0
        factor = max(1.0, min(2.5, (min_size * min_size) / area))
        return factor

    def iou_bbox(self, bbox1, bbox2):
        scale1 = self.get_expand_factor(bbox1)
        scale2 = self.get_expand_factor(bbox2)
        bbox1 = self.expand_bbox(bbox1, scale1)
        bbox2 = self.expand_bbox(bbox2, scale2)
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        inter_x1 = max(x1_min, x2_min)
        inter_y1 = max(y1_min, y2_min)
        inter_x2 = min(x1_max, x2_max)
        inter_y2 = min(y1_max, y2_max)
        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def is_same_object(self, obj1, obj2, iou_thresh=0.3):
        if obj1["class"] != obj2["class"]:
            return False
        iou = self.iou_bbox(obj1["bbox"], obj2["bbox"])
        return iou >= iou_thresh

    def assign_track_ids(self, current_objects, dt=None):
        """
        current_objects: list of dict
          期待至少有: class, bbox, distance, angle, scale, color
        回傳: 已附加 track_id，且 distance/angle 經 Kalman 平滑
        """
        if not current_objects:
            self.empty_streak += 1
            self.previous_objects = []
            if self.auto_reset and self.empty_streak >= self.empty_frames_to_reset:
                self.reset()
            return []
        else:
            self.empty_streak = 0

        if not self.previous_objects:
            assigned = []
            for obj in current_objects:
                obj = dict(obj)
                obj["track_id"] = self.track_id_counter
                d0 = float(obj.get("distance", 0.0))
                a0 = float(obj.get("angle", 0.0))
                self.kalman_tracks[self.track_id_counter] = KalmanTrack(d0, a0)
                d_f, a_f = self.kalman_tracks[self.track_id_counter].update(d0, a0, dt=dt)
                obj["distance"] = d_f
                obj["angle"]    = a_f
                assigned.append(obj)
                self.track_id_counter += 1
            self.previous_objects = assigned
            return assigned

        # 之後幀：Hungarian + IoU 配對
        num_prev = len(self.previous_objects)
        num_curr = len(current_objects)
        cost_matrix = np.ones((num_prev, num_curr), dtype=np.float32)

        for i, prev_obj in enumerate(self.previous_objects):
            for j, curr_obj in enumerate(current_objects):
                if prev_obj["class"] != curr_obj["class"]:
                    cost_matrix[i, j] = 1.0
                else:
                    iou = self.iou_bbox(prev_obj["bbox"], curr_obj["bbox"])
                    cost_matrix[i, j] = 1.0 - iou

        row_idx, col_idx = linear_sum_assignment(cost_matrix)

        assigned = []
        used_curr = set()
        used_prev_ids = set()

        for i, j in zip(row_idx, col_idx):
            if cost_matrix[i, j] < (1.0 - 0.3):  # IOU >= 0.3
                track_id = self.previous_objects[i]["track_id"]
                obj = dict(current_objects[j])
                obj["track_id"] = track_id

                if track_id not in self.kalman_tracks:
                    d0 = float(obj.get("distance", 0.0))
                    a0 = float(obj.get("angle", 0.0))
                    self.kalman_tracks[track_id] = KalmanTrack(d0, a0)

                d_meas = float(obj.get("distance", 0.0))
                a_meas = float(obj.get("angle", 0.0))
                d_f, a_f = self.kalman_tracks[track_id].update(d_meas, a_meas, dt=dt)
                obj["distance"] = d_f
                obj["angle"]    = a_f

                used_curr.add(j)
                used_prev_ids.add(track_id)
                assigned.append(obj)

        # 新 track
        for j, obj_in in enumerate(current_objects):
            if j in used_curr:
                continue
            obj = dict(obj_in)
            for candidate_id in range(self.track_id_counter + 1):
                if candidate_id not in used_prev_ids:
                    obj["track_id"] = candidate_id
                    break
            else:
                self.track_id_counter += 1
                obj["track_id"] = self.track_id_counter

            d0 = float(obj.get("distance", 0.0))
            a0 = float(obj.get("angle", 0.0))
            self.kalman_tracks[obj["track_id"]] = KalmanTrack(d0, a0)
            d_f, a_f = self.kalman_tracks[obj["track_id"]].update(d0, a0, dt=dt)
            obj["distance"] = d_f
            obj["angle"]    = a_f

            assigned.append(obj)
            used_prev_ids.add(obj["track_id"])

        self.previous_objects = assigned
        return assigned
