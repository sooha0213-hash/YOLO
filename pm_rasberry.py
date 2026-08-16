import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import cv2
import numpy as np
import onnxruntime as ort
import time
from scipy.optimize import linear_sum_assignment
from flask import Flask, Response
from collections import deque

# =========================
# 설정값 (라즈베리파이 최적화)
# =========================
YOLO_SIZE = 320
CONF_THRESHOLD = 0.20
NMS_THRESHOLD = 0.45
FRAME_SKIP = 2
ENABLE_LANE = False

# [추가] 거리 계산용 임시 초점거리
# Camera Module 3 장착 후 실제 캘리브레이션 필요
FOCAL_LENGTH = 500.0

# [추가] 객체 추적이 잠깐 끊겨도 ID를 유지할 프레임 수
MAX_LOST_FRAMES = 5

# [추가] 접근/이탈 판단 시 작은 거리 변화는 노이즈로 무시
MOTION_THRESHOLD = 0.3

# [추가] TTC가 지나치게 작은 노이즈 값으로 튀는 것을 방지
MIN_APPROACH_SPEED = 0.3

# [추가] 위험도 우선순위
RISK_LEVEL = {
    "SAFE": 0,
    "CAUTION": 1,
    "WARNING": 2,
    "DANGER": 3
}

# [추가] 위험도별 화면 색상 (BGR)
RISK_COLORS = {
    "SAFE": (0, 255, 0),        # 초록
    "CAUTION": (0, 255, 255),   # 노랑
    "WARNING": (0, 165, 255),   # 주황
    "DANGER": (0, 0, 255),      # 빨강
    "UNKNOWN": (200, 200, 200)
}

# 클래스별 실제 높이(m)
CLASS_INFO = {
    0: ("Person", 1.7),
    1: ("Bicycle", 1.6),
    2: ("Car", 1.5),
    3: ("Motorcycle", 1.6),
    5: ("Bus", 3.0),
    7: ("Truck", 2.5),
}

cv2.setNumThreads(1)
app = Flask(__name__)

# =========================
# IoU 계산
# =========================
def bbox_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    return inter_area / float(box1_area + box2_area - inter_area + 1e-5)

# =========================
# Tracking
# =========================
class STrack:
    def __init__(self, bbox, score, cls_id):
        self.bbox = np.array(bbox, dtype=np.float32)
        self.score = score
        self.cls_id = cls_id
        self.track_id = 0

        # [추가]
        # YOLO가 잠깐 객체를 놓쳤다고 바로 track을 삭제하지 않기 위해 사용
        self.lost_frames = 0

class SimpleByteTracker:
    def __init__(
        self,
        track_thresh=0.45,
        match_thresh=0.45,
        max_lost_frames=MAX_LOST_FRAMES
    ):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.max_lost_frames = max_lost_frames

        self.tracked_stracks = []
        self.next_id = 1

    def update(self, output_results):
        detections = []
        for res in output_results:
            x1, y1, w, h, score, cls_id = res

            if score >= self.track_thresh:
                detections.append(
                    STrack(
                        [x1, y1, x1 + w, y1 + h],
                        score,
                        cls_id
                    )
                )

        # [수정]
        # 검출 결과가 없더라도 기존 track을 즉시 전부 삭제하지 않음
        if len(detections) == 0:
            alive_tracks = []

            for track in self.tracked_stracks:
                track.lost_frames += 1
                if track.lost_frames <= self.max_lost_frames:
                    alive_tracks.append(track)
            self.tracked_stracks = alive_tracks

            # 화면에는 이번에 실제 검출된 객체만 반환
            return []

        matches, u_track, u_det = self.linear_assignment(
            self.tracked_stracks,
            detections,
            self.match_thresh
        )

        # =========================
        # 정상 매칭된 객체
        # =========================
        for t_idx, d_idx in matches:
            track = self.tracked_stracks[t_idx]
            det = detections[d_idx]

            track.bbox = det.bbox
            track.score = det.score
            track.cls_id = det.cls_id

            # [추가] 다시 검출됐으므로 lost 초기화
            track.lost_frames = 0

        # =========================
        # 이번 프레임에서 놓친 기존 객체
        # =========================
        for t_idx in u_track:
            self.tracked_stracks[t_idx].lost_frames += 1

        # 너무 오래 검출되지 않은 객체만 제거
        self.tracked_stracks = [
            track
            for track in self.tracked_stracks
            if track.lost_frames <= self.max_lost_frames
        ]

        # =========================
        # 새롭게 등장한 객체
        # =========================
        for d_idx in u_det:
            new_track = detections[d_idx]
            new_track.track_id = self.next_id
            self.next_id += 1
            self.tracked_stracks.append(new_track)

        # 실제 현재 프레임에서 검출된 track만 화면에 표시
        visible_tracks = [
            track
            for track in self.tracked_stracks
            if track.lost_frames == 0
        ]

        return visible_tracks

    def linear_assignment(self, tracks, dets, thresh):
        if not tracks or not dets:
            return (
                [],
                list(range(len(tracks))),
                list(range(len(dets)))
            )

        iou_matrix = np.zeros(
            (len(tracks), len(dets)),
            dtype=np.float32
        )

        for t, track in enumerate(tracks):
            for d, det in enumerate(dets):
                # [추가] 다른 클래스끼리는 같은 객체로 매칭하지 않음
                if track.cls_id != det.cls_id:
                    iou_matrix[t, d] = 0.0
                else:
                    iou_matrix[t, d] = bbox_iou(
                        track.bbox,
                        det.bbox
                    )

        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        matches = []
        u_track = list(range(len(tracks)))
        u_det = list(range(len(dets)))

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= thresh:
                matches.append((r, c))
                if r in u_track:
                    u_track.remove(r)
                if c in u_det:
                    u_det.remove(c)

        return matches, u_track, u_det


# =========================
# 도로 / 차선 인식
# =========================
def process_road_and_lanes(frame):
    h, w = frame.shape[:2]
    overlay = np.zeros_like(frame)
    gray_orig = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    equalized_gray = clahe.apply(gray_orig)
    ycrcb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2YCrCb
    )

    channels = list(cv2.split(ycrcb))
    channels[0] = equalized_gray

    blur_frame = cv2.cvtColor(
        cv2.merge(channels),
        cv2.COLOR_YCrCb2BGR
    )

    blur = cv2.GaussianBlur(
        blur_frame,
        (5, 5),
        0
    )

    edges_wall = cv2.Canny(
        equalized_gray,
        40,
        120
    )

    kernel_wall = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5)
    )

    wall_mask_dilated = cv2.dilate(
        edges_wall,
        kernel_wall,
        iterations=1
    )

    hsv = cv2.cvtColor(
        blur,
        cv2.COLOR_BGR2HSV
    )

    lower_road = np.array([0, 0, 50])
    upper_road = np.array([180, 50, 180])

    road_mask = cv2.inRange(
        hsv,
        lower_road,
        upper_road
    )

    road_mask = cv2.bitwise_and(
        road_mask,
        cv2.bitwise_not(wall_mask_dilated)
    )

    roi_mask = np.zeros_like(road_mask)

    poly_points = np.array([
        [0, h],
        [int(w * 0.35), int(h * 0.55)],
        [int(w * 0.65), int(h * 0.55)],
        [w, h]
    ], np.int32)

    cv2.fillPoly(
        roi_mask,
        [poly_points],
        255
    )

    dynamic_road = cv2.bitwise_and(
        road_mask,
        roi_mask
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5)
    )

    dynamic_road = cv2.morphologyEx(
        dynamic_road,
        cv2.MORPH_CLOSE,
        kernel
    )

    dynamic_road = cv2.morphologyEx(
        dynamic_road,
        cv2.MORPH_OPEN,
        kernel
    )

    contours, _ = cv2.findContours(
        dynamic_road,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    final_road_mask = np.zeros_like(dynamic_road)

    if contours:

        largest_contour = max(
            contours,
            key=cv2.contourArea
        )

        if cv2.contourArea(largest_contour) > (h * w * 0.04):

            cv2.drawContours(
                final_road_mask,
                [largest_contour],
                -1,
                255,
                -1
            )

    overlay[final_road_mask > 0] = [0, 255, 0]

    if ENABLE_LANE:

        edges = cv2.Canny(
            blur,
            50,
            150
        )

        lane_roi = cv2.bitwise_and(
            edges,
            roi_mask
        )

        lines = cv2.HoughLinesP(
            lane_roi,
            rho=1,
            theta=np.pi / 180,
            threshold=35,
            minLineLength=35,
            maxLineGap=25
        )

        if lines is not None:
            for line in lines[:30]:
                x1, y1, x2, y2 = line[0]

                dx = abs(x2 - x1)
                dy = abs(y2 - y1)

                if dx > 0 and (dy / dx) > 0.7:
                    cv2.line(
                        overlay,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        3
                    )

    return cv2.addWeighted(
        frame,
        1.0,
        overlay,
        0.25,
        0
    )


# ============================================================
# [추가 ①] 거리 기반 위험도
# ============================================================
def get_distance_risk(distance):

    # 5m 이하
    if distance <= 5.0:
        return "DANGER"

    # 5~10m
    elif distance <= 10.0:
        return "CAUTION"

    # 10m 초과
    else:
        return "SAFE"


# ============================================================
# [추가 ②] 객체별 거리 기록
# ============================================================

# 최근 거리 5개만 저장
distance_history = {}

# 해당 거리를 측정한 시간 저장
time_history = {}


def update_distance_history(track_id, distance, current_time):

    if track_id not in distance_history:

        distance_history[track_id] = deque(maxlen=5)
        time_history[track_id] = deque(maxlen=5)

    distance_history[track_id].append(distance)
    time_history[track_id].append(current_time)


# ============================================================
# [추가 ③] 접근속도 계산
# ============================================================
def calculate_approach_speed(track_id):
    distances = distance_history.get(track_id)
    times = time_history.get(track_id)

    if distances is None or times is None:
        return 0.0
    # 최소 3개 측정값이 있어야 판단
    if len(distances) < 3:
        return 0.0

    # 오래된 거리 - 현재 거리
    distance_change = distances[0] - distances[-1]
    time_change = times[-1] - times[0]
    if time_change <= 0:
        return 0.0

    # 양수 = 가까워지고 있음, 음수 = 멀어지고 있음
    approach_speed = distance_change / time_change
    return approach_speed


# ============================================================
# [추가 ③-2] 접근 / 정지 / 이탈 판단
# ============================================================
def get_motion_state(approach_speed):
    if approach_speed > MOTION_THRESHOLD:
        return "APPROACHING"
    elif approach_speed < -MOTION_THRESHOLD:
        return "LEAVING"
    else:
        return "STABLE"


# ============================================================
# [추가 ④] TTC(Time To Collision) 계산
# ============================================================
def calculate_ttc(distance, approach_speed):
    # 가까워지는 객체에 대해서만 TTC 계산
    if approach_speed <= MIN_APPROACH_SPEED:
        return None
    # TTC = 현재 거리 / 상대 접근속도
    return distance / approach_speed


# ============================================================
# [추가 ④-2] TTC 위험도
# ============================================================
def get_ttc_risk(ttc):
    # 접근 중이 아니면 TTC 위험 없음
    if ttc is None:
        return "SAFE"
    # 충돌까지 1.5초 이하
    if ttc <= 1.5:
        return "DANGER"
    # 1.5 ~ 3초
    elif ttc <= 3.0:
        return "WARNING"
    # 3 ~ 5초
    elif ttc <= 5.0:
        return "CAUTION"
    # 5초 초과
    else:
        return "SAFE"


# ============================================================
# [추가 ⑤] 거리 + TTC 종합 위험도
# ============================================================
def get_final_risk(distance_risk, ttc_risk):
    # 둘 중 더 위험한 단계 선택
    if RISK_LEVEL[ttc_risk] > RISK_LEVEL[distance_risk]:
        return ttc_risk

    return distance_risk

# FPS
fps_list = []

# =========================
# 메인 영상 처리
# =========================
def generate_frames(
    cap,
    yolo_session,
    yolo_input_name,
    tracker
):

    frame_count = 0
    last_online_targets = []

    while True:
        start_time = time.time()
        ret, frame = cap.read()

        # 영상 끝나면 처음으로
        if not ret:
            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                0
            )

            # 영상이 처음으로 돌아가면 이전 영상의 거리 기록 제거
            distance_history.clear()
            time_history.clear()
            continue

        frame_count += 1
        h_orig, w_orig = frame.shape[:2]
        raw_frame = frame.copy()
        display_frame = process_road_and_lanes(
            raw_frame
        )

        # ====================================================
        # YOLO 추론
        # ====================================================
        if frame_count % FRAME_SKIP == 0:
            # Letterbox resize
            scale = min(
                YOLO_SIZE / w_orig,
                YOLO_SIZE / h_orig
            )

            new_w = int(round(w_orig * scale))
            new_h = int(round(h_orig * scale))

            resized = cv2.resize(
                raw_frame,
                (new_w, new_h),
                interpolation=cv2.INTER_LINEAR
            )

            img_yolo = np.full(
                (YOLO_SIZE, YOLO_SIZE, 3),
                114,
                dtype=np.uint8
            )

            pad_x = (YOLO_SIZE - new_w) // 2
            pad_y = (YOLO_SIZE - new_h) // 2

            img_yolo[
                pad_y:pad_y + new_h,
                pad_x:pad_x + new_w
            ] = resized

            img_yolo = cv2.cvtColor(
                img_yolo,
                cv2.COLOR_BGR2RGB
            )

            yolo_tensor = np.ascontiguousarray(
                img_yolo
                .transpose(2, 0, 1)[np.newaxis]
            ).astype(np.float32) / 255.0

            try:

                yolo_outputs = yolo_session.run(
                    None,
                    {yolo_input_name: yolo_tensor}
                )

            except Exception as e:
                print(
                    f"❌ YOLO 추론 오류: {e}"
                )

                break

            predictions = np.squeeze(
                yolo_outputs[0]
            ).T

            target_classes = {
                0, 1, 2, 3, 5, 7
            }

            boxes = []
            confidences = []
            class_ids = []

            for pred in predictions:

                scores = pred[4:]

                class_id = int(
                    np.argmax(scores)
                )

                confidence = float(
                    scores[class_id]
                )

                if class_id not in target_classes:
                    continue

                # 사람은 오탐 방지를 위해 조금 높은 confidence 사용
                required_conf = (
                    0.40
                    if class_id == 0
                    else 0.20
                )

                if confidence <= required_conf:
                    continue

                cx, cy, w, h = pred[0:4]

                # Letterbox 좌표 → 원본 영상 좌표
                x1 = int(
                    (cx - w / 2 - pad_x)
                    / scale
                )

                y1 = int(
                    (cy - h / 2 - pad_y)
                    / scale
                )

                bw = int(w / scale)
                bh = int(h / scale)

                # [추가] 좌표가 영상 밖으로 나가지 않도록 제한
                x1 = max(0, min(x1, w_orig - 1))
                y1 = max(0, min(y1, h_orig - 1))

                bw = min(
                    bw,
                    w_orig - x1
                )

                bh = min(
                    bh,
                    h_orig - y1
                )

                if bw <= 0 or bh <= 0:
                    continue

                boxes.append(
                    [x1, y1, bw, bh]
                )

                confidences.append(
                    confidence
                )

                class_ids.append(
                    class_id
                )

            # =================================================
            # NMS
            # =================================================
            tracker_inputs = []
            if len(boxes) > 0:
                indices = cv2.dnn.NMSBoxes(
                    boxes,
                    confidences,
                    score_threshold=CONF_THRESHOLD,
                    nms_threshold=NMS_THRESHOLD
                )

                if len(indices) > 0:
                    for i in indices.flatten():
                        x, y, w, h = boxes[i]

                        # 너무 작은 객체 제거
                        if h < 20 or w < 10:
                            continue
                        tracker_inputs.append([
                            x,
                            y,
                            w,
                            h,
                            confidences[i],
                            class_ids[i]
                        ])
            # 객체 추적
            last_online_targets = tracker.update(
                tracker_inputs
            )

        # ====================================================
        # 위험도 계산
        # ====================================================

        # [추가]
        # 현재 프레임에 DANGER가 하나라도 있는지 확인
        danger_detected = False

        for target in last_online_targets:

            x1, y1, x2, y2 = map(
                int,
                target.bbox
            )

            # 좌표 안전 처리
            x1 = max(0, min(x1, w_orig - 1))
            y1 = max(0, min(y1, h_orig - 1))
            x2 = max(0, min(x2, w_orig - 1))
            y2 = max(0, min(y2, h_orig - 1))

            h_box = y2 - y1

            cls_id = target.cls_id
            track_id = target.track_id

            class_name, real_h = CLASS_INFO.get(
                cls_id,
                ("Unknown", 1.5)
            )

            if h_box > 0:

                # =============================================
                # 거리 추정
                # =============================================
                distance = (
                    real_h * FOCAL_LENGTH
                ) / h_box

                current_time = time.time()

                # =============================================
                # ① 거리 위험도
                # =============================================
                distance_risk = get_distance_risk(
                    distance
                )

                # =============================================
                # ② 동일 ID 거리 기록
                # =============================================
                update_distance_history(
                    track_id,
                    distance,
                    current_time
                )

                # =============================================
                # ③ 접근속도
                # =============================================
                approach_speed = (
                    calculate_approach_speed(
                        track_id
                    )
                )

                motion_state = get_motion_state(
                    approach_speed
                )

                # =============================================
                # ④ TTC
                # =============================================
                if motion_state == "APPROACHING":

                    ttc = calculate_ttc(
                        distance,
                        approach_speed
                    )

                else:

                    ttc = None

                ttc_risk = get_ttc_risk(
                    ttc
                )

                # =============================================
                # ⑤ 최종 위험도
                # =============================================
                final_risk = get_final_risk(
                    distance_risk,
                    ttc_risk
                )

                risk_color = RISK_COLORS[
                    final_risk
                ]

                # 화면에는 '클래스 + 거리만 표시'
                # TTC / ID / 접근속도 등은 내부 계산에만 사용
                label = (f"{class_name} " f"{distance:.1f}m")

                if final_risk == "DANGER":
                    danger_detected = True

                # 개발 중 확인용 터미널 출력
                print(
                    f"ID:{track_id} "
                    f"{class_name} | "
                    f"{distance:.1f}m | "
                    f"{motion_state} | "
                    f"speed:{approach_speed:.2f}m/s | "
                    f"TTC:{ttc:.2f}s | "
                    f"{final_risk}"
                    if ttc is not None
                    else
                    f"ID:{track_id} "
                    f"{class_name} | "
                    f"{distance:.1f}m | "
                    f"{motion_state} | "
                    f"speed:{approach_speed:.2f}m/s | "
                    f"TTC:- | "
                    f"{final_risk}"
                )

            else:
                final_risk = "UNKNOWN"
                risk_color = RISK_COLORS[ "UNKNOWN" ]
                label = class_name

            # =============================================
            # 객체 Bounding Box
            # =============================================
            cv2.rectangle(
                display_frame,
                (x1, y1),
                (x2, y2),
                risk_color,
                2
            )
            # =============================================
            # 깔끔한 라벨 배경
            # =============================================
            (text_w, text_h), baseline = (
                cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    1
                )
            )

            label_y = max(y1 - text_h - 10, 0)

            cv2.rectangle(
                display_frame,
                (x1, label_y),
                (
                    min(x1 + text_w + 10, w_orig - 1),
                    min(label_y + text_h + 8, h_orig - 1)
                ),
                risk_color,
                -1
            )
            cv2.putText(
                display_frame,
                label,
                (x1 + 5, label_y + text_h + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

        # ====================================================
        # [추가] DANGER가 있으면 화면 상단에 한 번만 경고
        # ====================================================
        if danger_detected:
            warning_text = "COLLISION RISK"
            (tw, th), _ = cv2.getTextSize(
                warning_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                2
            )

            center_x = w_orig // 2
            cv2.rectangle(
                display_frame,
                (
                    center_x - tw // 2 - 15,
                    15
                ),
                (
                    center_x + tw // 2 + 15,
                    55
                ),
                (0, 0, 255),
                -1
            )

            cv2.putText(
                display_frame,
                warning_text,
                (
                    center_x - tw // 2,
                    45
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        # ====================================================
        # FPS
        # ====================================================
        elapsed_time = max(
            time.time() - start_time,
            0.001
        )

        fps_list.append(
            1.0 / elapsed_time
        )

        if len(fps_list) > 30:
            fps_list.pop(0)

        fps = sum(fps_list) / len(fps_list)

        # 개발 중이므로 FPS만 표시
        cv2.putText(
            display_frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        # JPEG 스트리밍
        success, buffer = cv2.imencode(
            ".jpg",
            display_frame,
            [cv2.IMWRITE_JPEG_QUALITY, 70]
        )

        if not success:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


# =========================
# Flask
# =========================
cap = None
yolo_session = None
yolo_input_name = None
tracker = None


@app.route("/")
def index():

    return """
    <html>
    <head>
        <title>PM ADAS - RaspberryPi</title>
    </head>

    <body style="
        background:#000;
        text-align:center;
        margin:0;
    ">

        <h2 style="color:white;">
            PM ADAS Live Stream
        </h2>

        <img
            src="/video_feed"
            width="640"
        >

    </body>
    </html>
    """


@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(
            cap,
            yolo_session,
            yolo_input_name,
            tracker
        ),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# =========================
# Main
# =========================
def main():

    global cap
    global yolo_session
    global yolo_input_name
    global tracker

    print("1. 프로그램 시작됨...")

    yolo_onnx = "./yolov8n.onnx"

    cap = cv2.VideoCapture(
        "input_video.mp4"
    )

    if not cap.isOpened():

        print(
            "❌ 에러: 영상을 열 수 없습니다."
        )

        return

    print(
        "2. 영상 로드 성공!"
    )

    opts = ort.SessionOptions()

    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1

    opts.graph_optimization_level = (
        ort.GraphOptimizationLevel
        .ORT_ENABLE_ALL
    )

    try:

        yolo_session = (
            ort.InferenceSession(
                yolo_onnx,
                sess_options=opts,
                providers=[
                    "CPUExecutionProvider"
                ]
            )
        )

        print(
            "3. ONNX 모델 로드 완료!"
        )

    except Exception as e:

        print(
            f"❌ ONNX 모델 로드 실패: {e}"
        )

        return

    yolo_input_name = (
        yolo_session
        .get_inputs()[0]
        .name
    )

    # [수정]
    # 객체를 잠깐 놓쳐도 최대 5번까지 기존 ID 유지
    tracker = SimpleByteTracker(
        track_thresh=0.45,
        match_thresh=0.45,
        max_lost_frames=MAX_LOST_FRAMES
    )

    print(
        "🚀 Flask 서버 시작! "
        "브라우저에서 "
        "http://라즈베리파이IP:5000 접속"
    )

    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=False
    )


if __name__ == "__main__":
    main()