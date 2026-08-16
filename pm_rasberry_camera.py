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
from picamera2 import Picamera2

# =========================
# 설정값
# =========================
YOLO_SIZE = 320
CONF_THRESHOLD = 0.20
NMS_THRESHOLD = 0.45
FRAME_SKIP = 2

FOCAL_LENGTH = 524.0       # Camera Module 3 임시값, 추후 실제 캘리브레이션
MAX_LOST_FRAMES = 5        # 객체를 잠깐 놓쳐도 ID 유지
MOTION_THRESHOLD = 0.3     # 작은 거리 변화는 노이즈로 무시
MIN_APPROACH_SPEED = 0.3   # TTC 계산 최소 접근속도

RISK_COLORS = {
    "SAFE": (0, 255, 0),
    "CAUTION": (0, 255, 255),
    "WARNING": (0, 165, 255),
    "DANGER": (0, 0, 255),
    "UNKNOWN": (200, 200, 200)
}

# COCO 클래스: (이름, 평균 실제 높이 m)
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
# 객체 Tracking
# =========================
class STrack:
    def __init__(self, bbox, score, cls_id):
        self.bbox = np.array(bbox, dtype=np.float32)
        self.score = score
        self.cls_id = cls_id
        self.track_id = 0
        self.lost_frames = 0


class SimpleByteTracker:
    def __init__(self, track_thresh=0.45, match_thresh=0.45, max_lost_frames=MAX_LOST_FRAMES):
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
                detections.append(STrack([x1, y1, x1 + w, y1 + h], score, cls_id))

        # 검출이 잠깐 끊겨도 기존 ID 유지
        if len(detections) == 0:
            alive_tracks = []
            for track in self.tracked_stracks:
                track.lost_frames += 1
                if track.lost_frames <= self.max_lost_frames:
                    alive_tracks.append(track)
            self.tracked_stracks = alive_tracks
            return []

        matches, u_track, u_det = self.linear_assignment(
            self.tracked_stracks, detections, self.match_thresh
        )

        for t_idx, d_idx in matches:
            track = self.tracked_stracks[t_idx]
            det = detections[d_idx]
            track.bbox = det.bbox
            track.score = det.score
            track.cls_id = det.cls_id
            track.lost_frames = 0

        for t_idx in u_track:
            self.tracked_stracks[t_idx].lost_frames += 1

        self.tracked_stracks = [
            track for track in self.tracked_stracks
            if track.lost_frames <= self.max_lost_frames
        ]

        # 새 객체에 새로운 ID 부여
        for d_idx in u_det:
            new_track = detections[d_idx]
            new_track.track_id = self.next_id
            self.next_id += 1
            self.tracked_stracks.append(new_track)

        return [track for track in self.tracked_stracks if track.lost_frames == 0]

    def linear_assignment(self, tracks, dets, thresh):
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))

        iou_matrix = np.zeros((len(tracks), len(dets)), dtype=np.float32)

        for t, track in enumerate(tracks):
            for d, det in enumerate(dets):
                if track.cls_id != det.cls_id:  # 다른 클래스는 같은 객체로 매칭하지 않음
                    iou_matrix[t, d] = 0.0
                else:
                    iou_matrix[t, d] = bbox_iou(track.bbox, det.bbox)

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
# Collision Zone
# =========================
def get_collision_zone(frame_width, frame_height):
    # 킥보드 전방 예상 진행영역
    return np.array([
        [int(frame_width * 0.38), int(frame_height * 0.55)],
        [int(frame_width * 0.62), int(frame_height * 0.55)],
        [int(frame_width * 0.88), frame_height - 1],
        [int(frame_width * 0.12), frame_height - 1]
    ], dtype=np.int32)


def is_in_collision_zone(x1, y1, x2, y2, collision_zone):
    object_point = (int((x1 + x2) / 2), int(y2))  # bbox 아래 중앙 = 객체의 지면 위치
    return cv2.pointPolygonTest(collision_zone, object_point, False) >= 0


# =========================
# 거리 기록 / 접근속도 / TTC
# =========================
distance_history = {}
time_history = {}


def update_distance_history(track_id, distance, current_time):
    if track_id not in distance_history:
        distance_history[track_id] = deque(maxlen=5)
        time_history[track_id] = deque(maxlen=5)

    distance_history[track_id].append(distance)
    time_history[track_id].append(current_time)


def calculate_approach_speed(track_id):
    distances = distance_history.get(track_id)
    times = time_history.get(track_id)

    if distances is None or times is None or len(distances) < 3:
        return 0.0

    distance_change = distances[0] - distances[-1]  # 양수 = 가까워짐
    time_change = times[-1] - times[0]

    if time_change <= 0:
        return 0.0

    return distance_change / time_change


def get_motion_state(approach_speed):
    if approach_speed > MOTION_THRESHOLD:
        return "APPROACHING"
    elif approach_speed < -MOTION_THRESHOLD:
        return "LEAVING"
    return "STABLE"


def calculate_ttc(distance, approach_speed):
    if approach_speed <= MIN_APPROACH_SPEED:
        return None
    return distance / approach_speed  # TTC = 거리 / 상대 접근속도


# =========================
# 최종 위험도 판단
# 거리 + TTC + Collision Zone
# =========================
def get_final_risk(distance, ttc, in_collision_zone):
    if not in_collision_zone:  # 진행경로 밖
        return "CAUTION" if distance <= 5.0 else "SAFE"

    if ttc is not None and ttc <= 1.5:  # TTC 1.5초 이하
        return "DANGER"

    if distance <= 5.0:  # 진행경로 안 + 5m 이하
        return "DANGER"

    if ttc is not None and ttc <= 3.0:  # TTC 3초 이하
        return "WARNING"

    if distance <= 10.0:  # 진행경로 안 + 10m 이하
        return "CAUTION"

    if ttc is not None and ttc <= 5.0:  # TTC 5초 이하
        return "CAUTION"

    return "SAFE"


fps_list = []


# =========================
# 실시간 영상 처리
# =========================
def generate_frames(picam2, yolo_session, yolo_input_name, tracker):
    frame_count = 0
    last_online_targets = []

    while True:
        start_time = time.time()

        frame = picam2.capture_array("main")  # Camera Module 3 실시간 입력
        if frame is None:
            print("❌ 카메라 프레임 없음")
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_count += 1
        h_orig, w_orig = frame.shape[:2]
        raw_frame = frame.copy()
        display_frame = raw_frame.copy()  # 기존 도로 탐지 제거

        # =========================
        # YOLO 객체 탐지
        # =========================
        if frame_count % FRAME_SKIP == 0:  # Raspberry Pi 부하 감소
            scale = min(YOLO_SIZE / w_orig, YOLO_SIZE / h_orig)
            new_w = int(round(w_orig * scale))
            new_h = int(round(h_orig * scale))

            resized = cv2.resize(raw_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # 종횡비를 유지하는 Letterbox
            img_yolo = np.full((YOLO_SIZE, YOLO_SIZE, 3), 114, dtype=np.uint8)
            pad_x = (YOLO_SIZE - new_w) // 2
            pad_y = (YOLO_SIZE - new_h) // 2
            img_yolo[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
            img_yolo = cv2.cvtColor(img_yolo, cv2.COLOR_BGR2RGB)

            yolo_tensor = np.ascontiguousarray(
                img_yolo.transpose(2, 0, 1)[np.newaxis]
            ).astype(np.float32) / 255.0

            try:
                yolo_outputs = yolo_session.run(None, {yolo_input_name: yolo_tensor})
            except Exception as e:
                print(f"❌ YOLO 추론 오류: {e}")
                break

            predictions = np.squeeze(yolo_outputs[0]).T
            target_classes = {0, 1, 2, 3, 5, 7}
            boxes, confidences, class_ids = [], [], []

            for pred in predictions:
                scores = pred[4:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])

                if class_id not in target_classes:
                    continue

                required_conf = 0.40 if class_id == 0 else 0.20  # 사람은 오탐 방지 위해 높게 설정
                if confidence <= required_conf:
                    continue

                cx, cy, w, h = pred[:4]

                # Letterbox 좌표 → 원본 영상 좌표
                x1 = int((cx - w / 2 - pad_x) / scale)
                y1 = int((cy - h / 2 - pad_y) / scale)
                bw = int(w / scale)
                bh = int(h / scale)

                x1 = max(0, min(x1, w_orig - 1))
                y1 = max(0, min(y1, h_orig - 1))
                bw = min(bw, w_orig - x1)
                bh = min(bh, h_orig - y1)

                if bw <= 0 or bh <= 0:
                    continue

                boxes.append([x1, y1, bw, bh])
                confidences.append(confidence)
                class_ids.append(class_id)

            # NMS로 중복 Bounding Box 제거
            tracker_inputs = []

            if boxes:
                indices = cv2.dnn.NMSBoxes(
                    boxes, confidences,
                    score_threshold=CONF_THRESHOLD,
                    nms_threshold=NMS_THRESHOLD
                )

                if len(indices) > 0:
                    for i in indices.flatten():
                        x, y, w, h = boxes[i]

                        if h < 20 or w < 10:  # 너무 작은 객체 제거
                            continue

                        tracker_inputs.append([
                            x, y, w, h,
                            confidences[i],
                            class_ids[i]
                        ])

            last_online_targets = tracker.update(tracker_inputs)  # 객체 ID 추적

        # =========================
        # 위험도 판단
        # =========================
        danger_detected = False
        collision_zone = get_collision_zone(w_orig, h_orig)

        # 개발용 Collision Zone 표시
        cv2.polylines(display_frame, [collision_zone], True, (255, 255, 255), 1, cv2.LINE_AA)

        for target in last_online_targets:
            x1, y1, x2, y2 = map(int, target.bbox)

            x1 = max(0, min(x1, w_orig - 1))
            y1 = max(0, min(y1, h_orig - 1))
            x2 = max(0, min(x2, w_orig - 1))
            y2 = max(0, min(y2, h_orig - 1))

            h_box = y2 - y1
            cls_id = target.cls_id
            track_id = target.track_id
            class_name, real_h = CLASS_INFO.get(cls_id, ("Unknown", 1.5))

            if h_box > 0:
                distance = (real_h * FOCAL_LENGTH) / h_box  # ① 거리 추정
                current_time = time.time()

                update_distance_history(track_id, distance, current_time)  # ② 동일 객체 거리 기록

                approach_speed = calculate_approach_speed(track_id)  # ③ 접근속도 계산
                motion_state = get_motion_state(approach_speed)

                ttc = calculate_ttc(distance, approach_speed) if motion_state == "APPROACHING" else None  # ④ TTC

                in_collision_zone = is_in_collision_zone(
                    x1, y1, x2, y2, collision_zone
                )  # ⑤ 실제 진행경로 안인지 판단

                final_risk = get_final_risk(distance, ttc, in_collision_zone)  # ⑥ 최종 위험도
                risk_color = RISK_COLORS[final_risk]

                label = f"{class_name} {distance:.1f}m"  # 화면에는 핵심 정보만 표시

                if final_risk == "DANGER":
                    danger_detected = True

            else:
                risk_color = RISK_COLORS["UNKNOWN"]
                label = class_name

            # 위험도에 따라 Bounding Box 색상 변경
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), risk_color, 2)

            # 객체명 + 거리 라벨
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            label_y = max(y1 - text_h - 10, 0)

            cv2.rectangle(
                display_frame,
                (x1, label_y),
                (min(x1 + text_w + 10, w_orig - 1), min(label_y + text_h + 8, h_orig - 1)),
                risk_color,
                -1
            )

            cv2.putText(
                display_frame, label, (x1 + 5, label_y + text_h + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
            )

        # DANGER 객체가 하나라도 있으면 상단 충돌 경고
        if danger_detected:
            warning_text = "COLLISION RISK"
            (tw, _), _ = cv2.getTextSize(
                warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
            )

            center_x = w_orig // 2

            cv2.rectangle(
                display_frame,
                (center_x - tw // 2 - 15, 15),
                (center_x + tw // 2 + 15, 55),
                (0, 0, 255),
                -1
            )

            cv2.putText(
                display_frame, warning_text, (center_x - tw // 2, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA
            )

        # FPS 계산
        elapsed_time = max(time.time() - start_time, 0.001)
        fps_list.append(1.0 / elapsed_time)

        if len(fps_list) > 30:
            fps_list.pop(0)

        fps = sum(fps_list) / len(fps_list)
        cv2.putText(
            display_frame, f"FPS: {fps:.1f}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )

        # Flask 스트리밍용 JPEG 변환
        success, buffer = cv2.imencode(
            ".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
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
picam2 = None
yolo_session = None
yolo_input_name = None
tracker = None


@app.route("/")
def index():
    return """
    <html>
    <head><title>PM ADAS - RaspberryPi</title></head>
    <body style="background:#000; text-align:center; margin:0;">
        <h2 style="color:white;">PM ADAS Live Stream</h2>
        <img src="/video_feed" width="640">
    </body>
    </html>
    """


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(picam2, yolo_session, yolo_input_name, tracker),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# =========================
# Main
# =========================
def main():
    global picam2, yolo_session, yolo_input_name, tracker

    print("1. 프로그램 시작됨...")
    yolo_onnx = "./yolov8n.onnx"

    # Camera Module 3 초기화
    picam2 = Picamera2()

    camera_config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"},
        controls={"FrameRate": 30}
    )

    picam2.configure(camera_config)
    picam2.start()
    time.sleep(1.0)  # 카메라 자동 노출/초점 안정화 대기

    print("2. Camera Module 3 시작 완료!")

    # ONNX Runtime Raspberry Pi CPU 최적화
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    try:
        yolo_session = ort.InferenceSession(
            yolo_onnx,
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        print("3. ONNX 모델 로드 완료!")

    except Exception as e:
        print(f"❌ ONNX 모델 로드 실패: {e}")
        picam2.stop()
        return

    yolo_input_name = yolo_session.get_inputs()[0].name

    tracker = SimpleByteTracker(
        track_thresh=0.45,
        match_thresh=0.45,
        max_lost_frames=MAX_LOST_FRAMES
    )

    print("🚀 Flask 서버 시작! 브라우저에서 http://라즈베리파이IP:5000 접속")

    try:
        app.run(host="0.0.0.0", port=5000, threaded=False)
    finally:
        if picam2 is not None:
            picam2.stop()


if __name__ == "__main__":
    main()