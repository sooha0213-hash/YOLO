import os

import cv2
import numpy as np
import onnxruntime as ort
import time
from scipy.optimize import linear_sum_assignment
from flask import Flask, Response

# =========================
# 설정값 (PC / CPU 최적화)
# =========================
YOLO_SIZE = 320  # yolov8n.onnx가 320x320 고정 입력으로 export된 모델이라 이 값을 바꾸면 추론 오류 발생
CONF_THRESHOLD = 0.20
NMS_THRESHOLD = 0.45
FRAME_SKIP = 1
ENABLE_LANE = False

app = Flask(__name__)


def bbox_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter_area / float(box1_area + box2_area - inter_area + 1e-5)


class STrack:
    def __init__(self, bbox, score, cls_id):
        self.bbox = np.array(bbox)
        self.score = score
        self.cls_id = cls_id
        self.track_id = 0


class SimpleByteTracker:
    def __init__(self, track_thresh=0.45, match_thresh=0.45):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.tracked_stracks = []
        self.next_id = 1

    def update(self, output_results):
        if len(output_results) == 0:
            self.tracked_stracks.clear()
            return self.tracked_stracks

        detections = []
        for res in output_results:
            x1, y1, w, h, score, cls_id = res
            detections.append(STrack([x1, y1, x1 + w, y1 + h], score, cls_id))

        remain_inds = [i for i, d in enumerate(detections) if d.score >= self.track_thresh]
        dets = [detections[i] for i in remain_inds]

        matches, u_track, u_det = self.linear_assignment(self.tracked_stracks, dets, self.match_thresh)

        for t_idx, d_idx in matches:
            self.tracked_stracks[t_idx].bbox = dets[d_idx].bbox
            self.tracked_stracks[t_idx].score = dets[d_idx].score

        for t_idx in sorted(u_track, reverse=True):
            self.tracked_stracks.pop(t_idx)

        for d_idx in u_det:
            new_track = dets[d_idx]
            new_track.track_id = self.next_id
            self.next_id += 1
            self.tracked_stracks.append(new_track)

        return self.tracked_stracks

    def linear_assignment(self, tracks, dets, thresh):
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))

        iou_matrix = np.zeros((len(tracks), len(dets)), dtype=np.float32)
        for t, track in enumerate(tracks):
            for d, det in enumerate(dets):
                iou_matrix[t, d] = bbox_iou(track.bbox, det.bbox)

        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        matches = []
        u_track = list(range(len(tracks)))
        u_det = list(range(len(dets)))
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= thresh:
                matches.append((r, c))
                u_track.remove(r)
                u_det.remove(c)

        return matches, u_track, u_det


def process_road_and_lanes(frame):
    h, w = frame.shape[:2]
    overlay = np.zeros_like(frame)

    gray_orig = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_gray = clahe.apply(gray_orig)

    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    channels = list(cv2.split(ycrcb))
    channels[0] = equalized_gray
    blur_frame = cv2.cvtColor(cv2.merge(channels), cv2.COLOR_YCrCb2BGR)
    blur = cv2.GaussianBlur(blur_frame, (5, 5), 0)

    edges_wall = cv2.Canny(equalized_gray, 40, 120)
    kernel_wall = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    wall_mask_dilated = cv2.dilate(edges_wall, kernel_wall, iterations=1)

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lower_road = np.array([0, 0, 50])
    upper_road = np.array([180, 50, 180])
    road_mask = cv2.inRange(hsv, lower_road, upper_road)
    road_mask = cv2.bitwise_and(road_mask, cv2.bitwise_not(wall_mask_dilated))

    roi_mask = np.zeros_like(road_mask)
    poly_points = np.array([
        [0, h],
        [int(w * 0.35), int(h * 0.55)],
        [int(w * 0.65), int(h * 0.55)],
        [w, h]
    ], np.int32)
    cv2.fillPoly(roi_mask, [poly_points], 255)

    dynamic_road = cv2.bitwise_and(road_mask, roi_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dynamic_road = cv2.morphologyEx(dynamic_road, cv2.MORPH_CLOSE, kernel)
    dynamic_road = cv2.morphologyEx(dynamic_road, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(dynamic_road, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_road_mask = np.zeros_like(dynamic_road)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > (h * w * 0.04):
            cv2.drawContours(final_road_mask, [largest_contour], -1, 255, -1)
    overlay[final_road_mask > 0] = [0, 255, 0]

    if ENABLE_LANE:
        edges = cv2.Canny(blur, 50, 150)
        lane_roi = cv2.bitwise_and(edges, roi_mask)
        lines = cv2.HoughLinesP(
            lane_roi, rho=1, theta=np.pi / 180,
            threshold=35, minLineLength=35, maxLineGap=25
        )
        if lines is not None:
            for line in lines[:30]:
                x1, y1, x2, y2 = line[0]
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                if dx > 0 and (dy / dx) > 0.7:
                    cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)

    return cv2.addWeighted(frame, 1.0, overlay, 0.25, 0)


fps_list = []


def generate_frames(cap, yolo_session, yolo_input_name, tracker):
    FOCAL_LENGTH = 500.0
    # 클래스별 세부 정보: {COCO class_id: (표시 이름, 실제 높이(m), 박스 색상 BGR)}
    CLASS_INFO = {
        0: ("Person", 1.7, (0, 0, 255)),
        1: ("Bicycle", 1.6, (255, 0, 255)),
        2: ("Car", 1.5, (255, 255, 0)),
        3: ("Motorcycle", 1.6, (0, 165, 255)),
        5: ("Bus", 3.0, (0, 255, 255)),
        7: ("Truck", 2.5, (255, 128, 0)),
    }

    frame_count = 0
    last_online_targets = []

    while True:
        start_time = time.time()
        ret, frame = cap.read()

        # 변경 (영상 끝나면 처음으로 되돌아가기)
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1
        w_orig, h_orig = frame.shape[1], frame.shape[0]
        raw_frame = frame.copy()
        display_frame = process_road_and_lanes(raw_frame)

        if frame_count % FRAME_SKIP == 0:
            # 레터박스 리사이즈: 종횡비를 유지한 채 정사각형 안에 맞추고 남는 자리는 회색 패딩으로 채움
            # (기존의 단순 정사각형 리사이즈는 객체를 찌그러뜨려 인식률을 떨어뜨림)
            scale = min(YOLO_SIZE / w_orig, YOLO_SIZE / h_orig)
            new_w, new_h = int(round(w_orig * scale)), int(round(h_orig * scale))
            resized = cv2.resize(raw_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            img_yolo = np.full((YOLO_SIZE, YOLO_SIZE, 3), 114, dtype=np.uint8)
            pad_x, pad_y = (YOLO_SIZE - new_w) // 2, (YOLO_SIZE - new_h) // 2
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

                if class_id in target_classes:
                    required_conf = 0.40 if class_id == 0 else 0.20
                    if confidence > required_conf:
                        cx, cy, w, h = pred[0:4]
                        # 패딩/스케일을 되돌려 원본 프레임 좌표로 변환
                        x1 = int((cx - w / 2 - pad_x) / scale)
                        y1 = int((cy - h / 2 - pad_y) / scale)
                        boxes.append([x1, y1,
                                      int(w / scale),
                                      int(h / scale)])
                        confidences.append(confidence)
                        class_ids.append(class_id)

            tracker_inputs = []
            if len(boxes) > 0:
                indices = cv2.dnn.NMSBoxes(boxes, confidences,
                                           score_threshold=CONF_THRESHOLD,
                                           nms_threshold=NMS_THRESHOLD)
                if len(indices) > 0:
                    for i in indices.flatten():
                        x, y, w, h = boxes[i]
                        if h < 20:
                            continue
                        tracker_inputs.append([x, y, w, h, confidences[i], class_ids[i]])

            last_online_targets = tracker.update(tracker_inputs)

        for target in last_online_targets:
            x1, y1, x2, y2 = map(int, target.bbox)
            h_box = y2 - y1
            cls_id = target.cls_id
            track_id = target.track_id

            class_name, real_h, color = CLASS_INFO.get(cls_id, ("Unknown", 1.5, (200, 200, 200)))
            distance_text = f"{(real_h * FOCAL_LENGTH) / h_box:.1f}m" if h_box > 0 else "Unknown"
            label = f"[{track_id}] {class_name} | {distance_text}"

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame, label, (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        elapsed_time = max(time.time() - start_time, 0.001)
        fps_list.append(1.0 / elapsed_time)
        if len(fps_list) > 30:
            fps_list.pop(0)
        fps = sum(fps_list) / len(fps_list)

        cv2.putText(display_frame, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(display_frame,
                    f"SIZE: {YOLO_SIZE} | Skip: {FRAME_SKIP} | Lane: {ENABLE_LANE}",
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # JPEG로 인코딩해서 스트리밍
        _, buffer = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


# =========================
# Flask 라우트
# =========================
cap = None
yolo_session = None
yolo_input_name = None
tracker = None


@app.route("/")
def index():
    return """
    <html>
    <head><title>PM ADAS - PC</title></head>
    <body style="background:#000; text-align:center;">
        <h2 style="color:white;">PM ADAS Live Stream</h2>
        <img src="/video_feed" width="640">
    </body>
    </html>
    """


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(cap, yolo_session, yolo_input_name, tracker),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def main():
    global cap, yolo_session, yolo_input_name, tracker

    print("1. 프로그램 시작됨...")
    yolo_onnx = "./yolov8n.onnx"
    cap = cv2.VideoCapture("input_video.mp4")

    if not cap.isOpened():
        print("❌ 에러: 카메라를 열 수 없습니다.")
        return
    print("2. 카메라 로드 성공!")

    # PC의 CPU 코어를 최대한 활용 (라즈베리파이 버전은 코어가 적어 스레드를 1~2개로 제한했었음)
    cpu_count = os.cpu_count() or 4
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = cpu_count
    opts.inter_op_num_threads = 2
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
        return

    yolo_input_name = yolo_session.get_inputs()[0].name
    tracker = SimpleByteTracker(track_thresh=0.45, match_thresh=0.45)

    # macOS는 5000번 포트를 AirPlay 수신 기능이 기본으로 점유하고 있어서 5001번 사용
    print("🚀 Flask 서버 시작! 브라우저에서 http://PC의IP:5001 접속 (같은 PC라면 http://localhost:5001)")
    app.run(host="0.0.0.0", port=5001, threaded=False)


if __name__ == "__main__":
    main()