import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import cv2
import numpy as np
import onnxruntime as ort
import time
from scipy.optimize import linear_sum_assignment

# ==========================================
# [개선] 잔상 없는 초경량 ByteTrack
# ==========================================
def bbox_iou(box1, box2):
    """ 두 바운딩 박스 간의 IoU(교집합/합집합 비율) 계산 """
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
        self.bbox = np.array(bbox)  # [x1, y1, x2, y2]
        self.score = score
        self.cls_id = cls_id
        self.track_id = 0

class SimpleByteTracker:
    def __init__(self, track_thresh=0.45, match_thresh=0.45):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.tracked_stracks = []  # 유지되고 있는 추적 객체 리스트
        self.next_id = 1

    def update(self, output_results):
        """ output_results: [[x1, y1, w, h, score, cls_id], ...] """
        # 매칭 실패 대상을 추적하기 위한 배열 생성
        if len(output_results) == 0:
            self.tracked_stracks.clear()  # 검출된 게 없으면 트래커도 즉시 비움 (허공 잔상 제거)
            return self.tracked_stracks

        detections = []
        for res in output_results:
            x1, y1, w, h, score, cls_id = res
            detections.append(STrack([x1, y1, x1 + w, y1 + h], score, cls_id))

        # 1차 매칭 대상 설정 (트래커 문턱값 이상)
        remain_inds = [i for i, d in enumerate(detections) if d.score >= self.track_thresh]
        dets = [detections[i] for i in remain_inds]
        
        # 선형 할당(Hungarian Algorithm) 매칭
        matches, u_track, u_det = self.linear_assignment(self.tracked_stracks, dets, self.match_thresh)
        
        # 매칭 성공한 기존 추적 객체 업데이트
        for t_idx, d_idx in matches:
            self.tracked_stracks[t_idx].bbox = dets[d_idx].bbox
            self.tracked_stracks[t_idx].score = dets[d_idx].score

        # 💡 [핵심 수정] 매칭에 실패한(YOLO가 이번에 안 잡아준) 기존 트랙은 즉시 삭제
        # 역순으로 pop해야 인덱스 꼬임이 없습니다.
        for t_idx in sorted(u_track, reverse=True):
            self.tracked_stracks.pop(t_idx)

        # 매칭에 실패한 새로운 탐지 객체에 고유 ID 부여 및 추적 추가
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
    """ 영상 테스트 최적화형 주행구역 탐지 함수 (CPU 경량화 유지) """
    h, w = frame.shape[:2]
    overlay = np.zeros_like(frame)

    gray_orig = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_gray = clahe.apply(gray_orig)
    
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    channels = list(cv2.split(ycrcb))
    channels[0] = equalized_gray
    blur_frame = cv2.cvtColor(cv2.merge(channels), cv2.COLOR_YCrCb2BGR)
    blur = cv2.GaussianBlur(blur_frame, (11, 11), 0)

    edges_wall = cv2.Canny(equalized_gray, 40, 120)
    kernel_wall = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
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

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dynamic_road = cv2.morphologyEx(dynamic_road, cv2.MORPH_CLOSE, kernel)
    dynamic_road = cv2.morphologyEx(dynamic_road, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(dynamic_road, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_road_mask = np.zeros_like(dynamic_road)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > (h * w * 0.04):
            cv2.drawContours(final_road_mask, [largest_contour], -1, 255, -1)

    overlay[final_road_mask > 0] = [0, 255, 0]

    edges = cv2.Canny(blur, 50, 150)
    lane_roi = cv2.bitwise_and(edges, roi_mask)

    lines = cv2.HoughLinesP(
        lane_roi, 
        rho=1, 
        theta=np.pi/180, 
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
                cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)

    return cv2.addWeighted(frame, 1.0, overlay, 0.25, 0)


def main():
    print("1. 프로그램 시작됨...")
    video_path = "./input_video.mp4"
    yolo_onnx = "./yolov8n.onnx"
    
    print(f"2. 영상 파일 존재 여부 체크: {os.path.exists(video_path)}")
    print(f"3. ONNX 파일 존재 여부 체크: {os.path.exists(yolo_onnx)}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 에러: 동영상 파일(input_video.mp4)을 오픈할 수 없습니다.")
        return
    print("4. 동영상 파일 로드 성공!")

    print("5. ONNX Runtime 세션 빌드 시작...")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4  
    
    try:
        yolo_session = ort.InferenceSession(yolo_onnx, sess_options=opts, providers=['CPUExecutionProvider'])
        print("6. ONNX Runtime 세션 빌드 완료!")
    except Exception as e:
        print(f"❌ 에러: ONNX 모델 로드 중 예외 발생: {e}")
        return
        
    yolo_input_name = yolo_session.get_inputs()[0].name
    print("🚀 7. 메인 루프 진입 성공! 영상을 실시간 처리합니다.")
    
    # 💡 [수정] 트래커 진입 기준 신뢰도를 0.45로 대폭 상향 조정
    tracker = SimpleByteTracker(track_thresh=0.45, match_thresh=0.45)
    
    FOCAL_LENGTH = 500.0  
    REAL_HEIGHTS = {
        "Person": 1.7,
        "Rider": 1.6,
        "Vehicle": 1.5
    }

    frame_count = 0

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        
        if not ret:
            if frame_count == 0:
                print("❌ 첫 프레임 읽기 실패로 종료합니다.")
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"📊 현재 {frame_count}번째 프레임 처리 중...")
            
        w_orig, h_orig = frame.shape[1], frame.shape[0]

        frame = process_road_and_lanes(frame)

        img_yolo = cv2.resize(frame, (320, 320))
        img_yolo = cv2.cvtColor(img_yolo, cv2.COLOR_BGR2RGB)
        img_yolo = img_yolo.astype(np.float32) / 255.0
        img_yolo = img_yolo.transpose(2, 0, 1)
        yolo_tensor = np.expand_dims(img_yolo, axis=0)
        
        yolo_outputs = yolo_session.run(None, {yolo_input_name: yolo_tensor})
        predictions = np.squeeze(yolo_outputs[0]).T
        
        target_classes = {0, 1, 2, 3, 5, 7}
        boxes, confidences, class_ids = [], [], []

        for pred in predictions:
            scores = pred[4:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if class_id in target_classes:
                # 💡 [핵심 수정] 사람은 오탐지가 심하므로 점수 문턱값(Conf)을 0.50으로 강화
                # 다른 이동수단은 기존처럼 0.25 유지
                required_conf = 0.50 if class_id == 0 else 0.25
                
                if confidence > required_conf:
                    cx, cy, w, h = pred[0:4]
                    x1 = int((cx - w / 2) * (w_orig / 320))
                    y1 = int((cy - h / 2) * (h_orig / 320))
                    
                    boxes.append([x1, y1, int(w * (w_orig / 320)), int(h * (h_orig / 320))])
                    confidences.append(float(confidence))
                    class_ids.append(int(class_id))

        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.25, nms_threshold=0.45)
        
        tracker_inputs = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                conf = confidences[i]
                cls_id = class_ids[i]
                
                # 픽셀 크기 필터벽 유지 (해상도 대비 너무 작으면 탈락)
                if h < 30:
                    continue
                
                tracker_inputs.append([x, y, w, h, conf, cls_id])
        
        # 트래커 업데이트
        online_targets = tracker.update(tracker_inputs)
        
        # 추적 객체 시각화 및 거리 측정
        for target in online_targets:
            x1, y1, x2, y2 = map(int, target.bbox)
            w_box = x2 - x1
            h_box = y2 - y1
            cls_id = target.cls_id
            track_id = target.track_id
            
            if cls_id == 0: 
                class_name = "Person"
                color = (0, 0, 255)
            elif cls_id in [1, 3]: 
                class_name = "Rider"
                color = (255, 0, 255)
            else: 
                class_name = "Vehicle"
                color = (255, 255, 0)
                
            real_h = REAL_HEIGHTS.get(class_name, 1.5)
            if h_box > 0:
                distance = (real_h * FOCAL_LENGTH) / h_box
                distance_text = f"{distance:.1f}m"
            else:
                distance_text = "Unknown"
                
            label = f"[{track_id}] {class_name} | {distance_text}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        elapsed_time = time.time() - start_time
        if elapsed_time == 0:
            elapsed_time = 0.001
        fps = 1.0 / elapsed_time
        
        cv2.putText(frame, f"Total FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("PM ADAS Hybrid Engine (ByteTrack)", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            print("👋 사용자가 'q'를 눌러 안전 종료합니다.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🏁 프로그램이 정상 종료되었습니다.")


if __name__ == "__main__":
    main()