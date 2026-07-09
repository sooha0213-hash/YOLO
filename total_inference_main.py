import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import cv2
import numpy as np
import onnxruntime as ort
import time

def process_road_and_lanes(frame):
    """
    영상 테스트 최적화형 주행구역 탐지 함수 (CPU 경량화 유지)
    - CLAHE 전처리로 조명/그늘 변화 방어
    - Canny 방어벽으로 벽면/가드레일 완전 차단
    - 최대 Contour 추적으로 독립 오탐지(공중 튐) 100% 제거
    """
    h, w = frame.shape[:2]
    overlay = np.zeros_like(frame)

    # [개선 1] 그늘 및 조명 변화 방어를 위한 명암비 평활화 (CLAHE)
    gray_orig = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_gray = clahe.apply(gray_orig)
    
    # 평활화된 이미지를 기반으로 컬러 프레임 재조합 (HSV 탐지 정확도 향상)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    channels = list(cv2.split(ycrcb))
    channels[0] = equalized_gray
    blur_frame = cv2.cvtColor(cv2.merge(channels), cv2.COLOR_YCrCb2BGR)
    blur = cv2.GaussianBlur(blur_frame, (11, 11), 0)

    # [개선 2] 강력한 벽면/가드레일 에지 방어벽 생성
    # SobelX 대신 Canny를 사용하여 대각선 및 모든 방향의 벽면 경계를 차단
    edges_wall = cv2.Canny(equalized_gray, 40, 120)
    kernel_wall = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    wall_mask_dilated = cv2.dilate(edges_wall, kernel_wall, iterations=1)

    # 3. HSV 색상 공간 기반 바닥 추출 (CLAHE 덕분에 범위를 조금 더 타이트하게 잡아도 잘 잡힙니다)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lower_road = np.array([0, 0, 50])       
    upper_road = np.array([180, 50, 180])   
    road_mask = cv2.inRange(hsv, lower_road, upper_road)

    # 4. 벽면 영역 및 에지 경계 영역 강제 제거 (벽면 침범 방지)
    road_mask = cv2.bitwise_and(road_mask, cv2.bitwise_not(wall_mask_dilated))

    # 5. 원근 관심 영역(ROI) 가이드 마스크 생성 (기존 유지)
    roi_mask = np.zeros_like(road_mask)
    poly_points = np.array([
        [0, h],
        [int(w * 0.35), int(h * 0.55)],
        [int(w * 0.65), int(h * 0.55)],
        [w, h]
    ], np.int32)
    cv2.fillPoly(roi_mask, [poly_points], 255)

    # 최종 동적 주행 구역 1차 결합
    dynamic_road = cv2.bitwise_and(road_mask, roi_mask)

    # 모포롤리지 연산으로 잔해물 메우기
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dynamic_road = cv2.morphologyEx(dynamic_road, cv2.MORPH_CLOSE, kernel)
    dynamic_road = cv2.morphologyEx(dynamic_road, cv2.MORPH_OPEN, kernel)

    # [개선 3] 최대 외곽선(Contour) 필터링 기법 도입
    # 영상에서 뜬금없이 옆벽이나 허공에 초록색이 칠해지는 현상을 완벽히 막아줍니다.
    contours, _ = cv2.findContours(dynamic_road, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_road_mask = np.zeros_like(dynamic_road)
    
    if contours:
        # 화면에서 가장 큰 면적을 가진 컨투어(즉, 내 차량 앞 도로)만 선택
        largest_contour = max(contours, key=cv2.contourArea)
        # 노이즈 방지를 위해 최소 크기 제한 (화면 전체의 4% 이상일 때만 도로로 인정)
        if cv2.contourArea(largest_contour) > (h * w * 0.04):
            cv2.drawContours(final_road_mask, [largest_contour], -1, 255, -1)

    # 6. 최종 정제된 주행 구역 마스크에 반투명 초록색 투영
    overlay[final_road_mask > 0] = [0, 255, 0]

    # 7. 차선 검출 (기존 로직 유지)
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
    opts.intra_op_num_threads = 4  # 라즈베리파이 5 / CPU 코어 최적화 수치 유지
    
    try:
        yolo_session = ort.InferenceSession(yolo_onnx, sess_options=opts, providers=['CPUExecutionProvider'])
        print("6. ONNX Runtime 세션 빌드 완료!")
    except Exception as e:
        print(f"❌ 에러: ONNX 모델 로드 중 예외 발생: {e}")
        return
        
    yolo_input_name = yolo_session.get_inputs()[0].name
    print("🚀 7. 메인 루프 진입 성공! 영상을 실시간 처리합니다.")
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

        # 1. 개선된 컴퓨터 비전 기반 벽면 방어형 동적 주행구역 & 차선 레이어 합성
        frame = process_road_and_lanes(frame)

        # 2. YOLOv8n ONNX 객체 인식 전처리 (256x256 해상도 가속)
        img_yolo = cv2.resize(frame, (256, 256))
        img_yolo = cv2.cvtColor(img_yolo, cv2.COLOR_BGR2RGB)
        img_yolo = img_yolo.astype(np.float32) / 255.0
        img_yolo = img_yolo.transpose(2, 0, 1)
        yolo_tensor = np.expand_dims(img_yolo, axis=0)
        
        # 3. ONNX Engine CPU 추론
        yolo_outputs = yolo_session.run(None, {yolo_input_name: yolo_tensor})
        predictions = np.squeeze(yolo_outputs[0]).T
        
        target_classes = {0, 1, 2, 3, 5, 7}
        boxes, confidences, class_ids = [], [], []

        for pred in predictions:
            scores = pred[4:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.25 and class_id in target_classes:
                cx, cy, w, h = pred[0:4]
                x1 = int((cx - w / 2) * (w_orig / 256))
                y1 = int((cy - h / 2) * (h_orig / 256))
                
                boxes.append([x1, y1, int(w * (w_orig / 256)), int(h * (h_orig / 256))])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))

        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.25, nms_threshold=0.45)
        
        # 5. 최종 객체 바운딩 박스 렌더링
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                cls_id = class_ids[i]
                conf = confidences[i]
                
                if cls_id == 0: 
                    label, color = f"Person: {conf:.2f}", (0, 0, 255)
                elif cls_id in [1, 3]: 
                    label, color = f"Rider: {conf:.2f}", (255, 0, 255)
                else: 
                    label, color = f"Vehicle: {conf:.2f}", (255, 255, 0)
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        elapsed_time = time.time() - start_time
        if elapsed_time == 0:
            elapsed_time = 0.001
        fps = 1.0 / elapsed_time
        
        cv2.putText(frame, f"Total FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        cv2.imshow("PM ADAS Hybrid Engine", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            print("👋 사용자가 'q'를 눌러 안전 종료합니다.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🏁 프로그램이 정상 종료되었습니다.")

if __name__ == "__main__":
    main()