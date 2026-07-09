import cv2
from ultralytics import YOLO
from flask import Flask, Response

app = Flask(__name__)
model = YOLO("yolov8n.onnx")
# 업로드하신 input_video.mp4 경로 지정
video_path = "input_video.mp4" 

def generate_frames():
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            # 영상이 끝나면 처음부터 다시 재생 (무한 루프)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # YOLO 모델로 객체 인식 및 시각화
        results = model.track(frame, persist=True)
        annotated_frame = results[0].plot()

        # 프레임을 JPEG로 인코딩하여 웹으로 전송
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # 5000번 포트로 웹 서버 구동
    app.run(host='0.0.0.0', port=5000, threaded=True)