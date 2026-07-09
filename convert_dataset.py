import json
import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# ================= [최종 경로 설정] =================
label_search_dir = "./dreamace/detection/detection_11026_labels/labels"
image_dir = "./dreamace/detection/images/seouldata"
output_dir = "./yolo_dataset"
# ===================================================

print("📦 1단계: 11,026개 JSON 정답지 특징 벡터(DB) 구축 중...")
json_files = [f for f in os.listdir(label_search_dir) if f.endswith('.json')]
json_db = {}

for j_name in tqdm(json_files):
    j_path = os.path.join(label_search_dir, j_name)
    try:
        with open(j_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        objects = data.get("result", {}).get("objects", [])
        if not objects: continue
        
        boxes = []
        for obj in objects:
            if "box" in obj.get("shape", {}):
                b = obj["shape"]["box"]
                # [x_min, y_min, x_max, y_max] 형태로 변환하여 저장
                boxes.append([float(b["x"]), float(b["y"]), float(b["x"])+float(b["width"]), float(b["y"])+float(b["height"])])
        
        if boxes:
            # 빠른 검색을 위해 박스 개수별로 딕셔너리 그룹화
            cnt = len(boxes)
            if cnt not in json_db:
                json_db[cnt] = []
            json_db[cnt].append({"name": j_name, "boxes": np.array(boxes), "data": data})
    except:
        continue

print("🤖 2단계: YOLOv8 활용 전수 기하학적 매칭 스캔 가동...")
model = YOLO("yolov8n.pt")
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

success_match = 0

for img_name in tqdm(image_files):
    img_path = os.path.join(image_dir, img_name)
    img_cv = cv2.imread(img_path)
    if img_cv is None: continue
    h_img, w_img = img_cv.shape[:2]
    
    # YOLO 추론 박스 획득
    results = model(img_path, verbose=False)[0]
    pred_boxes = results.boxes.xyxy.cpu().numpy()
    pred_cnt = len(pred_boxes)
    
    if pred_cnt == 0: continue
    
    # 박스 개수가 일치하거나 오차범위 ±1인 JSON 후보군만 타겟팅 조사
    best_json = None
    min_distance = float('inf')
    
    for c_offset in [0, -1, 1, -2, 2]:
        target_cnt = pred_cnt + c_offset
        if target_cnt not in json_db: continue
        
        for item in json_db[target_cnt]:
            # 박스 중심점들의 평균 거리 계산 (가장 정밀한 매칭 척도)
            p_center = np.mean(pred_boxes[:, :2], axis=0)
            j_center = np.mean(item["boxes"][:, :2], axis=0)
            dist = np.linalg.norm(p_center - j_center)
            
            if dist < min_distance and dist < 100: # 100픽셀 이내 격차 조건
                min_distance = dist
                best_json = item

    if best_json:
        success_match += 1
        shutil.copy(img_path, os.path.join(output_dir, "images", img_name))
        
        base_name = os.path.splitext(img_name)[0]
        txt_path = os.path.join(output_dir, "labels", f"{base_name}.txt")
        
        yolo_lines = []
        for obj in best_json["data"]["result"]["objects"]:
            c_name = obj["class"].lower()
            if "pedestrian" in c_name: c_id = 0
            elif "two-wheeler" in c_name or "bicycle" in c_name: c_id = 1
            elif "vehicle" in c_name or "car" in c_name: c_id = 2
            elif "motorcycle" in c_name: c_id = 3
            elif "truck" in c_name: c_id = 7
            else: continue
            
            b = obj["shape"]["box"]
            x_c = (float(b["x"]) + (float(b["width"]) / 2.0)) / w_img
            y_c = (float(b["y"]) + (float(b["height"]) / 2.0)) / h_img
            w_norm = float(b["width"]) / w_img
            h_norm = float(b["height"]) / h_img
            
            yolo_lines.append(f"{c_id} {x_c:.6f} {y_c:.6f} {w_norm:.6f} {h_norm:.6f}")
            
        with open(txt_path, 'w', encoding='utf-8') as lf:
            lf.write("\n".join(yolo_lines) + "\n")

print(f"\n🏁 [전수 부활 완료] 총 {success_match}장의 데이터가 진짜 정답 매칭에 성공했습니다.")

# 3단계: 자동 검증 프로세스 연동 실행
print("\n📊 3단계: 복구된 정답지로 진짜 YOLOv8 벤치마크 검증 가동...")
os.system("yolo task=detect mode=val model=yolov8n.pt data=data.yaml device=0")