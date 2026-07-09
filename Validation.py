from ultralytics import YOLO
import os
import numpy as np

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def main():
    model_path = "C:/PM_ADAS_Workspace/YOLO/yolov8n.pt"
    model = YOLO(model_path)
    
    print("=== 순정 YOLOv8n 모델의 서울 주행 환경 제로샷 검증 시작 ===")
    
    image_dir = "C:/PM_ADAS_Workspace/YOLO/My First Project.yolov8/valid/images"
    label_dir = "C:/PM_ADAS_Workspace/YOLO/My First Project.yolov8/valid/labels"
    
    target_names = ['car', 'motorcycle', 'person', 'truck']
    coco_mapping = {0: 2, 1: 3, 2: 0, 3: 7}
    
    cls_tp_fp = {i: [] for i in range(4)}
    cls_scores = {i: [] for i in range(4)}
    cls_total_gt = {i: 0 for i in range(4)}
    
    img_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print("-> 순정 모델로 직접 추론 및 정확한 해상도 매칭 채점 진행 중...")
    for img_file in img_files:
        base_name = os.path.splitext(img_file)[0]
        lbl_file = base_name + ".txt"
        lbl_path = os.path.join(label_dir, lbl_file)
        
        if not os.path.exists(lbl_path):
            continue
            
        # 1. 순정 모델 예측을 먼저 수행하여 이미지의 진짜 원본 해상도를 얻습니다.
        img_path = os.path.join(image_dir, img_file)
        results = model.predict(source=img_path, imgsz=640, verbose=False)[0]
        
        # 원본 이미지의 진짜 세로(orig_h), 가로(orig_w) 픽셀 크기 추출
        orig_h, orig_w = results.orig_shape
        
        # 2. 원본 해상도를 기반으로 정답(Ground Truth) 박스를 픽셀 좌표로 정확히 복원
        gt_boxes = []
        with open(lbl_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if not parts or len(parts) < 5: 
                    continue
                
                cls_id = int(parts[0])
                x, y, w, h = map(float, parts[1:5]) 
                
                # 정규화된 좌표(0~1)를 진짜 이미지 픽셀 크기에 곱해줍니다.
                x1 = (x - w/2) * orig_w
                y1 = (y - h/2) * orig_h
                x2 = (x + w/2) * orig_w
                y2 = (y + h/2) * orig_h
                
                if cls_id in cls_total_gt:
                    gt_boxes.append({'cls': cls_id, 'box': [x1, y1, x2, y2], 'matched': False})
                    cls_total_gt[cls_id] += 1
                
        # 3. 모델의 예측 박스 로드
        pred_boxes = []
        if results.boxes is not None:
            for box in results.boxes:
                pred_coco_id = int(box.cls[0].item())
                score = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist() # 원본 이미지 해상도 기준 픽셀 좌표 [xmin, ymin, xmax, ymax]
                
                pred_my_id = -1
                for my_id, coco_id in coco_mapping.items():
                    if coco_id == pred_coco_id:
                        pred_my_id = my_id
                        break
                
                if pred_my_id != -1:
                    pred_boxes.append({'cls': pred_my_id, 'box': xyxy, 'score': score})
                    
        # 신뢰도 높은 순서로 정렬
        pred_boxes = sorted(pred_boxes, key=lambda k: k['score'], reverse=True)
        
        # 4. 픽셀 단위가 일치하는 상태에서 IoU 매칭 매핑
        for pred in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            
            for idx, gt in enumerate(gt_boxes):
                if gt['cls'] == pred['cls'] and not gt['matched']:
                    iou = calculate_iou(pred['box'], gt['box'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx
                        
            if best_iou >= 0.5 and best_gt_idx != -1:
                gt_boxes[best_gt_idx]['matched'] = True
                cls_tp_fp[pred['cls']].append(1)
            else:
                cls_tp_fp[pred['cls']].append(0)
            cls_scores[pred['cls']].append(pred['score'])

    print("\n" + "="*40)
    print("   순정 YOLOv8n 모델의 서울 환경 검증 결과   ")
    print("="*40)
    
    total_ap = 0
    valid_classes = 0
    
    for i in range(4):
        tps = cls_tp_fp[i]
        scores = cls_scores[i]
        total_gts = cls_total_gt[i]
        
        if total_gts == 0:
            print(f" - {target_names[i]} mAP50 성능: 데이터 없음")
            continue
            
        if len(tps) == 0:
            print(f" - {target_names[i]} mAP50 성능: 0.0000 (검출 실패)")
            valid_classes += 1
            continue
            
        sort_indices = np.argsort(scores)[::-1]
        tps = np.array(tps)[sort_indices]
        
        tp_cumsum = np.cumsum(tps)
        fp_cumsum = np.cumsum(1 - tps)
        
        recalls = tp_cumsum / total_gts
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            prec_at_t = precisions[recalls >= t]
            if len(prec_at_t) > 0:
                ap += np.max(prec_at_t)
        ap /= 11
        
        print(f" - {target_names[i]} mAP50 성능: {ap:.4f}")
        total_ap += ap
        valid_classes += 1
        
    mean_ap = total_ap / valid_classes if valid_classes > 0 else 0
    print("-"*40)
    print(f"종합 mAP50 성능 지표: {mean_ap:.4f}")
    print("="*40)
    print("결론: 추가 학습 없이 기존 가중치 그대로 서울 환경을 평가한 순수 제로샷 수치입니다.")

if __name__ == "__main__":
    main()