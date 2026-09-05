# 🛴 PM 전용 스마트 ADAS 시스템

## **💡1. 프로젝트 개요**

### **1-1. 프로젝트 소개**

- **프로젝트 명** : PM 전용 스마트 ADAS 시스템
- **프로젝트 정의** : Raspberry Pi 5와 Camera Module 3를 기반으로 전방 객체를 실시간 탐지하고, 객체와의 거리·접근속도·TTC(Time To Collision) 및 예상 진행 경로를 종합하여 충돌 위험을 사전에 판단하는 개인형 이동장치(PM) 전용 ADAS 시스템

<p align="center">
  <img src="./images/project_main.png" width="700">
</p>

### **1-2. 개발 배경 및 필요성**

개인형 이동장치(PM)는 간편한 이동수단으로 활용되고 있으나, 자동차와 달리 운전자의 안전운전을 지원하는 ADAS(Advanced Driver Assistance System)가 충분히 적용되지 않고 있다.

특히 보행자, 자전거, 차량 등이 혼재하는 도심 주행 환경에서는 전방 객체와의 거리뿐만 아니라 객체가 실제 PM의 진행 경로에 위치하는지, 얼마나 빠르게 가까워지고 있는지를 종합적으로 판단할 필요가 있다.

이에 본 프로젝트에서는 Raspberry Pi 5와 Camera Module 3를 활용하여 전방 객체를 실시간으로 인식하고, 거리·접근속도·TTC·Collision Zone을 기반으로 충돌 위험을 판단하여 사용자에게 실시간 경고를 제공하는 PM 전용 스마트 ADAS 시스템을 구현한다.


### **1-3. 프로젝트 특장점**

- **PM 환경에 특화된 ADAS** : 자동차 중심의 기존 ADAS와 달리 개인형 이동장치의 주행 환경을 고려한 위험 판단
- **실시간 객체 탐지** : YOLO 기반으로 보행자, 자전거, 자동차, 오토바이 등 전방 객체를 실시간 인식
- **다중 요소 기반 위험 판단** : 단순 거리뿐만 아니라 접근속도, TTC, Collision Zone을 함께 활용
- **Edge AI 기반 처리** : Raspberry Pi 5에서 객체 탐지와 위험 판단을 수행하여 실시간성 확보
- **주행 데이터 기록** : GPS 및 IMU 센서를 활용하여 주행 경로와 급정거·급가속·전도 등의 주행 이벤트 기록
- **실시간 안전 경고** : 위험 상황 발생 시 화면 및 Buzzer를 통해 사용자에게 즉시 경고


### **1-4. 주요 기능**

#### ① 전방 객체 실시간 탐지
Camera Module 3에서 입력되는 영상을 기반으로 YOLO 모델을 이용하여 보행자, 자전거, 차량 등의 객체를 실시간 탐지한다.

#### ② 객체 추적
동일 객체에 ID를 부여하여 프레임 간 객체의 위치 및 거리 변화를 지속적으로 추적한다.

#### ③ 객체 거리 추정
객체의 Bounding Box와 실제 객체의 평균 높이를 기반으로 PM과 객체 사이의 거리를 추정한다.

#### ④ 접근속도 및 TTC 계산
동일 객체의 거리 변화를 기반으로 상대 접근속도를 계산하고, TTC(Time To Collision)를 통해 예상 충돌 시간을 계산한다.

#### ⑤ Collision Zone 기반 위험 판단
PM의 예상 진행 경로를 Collision Zone으로 설정하고 객체가 실제 진행 경로에 위치하는지를 판단한다.

#### ⑥ 실시간 위험 알림
거리, 접근속도, TTC 및 Collision Zone을 종합하여 위험도를 SAFE / CAUTION / WARNING / DANGER 단계로 구분하고 위험 상황 발생 시 사용자에게 실시간 경고를 제공한다.

#### ⑦ 주행 데이터 기록 및 확인
GPS와 IMU 센서를 활용하여 주행 경로 및 급정거·급가속·전도 등의 이벤트를 기록하고 모바일 웹에서 확인할 수 있도록 한다.


### **1-5. 기대 효과 및 활용 분야**

- PM 이용자의 전방 충돌 위험 사전 인지
- 보행자 및 자전거 등 주변 도로 이용자와의 충돌 사고 예방
- 저비용 Edge AI 기반 PM 안전 보조 시스템 구현
- GPS·IMU 기반 주행 데이터 분석을 통한 안전운전 지원
- 향후 공유 킥보드 및 자전거 등 다양한 마이크로모빌리티 서비스로 확장 가능


### **1-6. 기술 스택**

| 구분 | 기술 |
|---|---|
| **AI / Object Detection** | YOLOv8n, ONNX |
| **Computer Vision** | OpenCV |
| **AI Runtime** | ONNX Runtime |
| **Edge Device** | Raspberry Pi 5 |
| **Camera** | Raspberry Pi Camera Module 3 |
| **Sensor** | IMU, GPS |
| **Backend / Streaming** | Python, Flask |
| **Frontend** | [수정 필요] |
| **Data** | GPS / IMU / ADAS 위험 이벤트 |
| **Version Control** | Git, GitHub |

---

## **💡2. 팀원 소개**

| 이름 | 역할 | 담당 업무 |
|:---:|:---:|:---|
| **곽연우** | AI / H/W |  객체 추적·거리 추정 및 충돌 위험 판단, H/W 연결 담당  |
| **구지원** | AI / Web | YOLO 기반 객체 인식 및 ADAS 파이프라인 구현, 모바일 웹 개발 |
| **박지민** | H/W | Raspberry Pi 환경 구축, Camera, GPS, IMU 센서 연동 |
| **서아연** | S/W | 주행 데이터 처리 및 위험 이벤트 기록 |

---

## **💡3. 시스템 구성도**

### **3-1. 시스템 아키텍처**

<p align="center">
  <img src="./images/system_architecture.png" width="750">
</p>

Camera Module 3, GPS 및 IMU 센서에서 수집한 데이터를 Raspberry Pi 5에서 처리한다.  
ADAS Engine은 객체 탐지 → 객체 추적 → 거리 및 접근속도 계산 → TTC 및 Collision Zone 판단을 수행하고 최종 위험도를 결정한다.

---

### **3-2. ADAS 위험 판단 파이프라인**

<p align="center">
  <img src="./images/adas_pipeline.png" width="700">
</p>

**위험 판단 과정**

`YOLO 객체 탐지 → Object Tracking → 거리 추정 → 접근속도 계산 → TTC 계산 → Collision Zone 판단 → 최종 위험도 판단`

최종 위험도는 다음 네 단계로 구분한다.

| 위험도 | 의미 |
|:---:|---|
| 🟢 **SAFE** | 정상 주행 |
| 🟡 **CAUTION** | 주의 필요 |
| 🟠 **WARNING** | 충돌 가능성 증가 |
| 🔴 **DANGER** | 즉각적인 충돌 위험 |

---

### **3-3. 서비스 시나리오**

<p align="center">
  <img src="./images/service_scenario.png" width="700">
</p>

1. 사용자가 모바일 웹에서 주행을 시작한다.
2. Camera Module 3가 전방 주행 영상을 실시간 입력한다.
3. YOLO 기반 객체 탐지 및 Tracking을 수행한다.
4. 거리·접근속도·TTC·Collision Zone을 기반으로 위험도를 판단한다.
5. GPS 및 IMU를 통해 주행 위치와 주행 이벤트를 기록한다.
6. 위험 상황 발생 시 사용자에게 실시간 경고를 제공한다.
7. 주행 종료 후 모바일 웹에서 주행 경로와 안전 정보를 확인한다.

---

### **3-4. 하드웨어 / 센서 구성도**

<p align="center">
  <img src="./images/hardware_architecture.png" width="750">
</p>

| 구성 요소 | 역할 |
|---|---|
| **Raspberry Pi 5** | 센서 데이터 수집 및 ADAS 처리 |
| **Camera Module 3** | 전방 영상 촬영 및 객체 탐지 입력 |
| **IMU 센서** | 가속도·자이로 측정 및 급정거·급가속·전도 감지 |
| **GPS 모듈** | 위치·속도 측정 및 주행 경로 기록 |
| **Buzzer** | 위험 상황 실시간 경고 |
| **Active Cooler** | 장시간 주행 시 Raspberry Pi 발열 관리 |
| **Power Bank** | 시스템 전원 공급 |

---

## **💡4. 작품 소개영상**

<p align="center">

[![PM 전용 스마트 ADAS 시스템 소개 영상](./images/video_thumbnail.png)]([유튜브 영상 URL])

</p>

> Camera Module 3를 활용한 실시간 객체 탐지, 위험도 판단 및 모바일 웹 기반 주행 기록 서비스를 확인할 수 있습니다.

---

## **💡5. 핵심 소스코드**

### **5-1. YOLO 기반 실시간 객체 탐지**

```python
yolo_outputs = yolo_session.run(
    None,
    {yolo_input_name: yolo_tensor}
)

predictions = np.squeeze(yolo_outputs[0]).T
