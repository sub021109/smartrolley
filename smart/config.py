# config.py - 모든 설정 값을 모아둔 파일입니다.

import os

# --- 파일 및 폴더 경로 (어떤 위치에서 실행해도 경로가 꼬이지 않도록 설정) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "object_database.pkl")
IMAGE_DIR = os.path.join(BASE_DIR, "registered_images")
LOG_FILE = os.path.join(BASE_DIR, "inventory_log.csv")
YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'yolov5s.pt')
RESNET_MODEL_PATH = os.path.join(BASE_DIR, 'resnet18-f37072fd.pth')

# --- 인식 관련 임계값 ---
SIMILARITY_THRESHOLD = 0.8         # 일반 인식 임계값
SIMILARITY_THRESHOLD_RELAXED = 0.6 # 샘플 추가 시의 완화된 임계값
IOU_THRESHOLD = 0.9                # 동일 객체 판단을 위한 IOU 임계값

# --- 동작 관련 설정 ---
STATIC_TIMEOUT = 10                  # 고정 객체 무시까지의 시간 (초)
EVENT_THRESHOLD_SECONDS = 15         # 입/출고 판단 기준 시간 (초)
STABILITY_FRAME_THRESHOLD = 8        # 자동 등록을 위한 안정성 프레임 수
STABILITY_FRAME_THRESHOLD_FAST = 4   # 샘플 추가를 위한 빠른 안정성 프레임 수
BARCODE_RECHECK_INTERVAL = 0.5       # 바코드 재탐색 주기 (초)

# --- 상태 코드 ---
STATE_IDLE = 0
STATE_AWAITING_INITIAL_PLACEMENT = 1
STATE_AWAITING_SAMPLE_PLACEMENT = 2