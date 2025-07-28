# vision_utils.py - 모델 로딩, 이미지 처리, 객체/바코드 인식 등 핵심 비전 로직

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
from scipy.spatial.distance import cosine
from pyzbar.pyzbar import decode
import config

# --- 모듈 내 전역 변수로 모델과 전처리기 선언 ---
device = None
detection_model = None
feature_extractor = None
preprocess = None
clahe = None

def initialize_vision():
    """
    프로그램 시작 시 한 번만 호출되어 AI 모델과 비전 도구를 초기화하고,
    위에서 선언한 전역 변수들에 실제 모델을 할당합니다.
    """
    global device, detection_model, feature_extractor, preprocess, clahe
    
    print("--- 모델 로딩 시작 ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    try:
        detection_model = torch.hub.load('ultralytics/yolov5', 'custom', path=config.YOLO_MODEL_PATH)
        feature_extractor = models.resnet18(weights=None)
        feature_extractor.load_state_dict(torch.load(config.RESNET_MODEL_PATH))
        feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-1])
        detection_model.to(device).eval()
        feature_extractor.to(device).eval()
        print("모든 모델 로딩 완료.\n")
    except Exception as e:
        print(f"!!! 모델 로딩 오류: {e}. 프로그램을 종료합니다.")
        exit()
        
    preprocess = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def _decode_and_append(image, found_barcodes, found_data):
    """pyzbar 디코딩을 수행하고 결과를 리스트에 추가하는 내부 함수"""
    for barcode in decode(image):
        barcode_data_str = barcode.data.decode('utf-8')
        if barcode_data_str not in found_data:
            found_barcodes.append({'data': barcode_data_str, 'rect': barcode.rect, 'type': barcode.type})
            found_data.add(barcode_data_str)

def optimized_decode_barcodes(frame):
    """최적화된 방식으로 바코드를 디코딩합니다."""
    found_barcodes, found_data = [], set()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _decode_and_append(gray, found_barcodes, found_data)
    if clahe:  # clahe가 초기화되었는지 확인
        enhanced_gray = clahe.apply(gray)
        _decode_and_append(enhanced_gray, found_barcodes, found_data)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    _decode_and_append(adaptive_thresh, found_barcodes, found_data)
    return found_barcodes

def extract_features(image_crop_np):
    """이미지에서 특징 벡터를 추출합니다."""
    with torch.no_grad():
        image_pil = Image.fromarray(cv2.cvtColor(image_crop_np, cv2.COLOR_BGR2RGB))
        input_tensor = preprocess(image_pil).unsqueeze(0).to(device)
        return feature_extractor(input_tensor).squeeze().cpu().numpy()

def recognize_object(image_crop, db, relaxed_threshold=False):
    """DB와 비교하여 객체를 인식합니다."""
    threshold = config.SIMILARITY_THRESHOLD_RELAXED if relaxed_threshold else config.SIMILARITY_THRESHOLD
    if not db: return "Unknown", 0.0
    current_vector = extract_features(image_crop)
    best_match_name, highest_similarity = "Unknown", 0.0
    for name, data in db.items():
        if 'vectors' in data:
            for known_vector in data['vectors']:
                similarity = 1 - cosine(current_vector, known_vector)
                if similarity > highest_similarity:
                    highest_similarity, best_match_name = similarity, name
    if highest_similarity >= threshold: return best_match_name, highest_similarity
    else: return "Unknown", highest_similarity

def calculate_iou(boxA, boxB):
    """IoU 계산 함수"""
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea)