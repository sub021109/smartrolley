# ===================================================================
#      Custom Recognizer v10 (재고 입출고 로깅 기능)
# ===================================================================

# ----------------- 1. 라이브러리 및 환경 설정 -----------------
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
from scipy.spatial.distance import cosine
import os
from pyzbar.pyzbar import decode
import time
from datetime import datetime # 시간 기록을 위해
import tkinter
from tkinter import simpledialog
import pickle

# ----------------- 2. 모델 로딩 및 설정 -----------------
# (이전 코드와 동일)
print("--- 모델 로딩 시작 ---")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 디바이스: {device}")
try:
    detection_model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')
    detection_model.to(device).eval()
    print("YOLOv5 모델 로딩 완료.")
    feature_extractor = models.resnet18(weights=None)
    feature_extractor.load_state_dict(torch.load('resnet18-f37072fd.pth'))
    feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-1])
    feature_extractor.to(device).eval()
    print("특징 추출기 로딩 완료.\n")
except Exception as e:
    print(f"!!! 모델 로딩 오류: {e}. 필요한 파일이 모두 있는지 확인하세요.")
    exit()

# ----------------- 3. 데이터베이스 및 함수 정의 -----------------
preprocess = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

DB_FILE = "object_database.pkl"
IMAGE_DIR = "registered_images"
if not os.path.exists(IMAGE_DIR): os.makedirs(IMAGE_DIR)

known_objects_db, known_barcodes = {}, set()
try:
    with open(DB_FILE, 'rb') as f: known_objects_db = pickle.load(f)
    for name, data in known_objects_db.items():
        if 'barcodes' in data:
            for bc in data['barcodes']: known_barcodes.add(bc)
    print(f"✅ 데이터베이스 로딩 완료. {len(known_objects_db)}개의 객체, {len(known_barcodes)}개의 바코드가 로드되었습니다.")
except FileNotFoundError: print("ℹ️ 저장된 데이터베이스 파일이 없습니다.")
except Exception as e: print(f"⚠️ 데이터베이스 로딩 중 오류 발생: {e}")

SIMILARITY_THRESHOLD = 0.75
object_tracker = {}
STATIC_TIMEOUT = 10
IOU_THRESHOLD = 0.9
all_scan_mode = False

# --- [새로운 기능] 재고 관리 로직 설정 ---
LOG_FILE = "inventory_log.csv"
EVENT_THRESHOLD_SECONDS = 15 # 15초를 기준으로 입출고 판단
inventory_tracker = {} # 재고 상태 추적기

# --- [새로운 기능] 로그 기록 함수 ---
def log_event(object_name, event_type):
    """CSV 파일에 재고 이벤트를 기록하는 함수"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp},{object_name},{event_type}\n"
    
    # 파일이 없으면 헤더를 추가
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("Timestamp,ObjectName,Event\n")

    # 이벤트 기록 추가
    with open(LOG_FILE, "a") as f:
        f.write(log_entry)
    print(f"✍️ LOG: {object_name} - {event_type}")

# --- (기존 함수들은 이전과 동일, 생략 없이 포함) ---
def get_name_from_popup(title, prompt):
    root = tkinter.Tk(); root.withdraw()
    user_input = simpledialog.askstring(title=title, prompt=prompt)
    root.destroy()
    return user_input

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea)

def extract_features(image_crop_np):
    with torch.no_grad():
        image_pil = Image.fromarray(cv2.cvtColor(image_crop_np, cv2.COLOR_BGR2RGB))
        input_tensor = preprocess(image_pil).unsqueeze(0).to(device)
        return feature_extractor(input_tensor).squeeze().cpu().numpy()

def register_new_object(image_crop, object_name, barcode_data=None):
    feature_vector = extract_features(image_crop)
    timestamp = int(time.time())
    image_filename = f"{object_name.replace(' ', '_')}_{timestamp}.jpg"
    image_path = os.path.join(IMAGE_DIR, image_filename)
    cv2.imwrite(image_path, image_crop)
    if object_name not in known_objects_db:
        known_objects_db[object_name] = {'vectors': [], 'images': [], 'barcodes': []}
    known_objects_db[object_name]['vectors'].append(feature_vector)
    known_objects_db[object_name]['images'].append(image_path)
    if barcode_data and barcode_data not in known_objects_db[object_name]['barcodes']:
        known_objects_db[object_name]['barcodes'].append(barcode_data)
        print(f"✅ '{object_name}'에 바코드 '{barcode_data}'를 연결했습니다.")
    print(f"✅ '{object_name}' 등록 완료!")

def recognize_object(image_crop):
    if not known_objects_db: return "Unknown", 0.0
    current_vector = extract_features(image_crop)
    best_match_name, highest_similarity = "Unknown", 0.0
    for name, data in known_objects_db.items():
        if 'vectors' in data:
            for known_vector in data['vectors']:
                similarity = 1 - cosine(current_vector, known_vector)
                if similarity > highest_similarity:
                    highest_similarity, best_match_name = similarity, name
    if highest_similarity >= SIMILARITY_THRESHOLD:
        return best_match_name, highest_similarity
    else:
        return "Unknown", highest_similarity

def enhanced_decode_barcodes(frame):
    barcodes_found = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    barcodes = decode(enhanced_gray)
    for barcode in barcodes:
        barcodes_found.append({'data': barcode.data.decode("utf-8"), 'rect': barcode.rect, 'type': barcode.type})
    return barcodes_found

# ----------------- 4. 메인 실행 루프 -----------------
cap = cv2.VideoCapture(1)
if not cap.isOpened(): print("!!! 오류: 웹캠을 열 수 없습니다."); exit()

print("\n--- 프로그램 시작 ---")
print(" 'r':수동등록 | 'e':이름수정 | 's':이미지보기 | 'a':모드전환 | 'q':종료")

while True:
    ret, frame = cap.read()
    if not ret: break

    current_time = time.time()
    
    # --- [수정] 메인 로직 순서 변경 ---
    # 1. 현재 프레임에서 보이는 모든 객체/바코드 식별
    detected_barcodes = enhanced_decode_barcodes(frame)
    results = detection_model(frame)
    detections = results.xyxy[0]
    
    current_frame_objects = set() # 현재 프레임에서 인식된 객체 이름 집합
    
    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        object_crop = frame[y1:y2, x1:x2]
        if object_crop.size == 0: continue
        name, score = recognize_object(object_crop)
        if name != "Unknown":
            current_frame_objects.add(name)

    # 2. 재고 상태 업데이트 및 이벤트 로깅
    # Step A: 현재 보이는 객체 처리 (입고 이벤트)
    for name in current_frame_objects:
        if name not in inventory_tracker: # 처음 보는 객체
            inventory_tracker[name] = {'status': 'IN', 'entry_time': current_time, 'last_seen': current_time}
            log_event(name, "IN")
        elif inventory_tracker[name]['status'] == 'OUT': # 사라졌다가 다시 나타난 객체
            if current_time - inventory_tracker[name].get('exit_time', 0) > EVENT_THRESHOLD_SECONDS:
                log_event(name, "IN")
            inventory_tracker[name]['status'] = 'IN'
            inventory_tracker[name]['entry_time'] = current_time
        
        inventory_tracker[name]['last_seen'] = current_time # 마지막으로 본 시간 갱신

    # Step B: 현재 안 보이는 객체 처리 (출고 이벤트)
    for name, data in inventory_tracker.items():
        if name not in current_frame_objects and data['status'] == 'IN': # 이전에 보였는데 지금은 안 보임
            # 15초 이상 머물렀던 객체인지?
            was_present_long_enough = data['last_seen'] - data['entry_time'] > EVENT_THRESHOLD_SECONDS
            # 15초 이상 사라진 상태인지?
            is_gone_long_enough = current_time - data['last_seen'] > EVENT_THRESHOLD_SECONDS
            
            if was_present_long_enough and is_gone_long_enough:
                data['status'] = 'OUT'
                data['exit_time'] = current_time
                log_event(name, "OUT")

    # 3. 화면 그리기 및 기타 로직 (기존 로직 대부분 재사용)
    largest_detection_area, largest_detection_crop = -1, None
    for *box, conf, cls in detections:
        # ... (이하 객체/바코드 그리기, 타임아웃, 모드전환 로직은 이전과 동일)
        x1, y1, x2, y2 = map(int, box)
        current_bbox = (x1, y1, x2, y2)
        object_crop = frame[y1:y2, x1:x2]
        if object_crop.size == 0: continue
        
        associated_barcode = None
        for bc in detected_barcodes:
            (bc_x, bc_y, bc_w, bc_h) = bc['rect']
            center_x, center_y = bc_x + bc_w // 2, bc_y + bc_h // 2
            if x1 < center_x < x2 and y1 < center_y < y2: associated_barcode = bc; break 

        if associated_barcode and associated_barcode['data'] not in known_barcodes:
            cv2.imshow('New Barcode Detected!', object_crop)
            object_name = get_name_from_popup("새 바코드 객체 등록", f"바코드: {associated_barcode['data']}\n\n이 객체의 이름을 입력하세요:")
            if object_name:
                register_new_object(object_crop, object_name, barcode_data=associated_barcode['data'])
                known_barcodes.add(associated_barcode['data'])
            else: print("⚠️ 등록이 취소되었습니다.")
            cv2.destroyWindow('New Barcode Detected!')

        name, score = recognize_object(object_crop)
        is_ignored = False
        if name != "Unknown":
            if name not in object_tracker: object_tracker[name] = {'last_bbox': current_bbox, 'last_seen_time': time.time(), 'is_ignored': False}
            else:
                iou = calculate_iou(object_tracker[name]['last_bbox'], current_bbox)
                if iou > IOU_THRESHOLD:
                    if time.time() - object_tracker[name]['last_seen_time'] > STATIC_TIMEOUT: object_tracker[name]['is_ignored'] = True
                else:
                    object_tracker[name]['last_bbox'] = current_bbox; object_tracker[name]['last_seen_time'] = time.time(); object_tracker[name]['is_ignored'] = False
            is_ignored = object_tracker[name]['is_ignored']

        if all_scan_mode or not is_ignored:
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            label = f"{name} ({score:.2f})"
            if not all_scan_mode:
                area = (x2 - x1) * (y2 - y1)
                if area > largest_detection_area:
                    largest_detection_area, largest_detection_crop = area, object_crop
        else:
            color = (128, 128, 128); label = f"{name} (ignored)"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    for bc in detected_barcodes:
        (x, y, w, h) = bc['rect']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 3)
        cv2.putText(frame, f"{bc['data']} ({bc['type']})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    mode_text = f"Mode: {'ALL SCAN' if all_scan_mode else 'TIMEOUT ACTIVE'}"
    cv2.putText(frame, mode_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, "r:Reg e:Edit s:Show a:Mode q:Quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow('Multi-Recognizer', frame)

    key = cv2.waitKey(1) & 0xFF
    
    # ... (q, r, e, s, a 키 로직은 이전과 동일)
    if key == ord('q'): break
    # ... 이하 키 입력 처리 로직은 생략 없이 이전과 동일하게 작동합니다.
    elif key == ord('r'):
        if all_scan_mode: print("⚠️ 수동 등록은 '타임아웃 활성화' 모드에서만 가능합니다."); continue
        if largest_detection_crop is not None:
            object_name = get_name_from_popup("수동 객체 등록", "등록할 객체의 이름을 입력하세요:")
            if object_name: register_new_object(largest_detection_crop, object_name)
            else: print("⚠️ 등록이 취소되었습니다.")
        else: print("⚠️ 등록할 활성 객체를 찾지 못했습니다.")
    elif key == ord('e'):
        if not known_objects_db: print("⚠️ 수정할 객체가 없습니다."); continue
        old_name = get_name_from_popup("이름 수정 (1/2)", f"현재 객체: {list(known_objects_db.keys())}\n\n변경할 객체 이름을 입력하세요:")
        if not old_name or old_name not in known_objects_db: print("⚠️ 작업이 취소되었거나 잘못된 이름입니다."); continue
        new_name = get_name_from_popup("이름 수정 (2/2)", f"'{old_name}'의 새로운 이름을 입력하세요:")
        if not new_name: print("⚠️ 작업이 취소되었습니다."); continue
        if new_name in known_objects_db:
            known_objects_db[new_name]['vectors'].extend(known_objects_db[old_name]['vectors'])
            known_objects_db[new_name]['images'].extend(known_objects_db[old_name]['images'])
            if 'barcodes' in known_objects_db[old_name]:
                if 'barcodes' not in known_objects_db[new_name]: known_objects_db[new_name]['barcodes'] = []
                known_objects_db[new_name]['barcodes'].extend(known_objects_db[old_name]['barcodes'])
            print(f"✅ '{new_name}'에 '{old_name}'의 샘플을 병합했습니다.")
        else:
            known_objects_db[new_name] = known_objects_db[old_name]
            print(f"✅ '{old_name}' -> '{new_name}'(으)로 이름이 변경되었습니다.")
        del known_objects_db[old_name]
    elif key == ord('s'):
        if not known_objects_db: print("⚠️ 보여줄 객체가 없습니다."); continue
        object_to_show = get_name_from_popup("이미지 보기", f"현재 객체: {list(known_objects_db.keys())}\n\n이미지를 볼 객체의 이름을 입력하세요:")
        if not object_to_show or object_to_show not in known_objects_db: print("⚠️ 작업이 취소되었거나 잘못된 이름입니다."); continue
        image_paths = known_objects_db[object_to_show]['images']
        if not image_paths: print(f"⚠️ '{object_to_show}'에 대한 이미지가 없습니다."); continue
        try:
            rep_image = cv2.imread(image_paths[0])
            if rep_image is None: print(f"⚠️ 이미지 파일을 읽을 수 없습니다: {image_paths[0]}"); continue
            cv2.imshow(f"Image of: {object_to_show}", rep_image)
            cv2.waitKey(0)
            cv2.destroyWindow(f"Image of: {object_to_show}")
        except Exception as e: print(f"이미지를 보여주는 중 오류 발생: {e}")
    elif key == ord('a'):
        all_scan_mode = not all_scan_mode
        mode_msg = "전체 인식 모드" if all_scan_mode else "타임아웃 활성화 모드"
        print(f"\n모드 변경: {mode_msg}")
        if all_scan_mode: object_tracker.clear()

# ----------------- 5. 종료 처리 -----------------
cap.release()
cv2.destroyAllWindows()
print("\n--- 데이터베이스 저장 중 ---")
try:
    with open(DB_FILE, 'wb') as f: pickle.dump(known_objects_db, f)
    print(f"✅ 데이터베이스 저장 완료. 총 {len(known_objects_db)}개의 객체가 저장되었습니다.")
except Exception as e: print(f"⚠️ 데이터베이스 저장 중 오류 발생: {e}")
print("\n--- 최종 등록된 사물 목록 ---")
if known_objects_db:
    for name, data in known_objects_db.items():
        print(f"- {name} (샘플 수: {len(data['vectors'])}, 바코드: {data.get('barcodes', 'N/A')})")
else:
    print("등록된 사물이 없습니다.")

