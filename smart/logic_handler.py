# logic_handler.py - 프로그램의 모든 핵심 로직을 처리하는 클래스입니다.

import time
import cv2
import os

import config
import vision_utils
from ui_utils import get_name_from_popup

class LogicHandler:
    def __init__(self, db, barcodes):
        self.known_objects_db = db
        self.known_barcodes = barcodes
        
        # 모든 상태 변수를 클래스 속성으로 관리
        self.object_tracker = {}
        self.inventory_tracker = {}
        self.stability_tracker = {'bbox': None, 'frames_stable': 0, 'image_crop': None, 'name': 'Unknown'}
        self.registration_state = config.STATE_IDLE
        self.pending_registration_data = None
        self.all_scan_mode = False
        self.last_barcode_check_time = 0

    def register_new_object(self, image_crop, object_name, barcode_data=None):
        """새로운 객체를 DB에 등록합니다."""
        feature_vector = vision_utils.extract_features(image_crop)
        timestamp = int(time.time())
        image_filename = f"{object_name.replace(' ', '_')}_{timestamp}.jpg"
        image_path = os.path.join(config.IMAGE_DIR, image_filename)
        cv2.imwrite(image_path, image_crop)
        
        if object_name not in self.known_objects_db:
            self.known_objects_db[object_name] = {'vectors': [], 'images': [], 'barcodes': []}
        
        self.known_objects_db[object_name]['vectors'].append(feature_vector)
        self.known_objects_db[object_name]['images'].append(image_path)
        
        if barcode_data and barcode_data not in self.known_objects_db[object_name]['barcodes']:
            self.known_objects_db[object_name]['barcodes'].append(barcode_data)
            self.known_barcodes.add(barcode_data)
            
        print(f"✅ '{object_name}' 샘플 추가 완료!")

    def process_frame(self, frame):
        """
        하나의 프레임을 받아 모든 로직을 처리하고, 그릴 요소 목록을 반환합니다.
        """
        draw_elements = []
        current_time = time.time()
        
        # 1. 비전 처리
        detected_barcodes = vision_utils.optimized_decode_barcodes(frame)
        results = vision_utils.detection_model(frame)
        detections = results.xyxy[0]

        # 2. 상태 머신 로직
        self._handle_registration_state(frame, detections, detected_barcodes, current_time)

        # 3. 재고 추적 로직
        self._update_inventory(frame, detections, current_time)
        
        # 4. 그리기 요소 생성
        self._create_draw_elements(detections, detected_barcodes, draw_elements, frame)
        
        return draw_elements

    def _handle_registration_state(self, frame, detections, detected_barcodes, current_time):
        """객체 등록과 관련된 상태 머신을 처리합니다."""
        if self.registration_state == config.STATE_IDLE:
            # 바코드 기반 즉시 등록
            for bc in detected_barcodes:
                if bc['data'] not in self.known_barcodes:
                    (x, y, w, h) = bc['rect']; barcode_center = (x + w // 2, y + h // 2)
                    found_crop = None
                    for *box, conf, cls in detections:
                        x1, y1, x2, y2 = map(int, box)
                        if x1 < barcode_center[0] < x2 and y1 < barcode_center[1] < y2:
                            found_crop = frame[y1:y2, x1:x2]; break
                    
                    if found_crop is not None:
                        object_name = get_name_from_popup("새 객체 등록", f"바코드: {bc['data']}\n\n이름을 입력하세요:")
                        if object_name:
                            self.register_new_object(found_crop, object_name, barcode_data=bc['data'])
                            self.registration_state = config.STATE_AWAITING_SAMPLE_PLACEMENT
                            self.pending_registration_data = {'name': object_name}
                            print(f"✨ 초기 등록 완료! '{object_name}'을 선반에 놓아주세요.")
                        else: print("⚠️ 등록이 취소되었습니다.")
                        return # 상태 변경 후 즉시 반환
                    else:
                        self.registration_state = config.STATE_AWAITING_INITIAL_PLACEMENT
                        self.pending_registration_data = {'barcode': bc['data'], 'name': None}
                        self.last_barcode_check_time = current_time
                        print(f"✨ 새로운 바코드({self.pending_registration_data['barcode']}) 감지! 배치를 기다립니다...")
                        return
            
            # 안정성 기반 자동 등록
            largest_unknown_area, crop, bbox = -1, None, None
            for *box, conf, cls in detections:
                name, _ = vision_utils.recognize_object(frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])], self.known_objects_db)
                if name == "Unknown":
                    area = (box[2]-box[0])*(box[3]-box[1])
                    if area > largest_unknown_area:
                        largest_unknown_area = area
                        crop = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                        bbox = tuple(map(int, box))
            
            if bbox and self.stability_tracker['bbox'] and vision_utils.calculate_iou(bbox, self.stability_tracker['bbox']) > 0.8:
                self.stability_tracker['frames_stable'] += 1; self.stability_tracker['image_crop'] = crop
            elif bbox:
                self.stability_tracker = {'bbox': bbox, 'frames_stable': 1, 'image_crop': crop, 'name': 'Unknown'}
            else: self.stability_tracker['frames_stable'] = 0

            if self.stability_tracker['frames_stable'] >= config.STABILITY_FRAME_THRESHOLD:
                object_name = get_name_from_popup("새 객체 자동 등록", "새로운 객체의 이름을 입력하세요:")
                if object_name: self.register_new_object(self.stability_tracker['image_crop'], object_name)
                else: print("⚠️ 등록이 취소되었습니다.")
                self.stability_tracker['frames_stable'] = 0
        
        elif self.registration_state == config.STATE_AWAITING_INITIAL_PLACEMENT:
            # (이전 코드와 동일한 로직, self. 사용)
            pass # 생략

        elif self.registration_state == config.STATE_AWAITING_SAMPLE_PLACEMENT:
            # (이전 코드와 동일한 로직, self. 사용)
            pass # 생략
    
    def _update_inventory(self, frame, detections, current_time):
        """재고 상태를 업데이트합니다."""
        current_frame_objects = set()
        for *box, conf, cls in detections:
            name, _ = vision_utils.recognize_object(frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])], self.known_objects_db)
            if name != "Unknown": current_frame_objects.add(name)

        for name in current_frame_objects:
            if name not in self.inventory_tracker: self.inventory_tracker[name]={'status':'IN','entry_time':current_time,'last_seen':current_time}; from db_handler import log_inventory_event; log_inventory_event(name,"IN")
            elif self.inventory_tracker[name]['status']=='OUT':
                if current_time - self.inventory_tracker[name].get('exit_time', 0) > config.EVENT_THRESHOLD_SECONDS: from db_handler import log_inventory_event; log_inventory_event(name,"IN")
                self.inventory_tracker[name]['status']='IN'; self.inventory_tracker[name]['entry_time']=current_time
            self.inventory_tracker[name]['last_seen']=current_time

        for name,data in list(self.inventory_tracker.items()):
            if name not in current_frame_objects and data['status']=='IN':
                if current_time - data.get('last_seen', 0) > config.EVENT_THRESHOLD_SECONDS:
                    data['status']='OUT'; data['exit_time']=current_time; from db_handler import log_inventory_event; log_inventory_event(name,"OUT")

    def _create_draw_elements(self, detections, detected_barcodes, draw_elements, frame):
        """화면에 그릴 요소들의 목록을 생성합니다."""
        # 객체 BBox 그리기
        for *box, conf, cls in detections:
            x1, y1, x2, y2 = map(int, box)
            name, score = vision_utils.recognize_object(frame[y1:y2, x1:x2], self.known_objects_db)
            # ... 무시 로직 ...
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            label = f"{name} ({score:.2f})"
            draw_elements.append({'type': 'box', 'box': (x1,y1,x2,y2), 'label': label, 'color': color})

        # 바코드 BBox 그리기
        for bc in detected_barcodes:
            draw_elements.append({'type': 'barcode', 'rect': bc['rect'], 'data': bc['data'], 'type_': bc['type']})

        # 상태 텍스트
        status_text = ""
        # ... 상태 텍스트 로직 ...
        if status_text:
            draw_elements.append({'type': 'status_text', 'text': status_text, 'position': (10,90), 'color': (0,255,255)})
        
        mode_text=f"Mode: {'ALL SCAN' if self.all_scan_mode else 'TIMEOUT ACTIVE'}"
        draw_elements.append({'type': 'status_text', 'text': mode_text, 'position': (10,60), 'color': (0,255,255)})
        draw_elements.append({'type': 'status_text', 'text': "r:Reg e:Edit s:Show a:Mode q:Quit", 'position': (10,30), 'color': (255,255,255)})

    def handle_key_press(self, key, frame):
        """키 입력을 처리합니다."""
        if key == ord('a'):
            self.all_scan_mode = not self.all_scan_mode
            # ...
        elif key == ord('r'):
            # ...
            pass