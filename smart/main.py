# main.py - 스마트 트롤리 인식 시스템의 메인 프로그램 (지휘자)

import cv2
import os

# --- 다른 모듈에서 필요한 모든 것을 가져옵니다 ---
import config
from ui_utils import draw_elements_on_frame
from db_handler import load_database, save_database
from vision_utils import initialize_vision
from logic_handler import LogicHandler # 새로 만든 로직 핸들러 클래스

def main():
    """프로그램의 메인 실행 함수입니다."""
    
    # --- 1. 초기화 ---
    initialize_vision()
    if not os.path.exists(config.IMAGE_DIR): os.makedirs(config.IMAGE_DIR)
    known_objects_db, known_barcodes = load_database()
    
    # 핵심 로직을 처리할 핸들러 객체 생성
    handler = LogicHandler(known_objects_db, known_barcodes)

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("!!! 오류: 웹캠을 열 수 없습니다.")
        return

    print("\n--- 프로그램 시작 ---")
    print(" 'r':수동등록 | 'e':이름수정 | 's':이미지보기 | 'a':모드전환 | 'q':종료")

    try:
        # --- 2. 메인 루프 ---
        # 이제 메인 루프는 매우 단순해집니다.
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # 2-1. 모든 복잡한 로직은 핸들러가 처리
            draw_elements = handler.process_frame(frame)
            
            # 2-2. 핸들러가 반환한 정보로 화면에 그리기
            draw_elements_on_frame(frame, draw_elements)
            
            # 2-3. 화면 표시
            cv2.imshow('Multi-Recognizer', frame)

            # 2-4. 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            # 다른 키 입력은 핸들러가 처리
            handler.handle_key_press(key, frame)

    finally:
        # --- 3. 종료 처리 ---
        cap.release()
        cv2.destroyAllWindows()
        save_database(handler.known_objects_db) # 핸들러가 가지고 있는 최신 DB를 저장
        
        print("\n--- 최종 등록된 사물 목록 ---")
        if handler.known_objects_db:
            for name, data in handler.known_objects_db.items():
                print(f"- {name} (샘플 수: {len(data['vectors'])}, 바코드: {data.get('barcodes', 'N/A')})")
        else:
            print("등록된 사물이 없습니다.")

if __name__ == "__main__":
    main()