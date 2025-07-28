# db_handler.py - Pickle DB와 CSV 로그 파일 관리를 담당합니다.

import pickle
import os
from datetime import datetime
import config

def load_database():
    """Pickle 파일에서 객체 데이터베이스를 로드합니다."""
    known_objects_db = {}
    known_barcodes = set()
    try:
        with open(config.DB_FILE, 'rb') as f:
            known_objects_db = pickle.load(f)
        for data in known_objects_db.values():
            for bc in data.get('barcodes', []):
                known_barcodes.add(bc)
        print(f"✅ DB 로딩 완료. {len(known_objects_db)}개 객체, {len(known_barcodes)}개 바코드 로드.")
    except FileNotFoundError:
        print("ℹ️ 저장된 DB 파일이 없습니다. 새로 시작합니다.")
    except Exception as e:
        print(f"⚠️ DB 로딩 오류: {e}")
    return known_objects_db, known_barcodes

def save_database(db):
    """객체 데이터베이스를 Pickle 파일에 저장합니다."""
    try:
        with open(config.DB_FILE, 'wb') as f:
            pickle.dump(db, f)
        print(f"\n✅ 데이터베이스 저장 완료.")
    except Exception as e:
        print(f"\n⚠️ 데이터베이스 저장 중 오류 발생: {e}")

def log_inventory_event(object_name, event_type):
    """CSV 파일에 재고 입출고 이벤트를 기록합니다."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp},{object_name},{event_type}\n"
    if not os.path.exists(config.LOG_FILE):
        with open(config.LOG_FILE, "w") as f:
            f.write("Timestamp,ObjectName,Event\n")
    with open(config.LOG_FILE, "a") as f:
        f.write(log_entry)
    print(f"✍️ LOG: {object_name} - {event_type}")