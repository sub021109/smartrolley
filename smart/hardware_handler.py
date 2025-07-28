# hardware_handler.py - PIR 센서 등 하드웨어 제어를 담당합니다.

import RPi.GPIO as GPIO
import time
import config

def setup_pir_sensor():
    """PIR 센서를 위한 GPIO 핀을 설정합니다."""
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(config.PIR_PIN, GPIO.IN)
    print("PIR 센서 설정 완료.")

def wait_for_motion(callback_function):
    """
    움직임이 감지될 때까지 무한 대기하다가, 감지되면 콜백 함수를 실행합니다.
    """
    print("--- 스마트 트롤리 시스템 대기 모드 ---")
    print("움직임 감지를 시작합니다...")
    
    last_triggered_time = 0
    try:
        while True:
            if GPIO.input(config.PIR_PIN): # 움직임 감지!
                if time.time() - last_triggered_time > config.COOLDOWN_SECONDS:
                    callback_function() # 메인 인식 세션 함수 호출
                    last_triggered_time = time.time()
                    print("\n다시 대기 모드로 전환합니다...")
                else:
                    # 쿨다운 중에는 아무것도 하지 않음
                    pass
            time.sleep(0.5) # CPU 사용량 줄이기
    except KeyboardInterrupt:
        print("\n프로그램 종료 신호 감지.")
    finally:
        GPIO.cleanup() # GPIO 리소스 정리