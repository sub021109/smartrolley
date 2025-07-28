# ui_utils.py - UI 관련 함수들을 담당합니다.

import tkinter
from tkinter import simpledialog
import cv2

def get_name_from_popup(title, prompt):
    """지정된 제목과 내용으로 사용자 입력을 받는 팝업창을 띄웁니다."""
    root = tkinter.Tk()
    root.withdraw()
    user_input = simpledialog.askstring(title=title, prompt=prompt)
    root.destroy()
    return user_input

def draw_elements_on_frame(frame, elements):
    """
    주어진 프레임에 그리기 요소(사각형, 텍스트)들을 그립니다.
    """
    for element in elements:
        if element['type'] == 'box':
            box = element['box']
            label = element['label']
            color = element['color']
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        elif element['type'] == 'barcode':
            rect = element['rect']
            data = element['data']
            type_ = element['type_']
            (x, y, w, h) = rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 3)
            cv2.putText(frame, f"{data} ({type_})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
        elif element['type'] == 'status_text':
            text = element['text']
            position = element['position']
            color = element['color']
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)