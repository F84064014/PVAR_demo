import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def draw_box(frame, pt1, pt2, color):
    cv2.rectangle(frame, pt1, pt2, color=color, thickness=2)

def draw_color(frame, pt1, pt2, color, pos='all'):
    (x1, y1), (x2, y2) = pt1, pt2
    x3, x4 = x2+3, x2+7
    if pos == 'all':
        y3, y4 = y1, y2
    elif pos == 'top':
        y3, y4 = y1, (y1+y2)//2
    elif pos == 'bot':
        y3, y4 = (y1+y2)//2, y2
    else:
        raise NotImplementedError(f"unknown arguemnt pos={pos}")
    cv2.rectangle(frame, (x3, y3), (x4, y4), color=color, thickness=3)

def draw_label(frame, pt1, pt2, text):
    (x1, y1), (x2, y2) = pt1, pt2

    xl, yl = x1, y1 - 10 if y1 - 10 > 10 else y1 + 20

    (wl, hl), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (xl, yl - hl), (xl + wl, yl + 5), (0, 255, 0), -1)
    cv2.putText(frame, text, (xl, yl), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)