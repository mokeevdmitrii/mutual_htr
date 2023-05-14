import cv2
import numpy as np


# returns coef, so not a transform
def resize_if_greater(image: np.ndarray, max_h: int, max_w: int):
    img = image.copy()
    img_h, img_w, img_c = img.shape
    if img_h <= max_h and img_w <= max_w:
        coef = 1.
    else:
        coef = max(img_h / max_h, img_w / max_w)
        
    h, w = int(img_h / coef), int(img_w / coef)
    img = cv2.resize(img, (w, h))
    return img, coef


def make_img_padding(image: np.ndarray, max_h: int, max_w: int):
    img = image.copy()
    img_h, img_w, img_c = img.shape
    bg = np.zeros((max_h, max_w, img_c), dtype=np.uint8)
    x1 = 0
    y1 = (max_h - img_h) // 2
    x2 = x1 + img_w
    y2 = y1 + img_h
    bg[y1:y2, x1:x2, :] = img.copy()
    return bg
