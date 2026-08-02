"""Original classical-CV plate detector (contours + morphology), kept for benchmarking.

This is the project's first-generation approach: Sobel edges, morphological
closing, contour filtering by aspect ratio/area, then OCR. The deep-learning
pipeline in `detector.py` replaces it; `scripts/benchmark.py` compares both.
"""

import numpy as np

from .ocr import read_plate


def _ratio_check(area: float, width: float, height: float) -> bool:
    ratio = width / height if height else 0
    if 0 < ratio < 1:
        ratio = 1 / ratio
    return not (area < 1063.62 or area > 73862.5) and (3 <= ratio <= 6)


def _clean_plate(plate_bgr: np.ndarray):
    import cv2

    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    areas = [cv2.contourArea(c) for c in contours]
    max_cnt = contours[int(np.argmax(areas))]
    x, y, w, h = cv2.boundingRect(max_cnt)
    if not _ratio_check(max(areas), w, h):
        return None
    return (x, y, w, h)


def detect_plate_classical(img_bgr: np.ndarray) -> tuple[str | None, tuple[int, int, int, int] | None]:
    """Return (plate_text, plate_box) using the classical pipeline."""
    import cv2

    blurred = cv2.GaussianBlur(img_bgr, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    edges = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    _, edges = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, element)
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        (_, _), (w, h), angle = cv2.minAreaRect(cnt)
        norm_angle = -angle if w > h else 90 + angle
        if norm_angle > 15 or not h or not w or not _ratio_check(h * w, w, h):
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        plate_img = img_bgr[y : y + bh, x : x + bw]
        if plate_img.size == 0 or np.mean(plate_img) < 115:
            continue
        inner = _clean_plate(plate_img)
        if inner:
            ix, iy, iw, ih = inner
            box = (x + ix, y + iy, x + ix + iw, y + iy + ih)
            crop = img_bgr[box[1] : box[3], box[0] : box[2]]
            text, _ = read_plate(crop)
            if text:
                return text, box
    return None, None
