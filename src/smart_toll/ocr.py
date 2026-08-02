"""Plate text reading via EasyOCR, with plate-oriented cleanup."""

import re
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _reader():
    import easyocr

    return easyocr.Reader(["en"], gpu=False, verbose=False)


def read_plate(plate_bgr: np.ndarray) -> tuple[str | None, float]:
    """Return (normalized_plate_text, confidence) or (None, 0.0)."""
    import cv2

    # Upscale small crops — OCR accuracy drops sharply below ~50px height.
    h, w = plate_bgr.shape[:2]
    if h < 60:
        scale = 60 / h
        plate_bgr = cv2.resize(plate_bgr, (int(w * scale), 60), interpolation=cv2.INTER_CUBIC)

    results = _reader().readtext(plate_bgr, detail=1)

    # Keep plate-like fragments (4-12 alphanumerics); dealer frames and state
    # names OCR as separate fragments and get filtered out here.
    candidates = []
    for box, text, conf in results:
        normalized = re.sub(r"[^A-Z0-9]", "", text.upper())
        if 4 <= len(normalized) <= 12:
            candidates.append((normalized, float(conf)))

    if not candidates:
        return None, 0.0

    def score(c):
        text, conf = c
        # Real plates almost always mix letters and digits — prefer those
        # over all-letter fragments like dealer-frame names.
        mixed = 2.0 if any(ch.isdigit() for ch in text) and any(ch.isalpha() for ch in text) else 1.0
        return conf * len(text) * mixed

    return max(candidates, key=score)
