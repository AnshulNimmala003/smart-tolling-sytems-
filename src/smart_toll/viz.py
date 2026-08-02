"""Draw pipeline results onto a frame."""

import numpy as np

GREEN = (80, 200, 120)
RED = (60, 60, 230)
YELLOW = (60, 200, 255)


def annotate(img_bgr: np.ndarray, records: list[dict]) -> np.ndarray:
    import cv2

    out = img_bgr.copy()
    for r in records:
        x1, y1, x2, y2 = r["box"]
        allowed = r["toll"]["allowed"]
        color = GREEN if allowed else RED
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f"{r['label']} | Rs.{r['toll']['toll_amount']}"
        if r["plate_text"]:
            label += f" | {r['plate_text']}"
        cv2.putText(out, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if r["plate_box"]:
            px1, py1, px2, py2 = r["plate_box"]
            cv2.rectangle(out, (px1, py1), (px2, py2), YELLOW, 2)
    return out
