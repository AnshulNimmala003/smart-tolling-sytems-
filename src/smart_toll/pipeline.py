"""End-to-end smart-toll pipeline: image → vehicles → plates → OCR → toll decision."""

from dataclasses import asdict

import numpy as np

from .detector import Detection, detect_plates, detect_vehicles
from .ocr import read_plate
from .pricing import decide

DEFAULT_BLACKLIST: set[str] = set()


def _box_center_inside(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> bool:
    cx = (inner[0] + inner[2]) / 2
    cy = (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def process_image(img_bgr: np.ndarray, blacklist: set[str] | None = None) -> list[dict]:
    """Run the full pipeline on one frame. Returns one record per detected vehicle."""
    blacklist = blacklist if blacklist is not None else DEFAULT_BLACKLIST

    vehicles = detect_vehicles(img_bgr)

    # Detect the plate inside each vehicle's crop — the plate is relatively
    # larger there, which the detector handles far better than full frames.
    for v in vehicles:
        x1, y1, x2, y2 = v.box
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        for (px1, py1, px2, py2), plate_conf in detect_plates(crop, conf=0.15):
            if plate_conf > v.plate_confidence:
                v.plate_box = (x1 + px1, y1 + py1, x1 + px2, y1 + py2)
                v.plate_confidence = plate_conf

    # Fallback for tight crops where no whole vehicle is visible.
    if not vehicles:
        for plate_box, plate_conf in detect_plates(img_bgr, conf=0.15):
            vehicles.append(
                Detection(
                    box=plate_box,
                    label="car",
                    confidence=plate_conf,
                    plate_box=plate_box,
                    plate_confidence=plate_conf,
                )
            )

    records = []
    for v in vehicles:
        if v.plate_box:
            x1, y1, x2, y2 = v.plate_box
            v.plate_text, ocr_conf = read_plate(img_bgr[y1:y2, x1:x2])
            v.extras["ocr_confidence"] = round(ocr_conf, 3)

        decision = decide(v.label, v.plate_text, blacklist)
        record = asdict(v)
        record["toll"] = asdict(decision)
        records.append(record)
    return records
