"""Benchmark: classical CV pipeline vs YOLO deep-learning pipeline.

Runs both plate-detection approaches over data/samples/*.jpg, using the
OpenALPR-style ground truth sitting next to each image
(`<name>.txt`: filename x y w h plate_text), and prints a comparison table.

Usage:
    python scripts/benchmark.py
"""

import glob
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2

from smart_toll.classical import detect_plate_classical
from smart_toll.pipeline import process_image

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union else 0.0


def char_similarity(a: str | None, b: str) -> float:
    return SequenceMatcher(None, a or "", b).ratio()


def run_deep(img):
    records = [r for r in process_image(img) if r["plate_box"]]
    if not records:
        return None, None
    best = max(records, key=lambda r: r["plate_confidence"])
    return best["plate_text"], tuple(best["plate_box"])


def main() -> None:
    cases = []
    for txt_path in sorted(glob.glob(str(SAMPLES / "*.txt"))):
        parts = Path(txt_path).read_text().split()
        img_path = SAMPLES / parts[0]
        if img_path.exists():
            x, y, w, h = map(int, parts[1:5])
            cases.append((img_path, (x, y, x + w, y + h), parts[5].upper()))

    if not cases:
        raise SystemExit("No ground-truth samples found in data/samples/")

    stats = {name: {"det": 0, "exact": 0, "sim": 0.0, "time": 0.0} for name in ("classical", "deep")}

    for img_path, gt_box, gt_text in cases:
        img = cv2.imread(str(img_path))
        for name, fn in (("classical", detect_plate_classical), ("deep", run_deep)):
            start = time.perf_counter()
            text, box = fn(img)
            elapsed = time.perf_counter() - start
            s = stats[name]
            s["time"] += elapsed
            s["det"] += bool(box and iou(box, gt_box) > 0.3)
            s["exact"] += text == gt_text
            s["sim"] += char_similarity(text, gt_text)
            print(f"{img_path.name[:12]}… {name:>9}: read={text!r:<16} truth={gt_text!r:<10} {elapsed:.2f}s")

    n = len(cases)
    print(f"\n{'Pipeline':<12}{'Plate found':<14}{'Exact OCR':<12}{'Char match':<12}{'Avg time'}")
    for name, s in stats.items():
        print(
            f"{name:<12}{s['det']}/{n:<12}{s['exact']}/{n:<10}"
            f"{s['sim'] / n:>7.0%}     {s['time'] / n:.2f}s"
        )


if __name__ == "__main__":
    main()
