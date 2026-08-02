"""CLI: run the smart-toll pipeline on an image.

Usage:
    python scripts/run_pipeline.py path/to/image.jpg [--save annotated.jpg]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2

from smart_toll import process_image
from smart_toll.viz import annotate


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart toll pipeline")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--save", help="Path to save annotated image", default=None)
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")

    records = process_image(img)
    print(json.dumps(records, indent=2))

    if args.save:
        cv2.imwrite(args.save, annotate(img, records))
        print(f"Annotated image saved to {args.save}")


if __name__ == "__main__":
    main()
