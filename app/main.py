"""FastAPI backend for the Smart Tolling demo.

Run locally:
    .venv/bin/uvicorn app.main:app --reload

POST /api/process with an image file → per-vehicle toll records
plus a base64-encoded annotated preview image.
"""

import base64
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from smart_toll import process_image
from smart_toll.viz import annotate

app = FastAPI(title="Smart Tolling System", version="2.0")

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Demo blacklist — plates that trigger a gate denial.
BLACKLIST = {"MH12DE1433", "STOLEN1"}

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@app.post("/api/process")
async def process(file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image too large (max 10 MB)")

    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image — upload a JPEG or PNG")

    # Bound the working resolution to keep CPU inference fast.
    h, w = img.shape[:2]
    if max(h, w) > 1920:
        scale = 1920 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    start = time.perf_counter()
    records = process_image(img, blacklist=BLACKLIST)
    elapsed = time.perf_counter() - start

    annotated = annotate(img, records)
    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    preview = base64.b64encode(buf).decode() if ok else None

    return {
        "vehicles": records,
        "processing_seconds": round(elapsed, 2),
        "annotated_image": f"data:image/jpeg;base64,{preview}" if preview else None,
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


SAMPLES_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"


@app.get("/api/samples")
def list_samples():
    """Example images the UI offers for one-click demos."""
    if not SAMPLES_DIR.exists():
        return {"samples": []}
    return {"samples": sorted(p.name for p in SAMPLES_DIR.glob("*.jpg"))[:6]}


@app.get("/samples/{name}")
def get_sample(name: str):
    path = (SAMPLES_DIR / name).resolve()
    if path.parent != SAMPLES_DIR.resolve() or not path.exists():
        raise HTTPException(404, "No such sample")
    return FileResponse(path)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
