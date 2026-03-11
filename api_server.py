"""
=============================================================
 Malaria Detection – FastAPI Backend Server
=============================================================
Endpoints:
  GET  /               → health check
  GET  /model-info     → loaded model metadata
  POST /predict        → single image prediction
  POST /predict-batch  → multiple images
  GET  /stats          → session statistics
=============================================================
Install:  pip install fastapi uvicorn python-multipart pillow tensorflow
Run:      uvicorn deployment.api_server:app --host 0.0.0.0 --port 8000 --reload
=============================================================
"""

import os
import io
import sys
import time
import uuid
import logging
from datetime import datetime
from typing import List, Optional
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from prediction.predictor import MalariaPredictor, preprocess_image

# ─── Logging ──────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── App setup ────────────────────────────────────────────
app = FastAPI(
    title="Malaria Detection API",
    description="Deep-learning powered malaria cell detection (Parasitized vs Uninfected)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Globals ──────────────────────────────────────────────
MODEL_PATH   = os.getenv("MODEL_PATH", "saved_models/best_model.keras")
MODEL_NAME   = os.getenv("MODEL_NAME", "best_model")
THRESHOLD    = float(os.getenv("THRESHOLD", "0.50"))
MAX_IMG_SIZE = 10 * 1024 * 1024   # 10 MB

predictor: Optional[MalariaPredictor] = None
session_stats = defaultdict(int)
session_start = datetime.now()

# ─── Pydantic schemas ─────────────────────────────────────
class PredictionResponse(BaseModel):
    request_id:   str
    filename:     str
    label:        str
    confidence:   float
    raw_proba:    float
    inference_ms: float
    timestamp:    str

class BatchPredictionResponse(BaseModel):
    request_id:  str
    total:       int
    results:     List[PredictionResponse]
    summary:     dict
    total_ms:    float

class ModelInfo(BaseModel):
    name:       str
    path:       str
    threshold:  float
    img_size:   List[int]
    status:     str

class HealthResponse(BaseModel):
    status:      str
    uptime_secs: float
    model_loaded: bool


# ─── Startup ──────────────────────────────────────────────
@app.on_event("startup")
async def load_model():
    global predictor
    if Path(MODEL_PATH).exists():
        logger.info(f"Loading model from {MODEL_PATH} …")
        predictor = MalariaPredictor(MODEL_PATH,
                                      threshold=THRESHOLD,
                                      model_name=MODEL_NAME)
        logger.info("✅  Model loaded successfully")
    else:
        logger.warning(f"⚠️   Model not found at {MODEL_PATH}. Run training first.")


# ─── Utility ──────────────────────────────────────────────
def _validate_image(file: UploadFile) -> bytes:
    allowed = {"image/jpeg", "image/png", "image/tiff", "image/bmp",
               "image/jpg", "image/tif"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400,
                            detail=f"Unsupported type: {file.content_type}. "
                                   f"Allowed: JPEG, PNG, TIFF, BMP")
    content = file.file.read()
    if len(content) > MAX_IMG_SIZE:
        raise HTTPException(status_code=413,
                            detail=f"File too large (max {MAX_IMG_SIZE//1024//1024} MB)")
    return content


def _make_pred_response(result, req_id: str) -> PredictionResponse:
    return PredictionResponse(
        request_id=req_id,
        filename=result.filename,
        label=result.label,
        confidence=round(result.confidence, 6),
        raw_proba=round(result.raw_proba, 6),
        inference_ms=round(result.inference_ms, 2),
        timestamp=datetime.now().isoformat(),
    )


# ─── Endpoints ────────────────────────────────────────────
@app.get("/", response_model=HealthResponse)
async def health():
    uptime = (datetime.now() - session_start).total_seconds()
    return HealthResponse(
        status="healthy",
        uptime_secs=round(uptime, 1),
        model_loaded=predictor is not None,
    )


@app.get("/model-info", response_model=ModelInfo)
async def model_info():
    status = "loaded" if predictor else "not_loaded"
    return ModelInfo(
        name=MODEL_NAME,
        path=MODEL_PATH,
        threshold=THRESHOLD,
        img_size=[128, 128],
        status=status,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    req_id  = str(uuid.uuid4())[:8]
    content = _validate_image(file)
    image   = Image.open(io.BytesIO(content))

    result  = predictor.predict_single(image, file.filename)
    session_stats["total"] += 1
    session_stats[result.label] += 1

    logger.info(f"[{req_id}] {file.filename} → {result.label} ({result.confidence:.3f})")
    return _make_pred_response(result, req_id)


@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(files: List[UploadFile] = File(...)):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Max 50 images per batch")

    req_id  = str(uuid.uuid4())[:8]
    t0      = time.perf_counter()
    results = []

    for f in files:
        content = _validate_image(f)
        image   = Image.open(io.BytesIO(content))
        r       = predictor.predict_single(image, f.filename)
        results.append(_make_pred_response(r, req_id))
        session_stats["total"] += 1
        session_stats[r.label] += 1

    total_ms = (time.perf_counter() - t0) * 1000
    n_para   = sum(1 for r in results if r.label == "Parasitized")
    summary  = {
        "total":        len(results),
        "Parasitized":  n_para,
        "Uninfected":   len(results) - n_para,
        "prevalence_%": round(n_para / len(results) * 100, 1),
    }
    logger.info(f"[{req_id}] Batch {len(files)} images → {summary}")
    return BatchPredictionResponse(
        request_id=req_id,
        total=len(results),
        results=results,
        summary=summary,
        total_ms=round(total_ms, 2),
    )


@app.get("/stats")
async def stats():
    uptime = (datetime.now() - session_start).total_seconds()
    return {
        "session_start":      session_start.isoformat(),
        "uptime_secs":        round(uptime, 1),
        "total_predictions":  session_stats["total"],
        "Parasitized":        session_stats["Parasitized"],
        "Uninfected":         session_stats["Uninfected"],
        "prevalence_%":       round(
            session_stats["Parasitized"] / max(session_stats["total"], 1) * 100, 1
        ),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
