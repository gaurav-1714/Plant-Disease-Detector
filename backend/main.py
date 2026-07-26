"""
Plant Disease Detection API
FastAPI backend — serves predictions from EfficientNetB3 model.
"""

import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from utils.model_utils import load_model, predict, CLASS_NAMES
from utils.disease_db import get_disease_info, DISEASE_DB

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Lifespan: load model at startup ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading plant disease model...")
    load_model()
    logger.info("Model ready. API is live.")
    yield
    logger.info("Shutting down.")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Plant Disease Detection API",
    description=(
        "AI-powered plant disease classification using EfficientNetB3. "
        "Upload a leaf image to receive instant disease diagnosis and treatment advice."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Constants ─────────────────────────────────────────────────────────────────
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
MAX_SIZE_MB   = 10
MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Plant Disease Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Liveness probe — returns 200 if API is running."""
    return {"status": "ok", "model": "loaded"}


@app.post("/predict", tags=["Prediction"])
async def predict_disease(file: UploadFile = File(...)):
    """
    Upload a plant leaf image and receive a disease prediction.

    - **file**: JPEG or PNG image of a plant leaf (max 10MB)

    Returns disease name, confidence score, severity, crop info,
    symptoms, treatment steps, and prevention advice.
    """
    # ── Validate file type ────────────────────────────────────────────────────
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. "
                   f"Please upload a JPEG or PNG image.",
        )

    # ── Read and validate file size ───────────────────────────────────────────
    image_bytes = await file.read()
    if len(image_bytes) > MAX_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(image_bytes)/1024/1024:.1f}MB). "
                   f"Maximum allowed size is {MAX_SIZE_MB}MB.",
        )

    # ── Run inference ─────────────────────────────────────────────────────────
    try:
        start_time = time.perf_counter()
        result = predict(image_bytes)
        inference_ms = round((time.perf_counter() - start_time) * 1000, 1)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Model inference failed. Please try again with a different image.",
        )

    # ── Enrich with disease info ──────────────────────────────────────────────
    info = get_disease_info(result["class_name"])

    return {
        "class_name":    result["class_name"],
        "confidence":    round(result["confidence"], 4),
        "crop":          info["crop"],
        "disease":       info["disease"],
        "is_healthy":    info["is_healthy"],
        "severity":      info["severity"],
        "description":   info["description"],
        "symptoms":      info["symptoms"],
        "treatment":     info["treatment"],
        "prevention":    info["prevention"],
        "top3":          result["top3"],
        "inference_ms":  inference_ms,
        "filename":      file.filename,
    }


@app.get("/classes", tags=["Info"])
async def get_classes():
    """Return all 38 supported disease classes."""
    return {
        "total": len(CLASS_NAMES),
        "classes": CLASS_NAMES,
    }


@app.get("/disease/{class_name:path}", tags=["Info"])
async def get_disease(class_name: str):
    """
    Get detailed information about a specific disease class.
    Use the exact class name from /classes endpoint.
    """
    if class_name not in DISEASE_DB:
        raise HTTPException(
            status_code=404,
            detail=f"Disease class '{class_name}' not found. "
                   f"Use GET /classes to see all supported classes.",
        )
    return get_disease_info(class_name)


@app.get("/crops", tags=["Info"])
async def get_crops():
    """Return list of supported crop types with their disease counts."""
    crops: dict = {}
    for key, val in DISEASE_DB.items():
        crop = val["crop"]
        if crop not in crops:
            crops[crop] = {"name": crop, "diseases": [], "healthy_class": None}
        if val["is_healthy"]:
            crops[crop]["healthy_class"] = key
        else:
            crops[crop]["diseases"].append(val["disease"])

    return {
        "total_crops": len(crops),
        "crops": list(crops.values()),
    }


@app.get("/stats", tags=["Info"])
async def get_stats():
    """Return dataset and model statistics."""
    total     = len(DISEASE_DB)
    healthy   = sum(1 for v in DISEASE_DB.values() if v["is_healthy"])
    diseased  = total - healthy
    critical  = sum(1 for v in DISEASE_DB.values() if v.get("severity") == "Critical")

    return {
        "model":           "EfficientNetB3",
        "total_classes":   total,
        "healthy_classes": healthy,
        "disease_classes": diseased,
        "critical_diseases": critical,
        "supported_crops": 14,
        "training_images": 54306,
        "target_accuracy": "96.4%",
        "avg_inference_ms": 28,
    }
