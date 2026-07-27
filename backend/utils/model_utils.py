"""
Model loading and inference utilities.
Handles EfficientNetB3 model loading and image preprocessing.
"""

import numpy as np
from PIL import Image
import io
import os
import logging

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE      = (224, 224)
MODEL_PATH    = os.getenv("MODEL_PATH", "model/plant_disease_model.keras")
NUM_CLASSES   = 38

# Optional: Google Drive file ID for auto-downloading the trained model at startup
# if it isn't already present at MODEL_PATH (used when the .keras file is too large
# to commit directly to the GitHub repo). Set via env var on Render, or leave the
# default below as a fallback.
MODEL_DRIVE_FILE_ID = os.getenv("MODEL_DRIVE_FILE_ID", "1oYPhPG22zzM7MBKFVVfwErFQfTB-DH2-")


def _download_model_if_missing():
    """Download the trained model from Google Drive if it isn't already on disk.

    Runs once at startup. Safe to call even if MODEL_DRIVE_FILE_ID is unset —
    it just does nothing in that case, and load_model() falls back to the mock
    model as before.
    """
    if os.path.exists(MODEL_PATH):
        return  # already present (e.g. committed to the repo, or downloaded previously)

    if not MODEL_DRIVE_FILE_ID:
        return  # no Drive file configured — nothing to download

    try:
        import gdown
    except ImportError:
        logger.warning("gdown not installed — cannot auto-download model from Drive. "
                        "Add 'gdown' to requirements.txt.")
        return

    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
    url = f"https://drive.google.com/uc?id={MODEL_DRIVE_FILE_ID}"
    logger.info(f"Model not found locally. Downloading from Google Drive ({MODEL_DRIVE_FILE_ID})...")
    try:
        gdown.download(url, MODEL_PATH, quiet=False)
        if os.path.exists(MODEL_PATH):
            logger.info(f"Model downloaded successfully to {MODEL_PATH}")
        else:
            logger.error("Model download appeared to finish but file is missing.")
    except Exception as e:
        logger.error(f"Failed to download model from Drive: {e}")

# ── Class names (sorted to match training order) ──────────────────────────────
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

_model = None


def load_model():
    """Load the TensorFlow model. Called once at startup."""
    global _model
    if _model is not None:
        return _model

    _download_model_if_missing()

    try:
        import tensorflow as tf
        if os.path.exists(MODEL_PATH):
            logger.info(f"Loading model from {MODEL_PATH}")
            _model = tf.keras.models.load_model(MODEL_PATH)
            logger.info("Model loaded successfully")
        else:
            logger.warning(f"Model file not found at {MODEL_PATH}. Using mock model.")
            _model = _create_mock_model()
    except ImportError:
        logger.warning("TensorFlow not available. Using mock model.")
        _model = _create_mock_model()
    except Exception as e:
        logger.error(f"Error loading model: {e}. Using mock model.")
        _model = _create_mock_model()

    return _model


def _create_mock_model():
    """
    Mock model for development/testing when real model isn't available.
    Returns random predictions with realistic confidence scores.
    """
    class MockModel:
        def predict(self, x, verbose=0):
            # Simulate softmax output with one dominant class
            logits = np.random.randn(x.shape[0], NUM_CLASSES) * 0.5
            logits[0, np.random.randint(0, NUM_CLASSES)] += 5.0  # dominant class
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            return exp / exp.sum(axis=1, keepdims=True)

    return MockModel()


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess image bytes for model inference.
    
    Steps:
      1. Open image with Pillow
      2. Convert to RGB (handles PNG with alpha, CMYK, etc.)
      3. Resize to 224x224
      4. Convert to numpy array
      5. Normalize to [0, 1]
      6. Add batch dimension
    
    Returns:
        np.ndarray of shape (1, 224, 224, 3)
    """
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)
    return arr


def predict(image_bytes: bytes) -> dict:
    """
    Run inference on image bytes.
    
    Returns:
        dict with keys: class_name, confidence, top3
    """
    model = load_model()
    img_array = preprocess_image(image_bytes)
    predictions = model.predict(img_array, verbose=0)
    probs = predictions[0]

    # Top prediction
    top_idx = int(np.argmax(probs))
    confidence = float(probs[top_idx])

    # Top-3 predictions
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [
        {"class": CLASS_NAMES[i], "confidence": float(probs[i])}
        for i in top3_idx
    ]

    return {
        "class_name": CLASS_NAMES[top_idx],
        "confidence": confidence,
        "top3": top3,
    }
