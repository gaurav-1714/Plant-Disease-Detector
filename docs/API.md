# API Documentation

Base URL (local): `http://localhost:8000`
Interactive docs (Swagger UI): `http://localhost:8000/docs`

## `GET /`
Root info. Returns API name, version, and links.

## `GET /health`
Liveness probe. Returns `{"status": "ok", "model": "loaded"}`.

## `POST /predict`
Upload a leaf image and get a diagnosis.

**Request:** `multipart/form-data`, field `file` (JPEG/PNG/WEBP, max 10MB)

**Response `200`:**
```json
{
  "class_name": "Tomato___Late_blight",
  "confidence": 0.9732,
  "crop": "Tomato",
  "disease": "Late Blight",
  "is_healthy": false,
  "severity": "High",
  "description": "...",
  "symptoms": ["..."],
  "treatment": ["..."],
  "prevention": ["..."],
  "top3": [
    {"class": "Tomato___Late_blight", "confidence": 0.9732},
    {"class": "Tomato___Early_blight", "confidence": 0.0181},
    {"class": "Tomato___healthy", "confidence": 0.0043}
  ],
  "inference_ms": 27.4,
  "filename": "leaf.jpg"
}
```

**Errors:**
| Status | Reason |
|---|---|
| 415 | Unsupported file type |
| 413 | File exceeds 10MB |
| 500 | Model inference failure |

## `GET /classes`
Returns all 38 supported class names.

## `GET /disease/{class_name}`
Returns disease info for a specific class (404 if unknown).

## `GET /crops`
Returns supported crops with disease lists.

## `GET /stats`
Returns model/dataset statistics used on the frontend dashboard.
