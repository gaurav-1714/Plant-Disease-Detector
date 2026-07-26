"""
Basic API tests for the Plant Disease Detection API.
Run with: pytest tests/ -v
(Uses the mock model automatically when no trained model file is present.)
"""

import io
import os
import sys

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from main import app  # noqa: E402

client = TestClient(app)


def _fake_image_bytes(fmt="JPEG", size=(256, 256)):
    img = Image.new("RGB", size, color=(60, 120, 60))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf


def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "message" in resp.json()


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_get_classes():
    resp = client.get("/classes")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 38
    assert len(data["classes"]) == 38


def test_get_crops():
    resp = client.get("/crops")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_crops"] > 0


def test_get_stats():
    resp = client.get("/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_classes"] == 38


def test_get_disease_valid():
    resp = client.get("/disease/Apple___Apple_scab")
    assert resp.status_code == 200
    data = resp.json()
    assert data["crop"] == "Apple"
    assert data["disease"] == "Apple Scab"


def test_get_disease_invalid():
    resp = client.get("/disease/Not___A_Real_Class")
    assert resp.status_code == 404


def test_predict_valid_image():
    img_buf = _fake_image_bytes()
    resp = client.post(
        "/predict",
        files={"file": ("leaf.jpg", img_buf, "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "class_name" in data
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["top3"]) == 3
    assert "treatment" in data
    assert "prevention" in data


def test_predict_rejects_bad_content_type():
    resp = client.post(
        "/predict",
        files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert resp.status_code == 415


def test_predict_rejects_oversized_file():
    big = io.BytesIO(b"0" * (11 * 1024 * 1024))
    resp = client.post(
        "/predict",
        files={"file": ("big.jpg", big, "image/jpeg")},
    )
    assert resp.status_code == 413
