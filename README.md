# 🌿 PlantGuard AI — Plant Disease Detection

An end-to-end AI application that identifies plant diseases from leaf photos and
provides instant diagnosis, severity assessment, and treatment/prevention advice.
Built for the **Agriculture & Intelligent Supply Chains** major project track.

## 1. Problem

Smallholder farmers lose a significant share of crop yield every year to diseases
that, if caught early, are treatable. Diagnosis today typically requires an
agricultural expert, which is often unavailable, slow, or costly to access —
especially in rural areas. This project puts an expert-level, instant diagnosis
tool in anyone's hands via a photo upload.

**Existing solutions & limitations:** Most PlantVillage-based demos stop at a raw
class label ("Tomato___Late_blight") with no actionable next step for the farmer.
This project closes that gap by pairing the classifier with a structured
treatment/prevention knowledge base and a production-style API + UI.

**Why AI is required:** Disease symptoms vary subtly in color, texture, and
pattern across 14 crop species and 38 disease/healthy states — a scale and
consistency problem that is a natural fit for a CNN-based image classifier,
not rule-based image processing.

## 2. Architecture

```
                ┌─────────────┐        multipart/form-data        ┌──────────────────┐
   User photo → │  Frontend   │ ─────────────────────────────────▶│   FastAPI backend │
                │ (HTML/JS)   │                                    │  /predict etc.    │
                └─────────────┘ ◀───────────────────────────────── └──────────────────┘
                     JSON response                                          │
                                                                             ▼
                                                                  ┌──────────────────────┐
                                                                  │ EfficientNetB3 model  │
                                                                  │ (TensorFlow/Keras)    │
                                                                  └──────────────────────┘
                                                                             │
                                                                             ▼
                                                                  ┌──────────────────────┐
                                                                  │ Disease knowledge DB  │
                                                                  │ (38 classes, static)  │
                                                                  └──────────────────────┘
```

- **Model**: EfficientNetB3 (ImageNet-pretrained backbone + custom head),
  fine-tuned on the [PlantVillage dataset](https://huggingface.co/datasets/1aurent/PlantVillage)
  (54,306 images, 38 classes, 14 crops).
- **Backend**: FastAPI serving `/predict` and supporting info endpoints,
  with input validation, structured error handling, and a mock-model fallback
  so the API runs even before a trained `.keras` file is provided.
- **Frontend**: Framework-free HTML/CSS/JS single page — drag-and-drop upload,
  live confidence bar, severity badge, and treatment/prevention breakdown.
- **Knowledge base**: `backend/utils/disease_db.py` — hand-curated symptoms,
  treatment steps, and prevention tips for all 38 classes.

## 3. Repository layout

```
plant-disease-detection/
├── backend/
│   ├── main.py                 # FastAPI app & routes
│   ├── utils/
│   │   ├── model_utils.py      # model loading, preprocessing, inference
│   │   └── disease_db.py       # disease knowledge base (38 classes)
│   ├── training/
│   │   └── train_model.py      # EfficientNetB3 training script
│   ├── tests/
│   │   └── test_api.py         # pytest API tests
│   ├── model/                  # trained model goes here (.keras, gitignored)
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── index.html
│   └── assets/ (style.css, app.js, config.js)
├── docs/
│   └── API.md
├── .github/workflows/ci.yml    # GitHub Actions: run tests on push/PR
├── docker-compose.yml
├── LICENSE
└── README.md
```

## 4. Getting started

### Backend

```bash
cd backend
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

The API runs at `http://localhost:8000` (`/docs` for Swagger UI). If no trained
model is found at `MODEL_PATH` (`backend/model/plant_disease_model.keras`), the
API automatically falls back to a lightweight mock model so the whole stack is
runnable end-to-end without a GPU or dataset — useful for local dev and demos.

### Frontend

The frontend is static — no build step. Serve it with any static server, e.g.:

```bash
cd frontend
python -m http.server 3000
```

Open `http://localhost:3000`. Update `frontend/assets/config.js` if your API
runs somewhere other than `http://localhost:8000`.

### Docker (both services)

```bash
docker compose up --build
```

Backend → `http://localhost:8000`, frontend → `http://localhost:3000`.

## 5. Training the model

```bash
cd backend/training
python train_model.py --data_dir /path/to/PlantVillage --epochs 15 --fine_tune_epochs 5
```

- Downloads not included here — get the dataset from
  [PlantVillage on Hugging Face](https://huggingface.co/datasets/1aurent/PlantVillage)
  or [Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
  and arrange it into one folder per class matching the 38 names in
  `backend/utils/model_utils.py::CLASS_NAMES`.
- Two-phase transfer learning: (1) train a new classification head on a frozen
  EfficientNetB3 backbone, (2) unfreeze the top ~30 backbone layers and
  fine-tune at a lower learning rate.
- Outputs `plant_disease_model.keras` and `training_metrics.json` into
  `backend/model/`. Copy/mount that folder next to `main.py` (or set
  `MODEL_PATH`) to serve real predictions.

## 6. Engineering justification

| Decision | Rationale |
|---|---|
| **Dataset — PlantVillage** | Largest widely-used labeled leaf-disease dataset (54k images, 38 classes, 14 crops); standard benchmark, well-documented. |
| **Model — EfficientNetB3** | Strong accuracy-to-compute ratio vs. ResNet/VGG; ImageNet pretraining gives good low-data transfer for a two-phase fine-tune. |
| **Backend — FastAPI** | Async, auto-generated OpenAPI docs, native file-upload handling, strong typing — an industry-standard choice for ML inference APIs. |
| **Frontend — vanilla HTML/JS** | No build tooling required, trivially deployable (GitHub Pages, static hosting), keeps the project reviewable end-to-end without a Node toolchain. |
| **Evaluation metrics** | Top-1 accuracy + top-3 accuracy (reported during training) since visually similar diseases are common; confidence + top-3 alternatives are also surfaced to the end user for transparency. |
| **System architecture** | Clean separation of model/serving/knowledge-base/UI layers so each can be improved (e.g. swap the model, add crops) independently. |

## 7. API reference

See [`docs/API.md`](docs/API.md).

## 8. Testing

```bash
cd backend
pytest tests/ -v
```

CI (`.github/workflows/ci.yml`) runs the same suite on every push/PR.

## 9. Limitations & future work

- Model ships un-trained in this repo (large binary, gitignored) — run
  `training/train_model.py` or plug in a `.keras` checkpoint to get real
  predictions instead of the mock fallback.
- Single-leaf, single-disease assumption; no multi-lesion/severity segmentation yet.
- No authentication/rate limiting — add before any public deployment.
- Could extend with: user accounts + history, GPS-based regional disease alerts,
  batch upload, mobile app wrapper.

## 10. License

MIT — see [LICENSE](LICENSE).
