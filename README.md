# LeafScan

A group project for onion plant disease detection, built with a FastAPI backend and a standalone HTML frontend. The backend wraps a trained MobileNetV2 Keras model into a REST API with a `/predict` endpoint.

---

## File Structure

```
leafscan/
├── app/
│   ├── main.py                        ← FastAPI app (all routes)
│   └── test_api.py                    ← Local test script
├── model/
│   ├── class_names.json               ← 15 disease class labels
│   └── plant_disease_model.keras      ← Trained model
├── trained_models/                    ← Cross-validated model folds (git-ignored)
│   ├── model_fold_1.keras
│   ├── model_fold_2.keras
│   ├── model_fold_3.keras
│   ├── model_fold_4.keras
│   ├── model_fold_5.keras
│   └── plant_disease_model(old).keras
├── frontend/
│   ├── leafscan_frontend.html         ← Production web UI (points to deployed API)
│   └── leafscan_frontend_local.html   ← Local dev UI (configurable backend URL)
├── index.html                         ← Redirect to frontend/leafscan_frontend.html
├── requirements.txt                   ← Python dependencies
├── Dockerfile                         ← Docker / cloud deployment
├── RUNBOOK.md                         ← Quick commands reference
└── .gitignore
```

---

## Quick Start (Local)

### Step 1 — Place model artifacts into `model/`

```cmd
copy ..\class_names.json model\
copy ..\plant_disease_model.keras model\
```

> Model files (`.keras`) and images are excluded from git via `.gitignore`.

---

### Step 2 — Create and activate a virtual environment

```cmd
python -m venv venv
venv\Scripts\activate
```

---

### Step 3 — Install dependencies

```cmd
pip install -r requirements.txt
```

---

### Step 4 — Start the server

```cmd
cd app
C:\Python312\python.exe -m uvicorn main:app --reload --port 8000
```

You should see:
```
Loaded 15 class names.
Model loaded and ready.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

> If no model file is found, the server will raise a `RuntimeError` on startup.

---

### Step 5 — Open the local frontend

```cmd
cd frontend
start leafscan_frontend_local.html
```

Set the **Backend URL** field in the UI to `http://127.0.0.1:8000`, then upload or capture a leaf photo.

> Use `leafscan_frontend_local.html` for local development — it has a configurable backend URL input and a raw JSON response viewer.
> `leafscan_frontend.html` is the production UI hardcoded to the deployed Hugging Face Spaces URL.

---

### Step 6 — Run the test suite (optional)

```cmd
cd app
python test_api.py
```

---

## API Reference

### `GET /`
Returns service info and whether the model is loaded.

### `GET /health`
Health check. Returns `{ "status": "ok", "model_ready": true }`.

### `GET /classes`
Returns the full list of 15 disease class names.

### `POST /predict`
Accepts a multipart image upload. Returns:

```json
{
  "disease": "Healthy leaves",
  "confidence": 94.7,
  "all_scores": {
    "Healthy leaves": 94.7,
    "Purple blotch": 2.1,
    "Alternaria_D": 1.4
  }
}
```

**Accepted formats:** JPEG, PNG, WebP, BMP  
**Max file size:** 10 MB

---

## Model Details

- **Architecture:** MobileNetV2 (transfer learning)
- **Framework:** TensorFlow / Keras
- **Input size:** 224 × 224 RGB
- **Classes:** 15 onion disease categories
- **Preprocessing:** Rescaling (÷255) baked into the model via `keras.layers.Rescaling`

The model is loaded from `model/plant_disease_model.keras` on startup. Cross-validated fold models (`model_fold_1.keras` through `model_fold_5.keras`) are stored in `trained_models/` and excluded from git.

---

## Disease Classes

| # | Class |
|---|-------|
| 1 | Alternaria_D |
| 2 | Botrytis Leaf Blight |
| 3 | Bulb Rot |
| 4 | Bulb_blight-D |
| 5 | Caterpillar-P |
| 6 | Downy mildew |
| 7 | Fusarium-D |
| 8 | Healthy leaves |
| 9 | Iris yellow virus_augment |
| 10 | Purple blotch |
| 11 | Rust |
| 12 | Virosis-D |
| 13 | Xanthomonas Leaf Blight |
| 14 | onion1 |
| 15 | stemphylium Leaf Blight |

---

## Deploy

### Hugging Face Spaces (Docker)

The app is deployed on Hugging Face Spaces using the included `Dockerfile`. The space runs at:

```
https://papabing84-leafscan.hf.space
```

The Dockerfile copies `app/`, `model/`, and `frontend/` into the image and starts the server on port 7860 (or `$PORT`):

```bash
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}
```

### Docker (local)

```bash
docker build -t leafscan-api .
docker run -p 7860:7860 leafscan-api
```

### CORS

The production API only allows requests from:

```
https://gshadow2005.github.io
```

To test locally or from another origin, temporarily change `allow_origins` in `app/main.py` to `["*"]`.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named tensorflow` | Run `venv\Scripts\activate` first |
| `RuntimeError` on startup | Model file not found — place `.keras` file in `model/` |
| CORS error from browser | Set `allow_origins=["*"]` in `app/main.py` for local testing |
| 413 error on upload | Image must be under 10 MB |
| Frontend shows "Could not connect" | Make sure the FastAPI server is running and the Backend URL is correct |
| Server waking up slowly | Hugging Face Spaces free tier sleeps after inactivity — wait ~30 seconds |