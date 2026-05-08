# 🌿 LeafScan — Backend API

FastAPI backend for the LeafScan onion plant disease detection app.
Wraps a trained MobileNetV2 Keras model into a REST API with a `/predict` endpoint.

---

## 📁 File Structure

```
leafscan-backend/
├── app/
│   ├── main.py                  ← FastAPI app (all routes)
│   └── test_api.py              ← Local test script
├── model/
│   ├── class_names.json         ← 15 disease class labels
│   ├── cv_models/
│   │   └── model_fold_1.keras   ← Cross-validated model (fold 1–5 supported)
│   └── plant_disease_model.keras ← Fallback single model
├── frontend/
│   └── leafscan_frontend.html   ← Standalone web UI
├── evaluate.py                  ← Confusion matrix + classification report
├── requirements.txt             ← Python dependencies
├── Dockerfile                   ← Docker / cloud deployment
├── render.yaml.txt              ← One-click Render.com deploy config
├── RUNBOOK.md                   ← Quick commands reference
└── .gitignore
```

---

## ⚡ Quick Start (Local)

### Step 1 — Place model artifacts into `model/`

```cmd
copy ..\class_names.json model\
copy ..\cv_models\model_fold_1.keras model\cv_models\
copy ..\plant_disease_model.keras model\
```

> Model files (`.keras`) and images are excluded from git via `.gitignore`.

> **No model yet?** The API runs in **demo mode** — it returns realistic random scores so you can test the frontend while training is still ongoing.

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
✅ Loaded 15 class names.
✅ Model loaded and ready.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

---

### Step 5 — Open the frontend

```cmd
cd frontend
start leafscan_frontend.html
```

Make sure the **Backend URL** field in the UI shows `http://127.0.0.1:8000`, then upload or capture a leaf photo.

---

### Step 6 — Run the test suite (optional)

```cmd
cd app
python test_api.py
```

---

## 🔌 API Reference

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
  },
  "demo_mode": false
}
```

**Accepted formats:** JPEG, PNG, WebP, BMP
**Max file size:** 10 MB

---

## 🧪 Model Details

- **Architecture:** MobileNetV2 (transfer learning)
- **Framework:** TensorFlow / Keras
- **Input size:** 224 × 224 RGB
- **Classes:** 15 onion disease categories
- **Preprocessing:** Rescaling (÷255) baked into the model via `keras.layers.Rescaling`

**Model priority** (first found is used):
1. `model/cv_models/model_fold_1.keras` → `model_fold_5.keras`
2. `model/plant_disease_model.keras`

---

## 📊 Evaluation

Run the evaluation script to generate a confusion matrix and classification report against your test dataset:

```cmd
python evaluate.py
```

Outputs:
- Accuracy and loss printed to console
- Full `classification_report` per class
- `confusion_matrix.png` saved to project root

> Requires `dataset/test/` directory and `plant_disease_model.keras` in project root.

---

## 🚀 Deploy

### Deploy to Render.com (Free — Singapore region)

1. Push this folder to GitHub as `leafscan-backend`
2. Create a new **Web Service** on Render
3. Use the config from `render.yaml.txt`
4. Render starts the server with:
   ```
   uvicorn app.main:app --host 0.0.0.0 --port $PORT
   ```
5. Update the **Backend URL** in `leafscan_frontend.html` to your Render URL

### Deploy with Docker

```bash
docker build -t leafscan-api .
docker run -p 8000:8000 leafscan-api
```

---

## ❓ Troubleshooting

| Problem | Fix |
|---|---|
| `No module named tensorflow` | Run `venv\Scripts\activate` first |
| `No model file found` | Put your `.keras` file in `model/cv_models/` or `model/` |
| CORS error from browser | Already handled — all origins allowed by default |
| 413 error on upload | Image must be under 10 MB |
| Frontend shows "Could not connect" | Make sure the FastAPI server is running on port 8000 |