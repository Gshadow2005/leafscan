# 🌿 LeafScan — Backend API

FastAPI backend for the LeafScan plant disease detection app.
Wraps your trained MobileNetV2 model into a REST API with a `/predict` endpoint.

---

## 📁 File Structure

```
leafscan-backend/
├── app/
│   ├── main.py              ← FastAPI app (all routes)
│   └── test_api.py         ← Local test script
├── model/
│   ├── class_names.json   ← Copy from your ML project
│   └── *.keras             ← Your trained model(s)
│
├── frontend/
│   └── leafscan_frontend.html
│
├── requirements.txt        ← Python dependencies
├── Dockerfile              ← For Docker / cloud deployment
├── render.yaml.txt        ← One-click Render.com deploy config
└── .gitignore
```

---

## ⚡ Quick Start (Local)

### Step 1 — Put model artifacts into `model/`

```cmd
copy ..\class_names.json model\
copy ..\cv_models\model_fold_1.keras model\cv_models\
copy ..\plant_disease_model.keras model\
```

> This repo expects models under `model/`.

> If you don’t have a trained model yet, the API still works in **demo mode** — it returns realistic random scores so you can develop the frontend without waiting for training.

---

### Step 2 — Create virtual environment

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
uvicorn app.main:app --reload
```

You should see:
```
✅ Loaded 15 class names.
✅ Model loaded and ready.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

---

### Step 5 — Test it

**Option A — Browser (Swagger UI)**  
Open: http://localhost:8000/docs  
Click `POST /predict` → "Try it out" → upload a leaf photo → Execute

**Option B — Run the test script**

```cmd
pip install requests Pillow
python app/test_api.py
```

---

## 🔌 API Reference

### `GET /`
Returns service info and whether the model is loaded.

### `GET /health`
Health check. Returns `{ "status": "ok", "model_ready": true }`.

### `GET /classes`
Returns the full list of disease class names.

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
**Max file size:** 10MB

---

## 🚀 Deploy

### Deploy to Render.com (Free)

- Push this folder to GitHub as `leafscan-backend`
- Create a Render Web Service
- Ensure `render.yaml.txt` is used by your workflow
- Render will start the server using:
  - `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

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
| `No model file found` | Put your `.keras` file under `model/` (or `model/cv_models/`) |
| CORS error from React | Already handled — all origins allowed by default |
| 413 error on large image | Upload images under 10MB |

