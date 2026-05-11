# LeafScan — Runbook

Quick reference for daily dev commands.

---

## Start the API server

```cmd
cd app
C:\Python312\python.exe -m uvicorn main:app --reload --port 8000
```

---

## Run the test suite

```cmd
cd app
python test_api.py
```

---

## Open the frontend

### Local dev frontend

```cmd
cd frontend
start leafscan_frontend_local.html
```

> Set Backend URL in the UI to: `http://127.0.0.1:8000`  
> Includes a configurable backend URL input and a raw JSON response viewer.

### Production frontend (deployed)

```cmd
start index.html
```

> Points to the live Hugging Face Spaces API at `https://papabing84-leafscan.hf.space`

---

## Docker (local)

```bash
docker build -t leafscan-api .
docker run -p 7860:7860 leafscan-api
```

---

## Useful URLs (while server is running)

| Page | URL |
|------|-----|
| Swagger UI | http://127.0.0.1:8000/docs |
| Health check | http://127.0.0.1:8000/health |
| Class list | http://127.0.0.1:8000/classes |

---

## Model files

| File | Location | Notes |
|------|----------|-------|
| `plant_disease_model.keras` | `model/` | Active model loaded by the API |
| `model_fold_1–5.keras` | `trained_models/` | Cross-validated folds, git-ignored |
| `plant_disease_model(old).keras` | `trained_models/` | Previous model, kept for reference |
| `class_names.json` | `model/` | 15 class labels loaded on startup |

> To swap the active model, replace `model/plant_disease_model.keras` and restart the server.

---

## CORS

Production API only allows: `https://gshadow2005.github.io`

To test from localhost or another origin, temporarily set in `app/main.py`:

```python
allow_origins=["*"]
```

Remember to revert before deploying. ok