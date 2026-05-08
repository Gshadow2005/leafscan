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

```cmd
cd frontend
start leafscan_frontend.html
```

> Set Backend URL in the UI to: `http://127.0.0.1:8000`

---

## Run evaluation (confusion matrix)

```cmd
python evaluate.py
```

---

## Useful URLs (while server is running)

| Page | URL |
|------|-----|
| Swagger UI | http://127.0.0.1:8000/docs |
| Health check | http://127.0.0.1:8000/health |
| Class list | http://127.0.0.1:8000/classes |