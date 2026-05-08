cd app
C:\Python312\python.exe -m uvicorn main:app --reload --port 8000
python test_api.py

cd frontend
start http://127.0.0.1:8000 
start leafscan_frontend.html