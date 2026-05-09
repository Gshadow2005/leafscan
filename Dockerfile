FROM python:3.11-slim

WORKDIR /app

RUN useradd -m -u 1000 user
USER user

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY model ./model
COPY frontend ./frontend
COPY class_names.json ./class_names.json

EXPOSE 7860

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]