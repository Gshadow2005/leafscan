FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app + assets
COPY app ./app
COPY model ./model
COPY frontend ./frontend
COPY class_names.json ./class_names.json

# Expose port
EXPOSE 8000

# Start API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

