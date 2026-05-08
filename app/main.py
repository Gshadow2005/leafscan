import os
import json
import io
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

# ── Lazy model loading (loaded once on startup) ──────────────────────────────
model = None
CLASS_NAMES = []

def project_paths():
    """Compute model/class file locations relative to this file."""
    # app/main.py -> project root
    root_dir = Path(__file__).resolve().parents[1]
    model_dir = root_dir / "model"
    return {
        "root_dir": root_dir,
        "class_names": model_dir / "class_names.json",
        "model_root": model_dir,
        "cv_models_dir": model_dir / "cv_models",
        "plant_disease_model": model_dir / "plant_disease_model.keras",
    }


def find_model():
    """Find the best available model file."""
    paths = project_paths()
    candidates = [
        paths["cv_models_dir"] / "model_fold_1.keras",
        paths["cv_models_dir"] / "model_fold_2.keras",
        paths["cv_models_dir"] / "model_fold_3.keras",
        paths["cv_models_dir"] / "model_fold_4.keras",
        paths["cv_models_dir"] / "model_fold_5.keras",
        paths["plant_disease_model"],
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, CLASS_NAMES

    # Load class names
    paths = project_paths()
    if paths["class_names"].exists():
        with open(paths["class_names"], encoding="utf-8") as f:
            CLASS_NAMES = json.load(f)
        print(f"✅ Loaded {len(CLASS_NAMES)} class names.")
    else:

        CLASS_NAMES = [
            "Alternaria_D", "Botrytis Leaf Blight", "Bulb Rot", "Bulb_blight-D",
            "Caterpillar-P", "Downy mildew", "Fusarium-D", "Healthy leaves",
            "Iris yellow virus_augment", "Purple blotch", "Rust", "Virosis-D",
            "Xanthomonas Leaf Blight", "onion1", "stemphylium Leaf Blight"
        ]
        print("⚠️  class_names.json not found — using default class list.")

    # Load TensorFlow + model
    model_path = find_model()
    if model_path:
        print(f"Loading model from: {model_path} ...")
        import tensorflow as tf
        from keras.models import load_model as keras_load
        global tf_available
        tf_available = True
        model = keras_load(model_path)
        print("✅ Model loaded and ready.")
    else:
        print("⚠️  No model file found. /predict will return a demo response.")
        print("    Place your .keras model in cv_models/ or the project root.")

    yield  # App runs here

    # Cleanup (optional)
    print("Shutting down LeafScan backend.")


app = FastAPI(
    title="LeafScan API",
    description="Onion plant disease detection using MobileNetV2 transfer learning.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS — allow all origins (restrict in production if needed) ───────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

IMG_SIZE = (224, 224)
MAX_FILE_SIZE_MB = 10
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Open, resize, and prepare image for model inference."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)          # shape (224, 224, 3)
    return np.expand_dims(arr, axis=0)              # shape (1, 224, 224, 3)
    # Note: Rescaling (÷255) is baked into the model via keras.layers.Rescaling


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "LeafScan API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None,
        "classes": len(CLASS_NAMES),
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": model is not None}


@app.get("/classes")
def get_classes():
    return {"classes": CLASS_NAMES, "count": len(CLASS_NAMES)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # ── Validate file type ────────────────────────────────────────────────
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Allowed: jpeg, png, webp, bmp."
        )

    image_bytes = await file.read()

    # ── Validate file size ────────────────────────────────────────────────
    if len(image_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE_MB}MB."
        )

    # ── Demo mode (no model loaded) ───────────────────────────────────────
    if model is None:
        import random
        demo_scores = {name: round(random.uniform(0.5, 5.0), 2) for name in CLASS_NAMES}
        top = max(demo_scores, key=demo_scores.get)
        demo_scores[top] = round(random.uniform(70, 95), 2)
        total = sum(demo_scores.values())
        demo_scores = {k: round(v / total * 100, 2) for k, v in demo_scores.items()}
        return JSONResponse({
            "disease": top,
            "confidence": demo_scores[top],
            "all_scores": demo_scores,
            "demo_mode": True,
            "note": "No model loaded — returning demo data. Add your .keras model to cv_models/."
        })

    # ── Run inference ─────────────────────────────────────────────────────
    try:
        arr = preprocess_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not process image: {str(e)}")

    try:
        preds = model.predict(arr, verbose=0)[0]          # shape (num_classes,)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

    idx = int(np.argmax(preds))
    disease = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class_{idx}"
    confidence = float(preds[idx]) * 100

    all_scores = {
        CLASS_NAMES[i]: round(float(p) * 100, 2)
        for i, p in enumerate(preds)
        if i < len(CLASS_NAMES)
    }

    return JSONResponse({
        "disease": disease,
        "confidence": round(confidence, 2),
        "all_scores": all_scores,
        "demo_mode": False,
    })