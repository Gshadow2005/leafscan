import os
import json
import io
import numpy as np
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

# ── Model configuration ───────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "plant_disease_model.keras"
CLASS_NAMES_PATH = MODEL_PATH.parent / "class_names.json"

# ── Lazy model loading (loaded once on startup) ──────────────────────────────
model = None
CLASS_NAMES = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, CLASS_NAMES

    # Load class names
    if CLASS_NAMES_PATH.exists():
        with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
            CLASS_NAMES = json.load(f)
        print(f"Loaded {len(CLASS_NAMES)} class names.")
    else:
        CLASS_NAMES = [
            "Alternaria_D", "Botrytis Leaf Blight", "Bulb Rot", "Bulb_blight-D",
            "Caterpillar-P", "Downy mildew", "Fusarium-D", "Healthy leaves",
            "Iris yellow virus_augment", "Purple blotch", "Rust", "Virosis-D",
            "Xanthomonas Leaf Blight", "onion1", "stemphylium Leaf Blight"
        ]
        print("class_names.json not found — using default class list.")

    # Load TensorFlow + model
    if not MODEL_PATH.exists():
        raise RuntimeError("Service is unavailable. Please contact the administrator.")

    print(f"Loading model from: {MODEL_PATH} ...")
    from keras.models import load_model as keras_load
    model = keras_load(str(MODEL_PATH))
    print("Model loaded and ready.")

    yield  # App runs here

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
    from keras.utils import load_img, img_to_array
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        img = load_img(tmp_path, target_size=IMG_SIZE)
        arr = img_to_array(img)
    finally:
        os.unlink(tmp_path)

    return np.expand_dims(arr, axis=0)


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

    # ── Guard: model must be loaded ───────────────────────────────────────
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable. Please try again later or contact support."
        )

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
    })