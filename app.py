"""
Digit recognizer — FastAPI backend.

Serves a trained neural network (64-32-16-10 dense layers, trained on MNIST,
~97% test accuracy) via a /predict endpoint, plus the frontend page.

The model runs as a plain numpy forward pass, not full TensorFlow — this keeps
the deployed service light (fast Render builds, low memory), since only the
learned weights are needed at inference time, not the training framework.
"""

import io
import os

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

APP_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(APP_DIR, "digit_weights.npz")
CONFIDENCE_THRESHOLD = 0.9  # matches the threshold chosen during model validation

app = FastAPI(title="Digit Recognizer")

# ---- load trained weights once at startup ----
_weights = np.load(WEIGHTS_PATH)
W1, b1 = _weights["W1"], _weights["b1"]
W2, b2 = _weights["W2"], _weights["b2"]
W3, b3 = _weights["W3"], _weights["b3"]
W4, b4 = _weights["W4"], _weights["b4"]


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def forward(x_flat: np.ndarray) -> np.ndarray:
    """x_flat: (N, 784) normalized [0,1] pixel values. Returns (N, 10) probabilities."""
    a1 = relu(x_flat @ W1 + b1)
    a2 = relu(a1 @ W2 + b2)
    a3 = relu(a2 @ W3 + b3)
    logits = a3 @ W4 + b4
    return softmax(logits)


def preprocess_image(raw_bytes: bytes) -> np.ndarray:
    """
    Turn an arbitrary uploaded/drawn image into the 28x28, white-digit-on-black,
    normalized format the model was trained on.

    Real uploaded images won't already look like MNIST, so this does the actual
    work: grayscale, resize, and — the part that matters most — auto-detect and
    correct polarity, since a photo of pencil-on-paper is dark-on-light, the
    opposite of MNIST's light-on-dark convention.
    """
    img = Image.open(io.BytesIO(raw_bytes)).convert("L")  # grayscale

    # if the image has an alpha channel composited in already via convert("L"),
    # transparent regions typically read as black; a canvas drawing is usually
    # already black-background/white-stroke, so this mainly corrects photos.
    img = ImageOps.exif_transpose(img)

    # resize to 28x28 with a good downsampling filter
    img = img.resize((28, 28), Image.LANCZOS)

    arr = np.array(img, dtype=np.float32)

    # auto-detect polarity: MNIST digits are bright strokes on a dark background.
    # if the image is mostly bright (a typical white-paper photo), invert it.
    if arr.mean() > 127:
        arr = 255.0 - arr

    arr = arr / 255.0
    return arr.reshape(1, -1)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        x = preprocess_image(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    probs = forward(x)[0]  # (10,)
    digit = int(np.argmax(probs))
    confidence = float(probs[digit])
    needs_review = confidence < CONFIDENCE_THRESHOLD

    return JSONResponse({
        "digit": digit,
        "confidence": confidence,
        "needs_review": needs_review,
        "threshold": CONFIDENCE_THRESHOLD,
        "probabilities": [float(p) for p in probs],
    })


@app.get("/health")
def health():
    return {"status": "ok"}


# ---- serve the frontend ----
app.mount("/", StaticFiles(directory=os.path.join(APP_DIR, "static"), html=True), name="static")
