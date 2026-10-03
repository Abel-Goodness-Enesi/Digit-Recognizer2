# Digit Recognizer

A dense neural network (64→32→16→10), trained on MNIST, served through a FastAPI
backend. Draw, upload, or use your camera — the page sends the image to `/predict`,
which runs the real trained model and returns a digit, a confidence score, and a
"needs review" flag for anything under 90% confidence.

**Real numbers, not placeholders:** 97.0% accuracy on 10,000 held-out test images
the model never trained on.

## What's in here

- `app.py` — FastAPI backend. Loads the trained weights once at startup, does
  image preprocessing (grayscale, resize to 28×28, auto-corrects for photos that
  are dark-on-light instead of MNIST's light-on-dark), runs a plain numpy forward
  pass, and serves the frontend.
- `digit_weights.npz` — the actual trained weights (not a placeholder — verified
  to match the original Keras model to within 1e-6 on real test images).
- `static/index.html` — the frontend: draw/upload/camera tabs, calls `/predict`.
- `requirements.txt` — exact pinned dependencies.

No TensorFlow dependency at runtime — inference is a handful of numpy matrix
multiplications, which keeps the deploy light and fast.

## Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Then open `http://localhost:8000`.

## Deploy to Render

1. Push this folder to a GitHub repo.
2. On Render: **New → Web Service**, connect the repo.
3. Settings:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Environment:** Python 3
4. Deploy. Render assigns a public URL — that's the live app, no further setup needed.

## API

`POST /predict` — multipart form, field name `file`, any common image format.

Response:
```json
{
  "digit": 7,
  "confidence": 0.9999967813491821,
  "needs_review": false,
  "threshold": 0.9,
  "probabilities": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
}
```

`GET /health` — returns `{"status": "ok"}`, useful for Render's health checks.
