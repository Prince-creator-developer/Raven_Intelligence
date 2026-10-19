import os
import sys
import json
import time
import glob
import shutil
from typing import Dict, List, Optional
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
if MODELS_DIR not in sys.path:
    sys.path.append(MODELS_DIR)

from numpy_inference import NumpyVisemeNet
from agent_synthesizer import agent_synthesizer
from dataset_manager import (
    save_user_sample,
    get_dataset_statistics,
    get_dynamic_vocabulary,
    add_custom_phrase,
    clean_folder_name,
    VOCAB_FILE,
    DATASET_DIR
)

# Optional PyTorch for local VS Code training
try:
    import torch
    from lip_model import VisemeTemporalNet
    from train_custom import train_custom_100_epochs
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    VisemeTemporalNet = None
    train_custom_100_epochs = None

app = FastAPI(
    title="SilentVoice AI",
    description="Neural Silent Speech Interface — Samsung Innovation Campus",
    version="4.0.0"
)

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

DEVICE = torch.device("cpu") if HAS_TORCH else None
WEIGHTS_PATH = os.path.join(MODELS_DIR, "lip_model.pth")
NPZ_WEIGHTS_PATH = os.path.join(MODELS_DIR, "lip_model_weights.npz")

active_vocabulary: List[str] = get_dynamic_vocabulary()
torch_model = None
numpy_model: Optional[NumpyVisemeNet] = None
REQUIRED_SAMPLES_PER_WORD = 10  # User must record exactly 10 samples per word


def initialize_or_reload_model():
    global torch_model, numpy_model, active_vocabulary
    active_vocabulary = get_dynamic_vocabulary()
    num_classes = len(active_vocabulary)

    if os.path.exists(NPZ_WEIGHTS_PATH):
        try:
            numpy_model = NumpyVisemeNet(NPZ_WEIGHTS_PATH)
            print(f"[NUMPY] Loaded {num_classes} classes from weights.npz")
        except Exception as e:
            print(f"[NUMPY WARNING] {e}")

    if HAS_TORCH and VisemeTemporalNet is not None:
        try:
            torch_model = VisemeTemporalNet(input_dim=12, hidden_dim=64, num_classes=num_classes).to(DEVICE)
            if os.path.exists(WEIGHTS_PATH):
                state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
                if state_dict["fc2.weight"].shape[0] == num_classes:
                    torch_model.load_state_dict(state_dict)
                    print(f"[TORCH] Loaded weights for {num_classes} classes")
            torch_model.eval()
        except Exception as e:
            print(f"[TORCH WARNING] {e}")


initialize_or_reload_model()


# ──────────────────────────────────────────────
# REQUEST SCHEMAS
# ──────────────────────────────────────────────
class TemporalTensorRequest(BaseModel):
    tensor: List[List[float]]

class RecordSampleRequest(BaseModel):
    word: str
    tensor: List[List[float]]

class AddWordRequest(BaseModel):
    phrase: str

class ClearWordRequest(BaseModel):
    word: str


# ──────────────────────────────────────────────
# TEMPLATE LOADER
# ──────────────────────────────────────────────
import jinja2

def get_template_html() -> str:
    candidates = [
        os.path.join(BASE_DIR, "templates", "index.html"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "index.html"),
        os.path.join("/var/task", "templates", "index.html"),
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                with open(c, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                pass
    return ""


# ──────────────────────────────────────────────
# PAGES
# ──────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    stats = get_dataset_statistics()
    vocab = stats.get("vocabulary", [])
    raw_html = get_template_html()
    if raw_html:
        try:
            t = jinja2.Template(raw_html)
            rendered = t.render(
                request=request,
                vocabulary=vocab,
                dataset_stats=stats,
                required_samples=REQUIRED_SAMPLES_PER_WORD
            )
            return HTMLResponse(content=rendered)
        except Exception as e:
            print(f"[TEMPLATE ERROR] {e}")
    try:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "vocabulary": vocab,
            "dataset_stats": stats,
            "required_samples": REQUIRED_SAMPLES_PER_WORD
        })
    except Exception as e:
        return HTMLResponse(content=f"<html><body><h1>SilentVoice AI</h1><p>Error: {e}</p></body></html>", status_code=200)


# ──────────────────────────────────────────────
# API: INFERENCE
# ──────────────────────────────────────────────
@app.post("/api/predict")
async def predict_silent_speech(req: TemporalTensorRequest):
    t0 = time.perf_counter()
    data = np.array(req.tensor, dtype=np.float32)

    if data.shape != (30, 12):
        return JSONResponse({"status": "error", "message": f"Expected (30,12) got {data.shape}"}, status_code=400)

    probs = None
    attn_list = []

    if numpy_model is not None:
        probs, attn_raw = numpy_model.predict(data)
        attn_list = attn_raw.tolist() if hasattr(attn_raw, "tolist") else list(attn_raw)
    elif HAS_TORCH and torch_model is not None:
        inp = torch.tensor(data).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probabilities, attn_weights = torch_model(inp)
            probs = probabilities[0].cpu().numpy()
            attn_list = attn_weights[0].squeeze().cpu().numpy().tolist()
    else:
        return JSONResponse({"status": "error", "message": "No model loaded."}, status_code=500)

    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    pred_word = active_vocabulary[pred_idx] if pred_idx < len(active_vocabulary) else "RESTING"

    latency = round((time.perf_counter() - t0) * 1000, 2)
    agent_out = agent_synthesizer.process_decoded_word(pred_word, confidence)
    prob_dict = {active_vocabulary[i]: round(float(probs[i]), 4) for i in range(min(len(active_vocabulary), len(probs)))}

    return JSONResponse({
        "status": "success",
        "predicted_word": pred_word,
        "confidence": round(confidence, 4),
        "synthesized_sentence": agent_out.get("synthesized_sentence") or pred_word,
        "ai_dialogue_reply": agent_out.get("ai_dialogue_reply", ""),
        "suggested_next_words": agent_out.get("suggested_next_words", []),
        "continuous_sentence": agent_out.get("continuous_sentence", ""),
        "patient_name": agent_out.get("patient_name", ""),
        "is_active_speech": agent_out["is_active_speech"],
        "compound_detected": agent_out["compound_detected"],
        "probabilities": prob_dict,
        "latency_ms": latency,
        "attention_weights": attn_list
    })


# ──────────────────────────────────────────────
# API: DATASET MANAGEMENT
# ──────────────────────────────────────────────
@app.get("/api/dataset_stats")
async def dataset_stats():
    stats = get_dataset_statistics()
    # Enrich with per-word sample count for frontend progress bars
    enriched = {}
    for word in stats["vocabulary"]:
        folder = clean_folder_name(word)
        folder_path = os.path.join(DATASET_DIR, folder)
        if os.path.exists(folder_path):
            # Only count REAL user recordings, not baseline files
            real = [f for f in os.listdir(folder_path) if f.startswith("sample_") and f.endswith(".npy")]
            enriched[word] = len(real)
        else:
            enriched[word] = 0
    stats["real_recordings"] = enriched
    stats["required_per_word"] = REQUIRED_SAMPLES_PER_WORD
    return JSONResponse(stats)


@app.post("/api/add_word")
async def add_word(req: AddWordRequest):
    try:
        display_phrase = add_custom_phrase(req.phrase)
        initialize_or_reload_model()
        stats = get_dataset_statistics()
        return JSONResponse({
            "status": "success",
            "phrase": display_phrase,
            "vocabulary": stats["vocabulary"],
            "total_classes": stats["vocabulary_size"]
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/record_sample")
async def record_sample(req: RecordSampleRequest):
    phrase = req.word.strip().upper()
    try:
        folder = clean_folder_name(phrase)
        folder_path = os.path.join(DATASET_DIR, folder)
        # Count real user recordings only
        existing_real = [f for f in os.listdir(folder_path) if f.startswith("sample_") and f.endswith(".npy")] if os.path.exists(folder_path) else []

        if len(existing_real) >= REQUIRED_SAMPLES_PER_WORD:
            return JSONResponse({
                "status": "error",
                "message": f"Already have {REQUIRED_SAMPLES_PER_WORD} recordings for '{phrase}'. Clear and re-record if needed.",
                "current_count": len(existing_real)
            }, status_code=400)

        saved_path = save_user_sample(phrase, req.tensor)
        real_after = len(existing_real) + 1
        stats = get_dataset_statistics()

        return JSONResponse({
            "status": "success",
            "message": f"Sample {real_after}/{REQUIRED_SAMPLES_PER_WORD} recorded for '{phrase}'",
            "word": phrase,
            "current_count": real_after,
            "required": REQUIRED_SAMPLES_PER_WORD,
            "complete": real_after >= REQUIRED_SAMPLES_PER_WORD,
            "total_samples": stats["total_samples"]
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/clear_word_samples")
async def clear_word_samples(req: ClearWordRequest):
    """Clears only the real user-recorded samples (sample_*.npy) for a word. Keeps baseline files."""
    phrase = req.word.strip().upper()
    folder = clean_folder_name(phrase)
    folder_path = os.path.join(DATASET_DIR, folder)
    try:
        deleted = 0
        if os.path.exists(folder_path):
            for f in os.listdir(folder_path):
                if f.startswith("sample_") and f.endswith(".npy"):
                    os.remove(os.path.join(folder_path, f))
                    deleted += 1
        return JSONResponse({
            "status": "success",
            "message": f"Cleared {deleted} recordings for '{phrase}'",
            "word": phrase,
            "deleted": deleted
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/train")
async def trigger_training():
    """Train the model for 100 epochs on all stored dataset samples."""
    if not HAS_TORCH or train_custom_100_epochs is None:
        return JSONResponse({
            "status": "error",
            "message": "Training requires PyTorch. Run: pip install torch torchvision. Then retry."
        }, status_code=400)

    # Check sufficient data exists
    stats = get_dataset_statistics()
    if stats["total_samples"] < 20:
        return JSONResponse({
            "status": "error",
            "message": f"Not enough data to train. Only {stats['total_samples']} samples found. Record at least 10 samples per word."
        }, status_code=400)

    try:
        result = train_custom_100_epochs(epochs=100)
        initialize_or_reload_model()
        return JSONResponse({"status": "success", "training_result": result})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/health")
async def health():
    return {
        "status": "online",
        "version": "4.0.0",
        "model": "loaded" if (numpy_model or torch_model) else "none",
        "classes": len(active_vocabulary),
        "has_torch": HAS_TORCH
    }


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 65)
    print("  SILENTVOICE AI 4.0 — Neural Silent Speech Interface")
    print("  http://127.0.0.1:8000")
    print("=" * 65 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
