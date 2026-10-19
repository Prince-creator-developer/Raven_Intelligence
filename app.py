import os
import sys
import json
import time
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
    clear_phrase_samples,
    reset_entire_dataset,
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
    title="SilentVoice AI • Neural Silent Speech Interface",
    description="Samsung Innovation Campus (SIC-AI) • Dynamic Articulatory Kinematics",
    version="4.1.0"
)

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

DEVICE = torch.device("cpu") if HAS_TORCH else None
WEIGHTS_PATH = os.path.join(MODELS_DIR, "lip_model.pth")
NPZ_WEIGHTS_PATH = os.path.join(MODELS_DIR, "lip_model_weights.npz")

active_vocabulary: List[str] = get_dynamic_vocabulary()
torch_model = None
numpy_model: Optional[NumpyVisemeNet] = None
REQUIRED_SAMPLES_PER_WORD = 10


def initialize_or_reload_model():
    global torch_model, numpy_model, active_vocabulary
    active_vocabulary = get_dynamic_vocabulary()
    num_classes = len(active_vocabulary)

    # 1. Pure NumPy model (ideal for serverless Vercel, zero CUDA bloat)
    if os.path.exists(NPZ_WEIGHTS_PATH):
        try:
            numpy_model = NumpyVisemeNet(NPZ_WEIGHTS_PATH)
            print(f"[NUMPY INFERENCE] Loaded weights for {num_classes} classes from {NPZ_WEIGHTS_PATH}")
        except Exception as e:
            print(f"[NUMPY WARNING] Could not load .npz weights: {e}")

    # 2. PyTorch model (for local VS Code training and inference)
    if HAS_TORCH and VisemeTemporalNet is not None:
        try:
            torch_model = VisemeTemporalNet(input_dim=12, hidden_dim=64, num_classes=num_classes).to(DEVICE)
            if os.path.exists(WEIGHTS_PATH):
                state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
                if state_dict["fc2.weight"].shape[0] == num_classes:
                    torch_model.load_state_dict(state_dict)
                    print(f"[TORCH LOADED] Loaded PyTorch weights for {num_classes} classes")
            torch_model.eval()
        except Exception as e:
            print(f"[TORCH WARNING] Model init: {e}")


initialize_or_reload_model()


# ──────────────────────────────────────────────
# REQUEST SCHEMAS
# ──────────────────────────────────────────────
class TemporalTensorRequest(BaseModel):
    tensor: List[List[float]]

class RecordSampleRequest(BaseModel):
    word: str
    tensor: List[List[float]]
    sample_num: Optional[int] = None

class AddWordRequest(BaseModel):
    phrase: str

class ClearWordRequest(BaseModel):
    word: str


# ──────────────────────────────────────────────
# TEMPLATE LOADER (Safe for local & Vercel)
# ──────────────────────────────────────────────
import jinja2

def get_template_html() -> str:
    candidates = [
        os.path.join(BASE_DIR, "templates", "index.html"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "index.html"),
        os.path.join(os.getcwd(), "templates", "index.html"),
        "templates/index.html",
        os.path.join("/var/task", "templates", "index.html")
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                with open(c, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                pass
    return ""


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
        return HTMLResponse(content=f"<html><body><h1>SilentVoice AI</h1><p>Online</p><p>Notice: {e}</p></body></html>", status_code=200)


# ──────────────────────────────────────────────
# API: INFERENCE (Handles both /api/predict and /api/predict_silent_speech)
# ──────────────────────────────────────────────
@app.post("/api/predict")
@app.post("/api/predict_silent_speech")
async def predict_silent_speech(req: TemporalTensorRequest):
    t0 = time.perf_counter()
    data = np.array(req.tensor, dtype=np.float32)

    if data.shape != (30, 12):
        return JSONResponse({"status": "error", "message": f"Expected shape (30, 12), got {data.shape}"}, status_code=400)

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
        return JSONResponse({"status": "error", "message": "No model loaded for inference."}, status_code=500)

    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    pred_word = active_vocabulary[pred_idx] if pred_idx < len(active_vocabulary) else "RESTING"

    inference_latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    agent_output = agent_synthesizer.process_decoded_word(pred_word, confidence)
    prob_dict = {active_vocabulary[i]: round(float(probs[i]), 4) for i in range(min(len(active_vocabulary), len(probs)))}

    return JSONResponse({
        "status": "success",
        "predicted_word": pred_word,
        "confidence": round(confidence, 4),
        "synthesized_sentence": agent_output.get("synthesized_sentence") or pred_word,
        "ai_dialogue_reply": agent_output.get("ai_dialogue_reply", ""),
        "suggested_next_words": agent_output.get("suggested_next_words", []),
        "continuous_sentence": agent_output.get("continuous_sentence", ""),
        "patient_name": agent_output.get("patient_name", "Patient"),
        "conversation_turn": agent_output.get("conversation_turn", 0),
        "is_active_speech": agent_output["is_active_speech"],
        "compound_detected": agent_output["compound_detected"],
        "probabilities": prob_dict,
        "latency_ms": inference_latency_ms,
        "attention_weights": attn_list
    })


# ──────────────────────────────────────────────
# API: DATASET & RECORDING STUDIO
# ──────────────────────────────────────────────
@app.get("/api/dataset_stats")
async def dataset_stats():
    stats = get_dataset_statistics()
    stats["required_per_word"] = REQUIRED_SAMPLES_PER_WORD
    return JSONResponse(stats)


@app.post("/api/record_sample")
async def record_sample(req: RecordSampleRequest):
    phrase = req.word.strip().upper()
    try:
        folder = clean_folder_name(phrase)
        folder_path = os.path.join(DATASET_DIR, folder)
        existing = [f for f in os.listdir(folder_path) if f.endswith(".npy")] if os.path.exists(folder_path) else []
        current_count = len(existing)

        if current_count >= REQUIRED_SAMPLES_PER_WORD:
            return JSONResponse({
                "status": "error",
                "message": f"Already have {REQUIRED_SAMPLES_PER_WORD} recordings for '{phrase}'. Use Clear to re-record.",
                "current_count": current_count
            }, status_code=400)

        sample_num = req.sample_num or (current_count + 1)
        saved_path = save_user_sample(phrase, req.tensor, sample_num=sample_num)
        new_count = current_count + 1
        stats = get_dataset_statistics()

        return JSONResponse({
            "status": "success",
            "message": f"Saved sample {new_count}/{REQUIRED_SAMPLES_PER_WORD} for '{phrase}'",
            "word": phrase,
            "sample_num": new_count,
            "current_count": new_count,
            "required": REQUIRED_SAMPLES_PER_WORD,
            "complete": new_count >= REQUIRED_SAMPLES_PER_WORD,
            "total_samples": stats["total_samples"],
            "saved_path": saved_path
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/clear_word_samples")
async def clear_word_samples(req: ClearWordRequest):
    phrase = req.word.strip().upper()
    try:
        deleted = clear_phrase_samples(phrase)
        stats = get_dataset_statistics()
        return JSONResponse({
            "status": "success",
            "message": f"Emptied {deleted} sample(s) for '{phrase}'",
            "word": phrase,
            "deleted": deleted,
            "current_count": 0,
            "total_samples": stats["total_samples"]
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/reset_all_dataset")
async def reset_all_dataset():
    """Wipes all sample recordings from every word folder in dataset/."""
    try:
        deleted = reset_entire_dataset()
        stats = get_dataset_statistics()
        return JSONResponse({
            "status": "success",
            "message": f"Successfully emptied all trained boxes! Wiped {deleted} sample(s).",
            "deleted_count": deleted,
            "dataset_stats": stats
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/add_word")
@app.post("/api/add_custom_word")
async def add_word(req: AddWordRequest):
    try:
        display_phrase = add_custom_phrase(req.phrase)
        initialize_or_reload_model()
        stats = get_dataset_statistics()
        return JSONResponse({
            "status": "success",
            "message": f"Created clean training box for '{display_phrase}'",
            "phrase": display_phrase,
            "vocabulary": stats["vocabulary"],
            "total_classes": stats["vocabulary_size"]
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/train")
@app.post("/api/train_100_epochs")
async def trigger_training():
    """Train the model for 100 epochs on all stored dataset samples."""
    if not HAS_TORCH or train_custom_100_epochs is None:
        return JSONResponse({
            "status": "error",
            "message": "Local 100-Epoch training requires PyTorch. Please install PyTorch locally (pip install torch torchvision) or run 'python models/train_custom.py'."
        }, status_code=400)

    try:
        train_result = train_custom_100_epochs(epochs=100)
        initialize_or_reload_model()
        return JSONResponse({
            "status": "success",
            "training_result": train_result
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/health")
async def health():
    return {
        "status": "online",
        "system": "SilentVoice AI",
        "version": "4.1.0",
        "classes": len(active_vocabulary),
        "has_torch": HAS_TORCH
    }


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 65)
    print("  SILENTVOICE AI 4.1 • Neural Silent Speech Interface")
    print("  Local Web App: http://127.0.0.1:8000")
    print("=" * 65 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
