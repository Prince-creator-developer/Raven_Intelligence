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
    get_dataset_statistics,
    get_dynamic_vocabulary,
    add_custom_phrase,
    VOCAB_FILE
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
    title="SilentVoice AI • Dynamic Neural Speech Interface",
    description="Arbitrary Word & Full Sentence Lip-Reading for Samsung Innovation Campus",
    version="3.0.0"
)

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

DEVICE = torch.device("cpu") if HAS_TORCH else None
WEIGHTS_PATH = os.path.join(MODELS_DIR, "lip_model.pth")
NPZ_WEIGHTS_PATH = os.path.join(MODELS_DIR, "lip_model_weights.npz")

# Dynamic Vocabulary state
active_vocabulary: List[str] = get_dynamic_vocabulary()
torch_model = None
numpy_model: Optional[NumpyVisemeNet] = None


def initialize_or_reload_model():
    global torch_model, numpy_model, active_vocabulary
    active_vocabulary = get_dynamic_vocabulary()
    num_classes = len(active_vocabulary)

    # 1. Load ultra-lightweight pure NumPy model (Ideal for Vercel Serverless, 0 CUDA bloat)
    if os.path.exists(NPZ_WEIGHTS_PATH):
        try:
            numpy_model = NumpyVisemeNet(NPZ_WEIGHTS_PATH)
            print(f"[NUMPY INFERENCE] Loaded {num_classes} classes from {NPZ_WEIGHTS_PATH}")
        except Exception as e:
            print(f"[NUMPY WARNING] Could not load .npz weights: {e}")

    # 2. Load PyTorch model if torch is available (for local VS Code training & testing)
    if HAS_TORCH and VisemeTemporalNet is not None:
        try:
            torch_model = VisemeTemporalNet(input_dim=12, hidden_dim=64, num_classes=num_classes).to(DEVICE)
            if os.path.exists(WEIGHTS_PATH):
                state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
                if state_dict["fc2.weight"].shape[0] == num_classes:
                    torch_model.load_state_dict(state_dict)
                    print(f"[TORCH LOADED] Loaded weights for {num_classes} classes from {WEIGHTS_PATH}")
            torch_model.eval()
        except Exception as e:
            print(f"[TORCH WARNING] Torch model initialization: {e}")


initialize_or_reload_model()


class TemporalTensorRequest(BaseModel):
    tensor: List[List[float]]


class RecordSampleRequest(BaseModel):
    word: str
    tensor: List[List[float]]


class AddWordRequest(BaseModel):
    phrase: str


@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    stats = get_dataset_statistics()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "vocabulary": stats["vocabulary"],
        "dataset_stats": stats
    })


@app.post("/api/predict_silent_speech")
async def predict_silent_speech(req: TemporalTensorRequest):
    """
    Decodes real-time silent lip movement across all registered words/sentences.
    """
    t0 = time.perf_counter()
    data = np.array(req.tensor, dtype=np.float32)

    if data.shape != (30, 12):
        return JSONResponse({"status": "error", "message": f"Expected shape (30, 12), got {data.shape}"}, status_code=400)

    if numpy_model is not None:
        probs, attn_weights_raw = numpy_model.predict(data)
        attn_list = attn_weights_raw.tolist() if hasattr(attn_weights_raw, "tolist") else list(attn_weights_raw)
    elif HAS_TORCH and torch_model is not None:
        input_tensor = torch.tensor(data).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probabilities, attn_weights = torch_model(input_tensor)
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
        "synthesized_sentence": agent_output["synthesized_sentence"] or pred_word,
        "is_active_speech": agent_output["is_active_speech"],
        "compound_detected": agent_output["compound_detected"],
        "probabilities": prob_dict,
        "latency_ms": inference_latency_ms,
        "attention_weights": attn_list
    })


@app.post("/api/add_custom_word")
async def add_custom_word(req: AddWordRequest):
    """
    Creates a new folder in dataset/ for ANY word or full sentence.
    Example: 'I WANT TO GO HOME' -> dataset/I_WANT_TO_GO_HOME/
    """
    try:
        display_phrase = add_custom_phrase(req.phrase)
        initialize_or_reload_model()
        stats = get_dataset_statistics()
        return JSONResponse({
            "status": "success",
            "message": f"Created new class directory for '{display_phrase}'",
            "phrase": display_phrase,
            "vocabulary": stats["vocabulary"],
            "total_classes": stats["vocabulary_size"]
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/record_sample")
async def record_sample(req: RecordSampleRequest):
    """
    Saves a captured 30x12 kinematic trajectory into dataset/<PHRASE>/sample_<timestamp>.npy
    """
    phrase = req.word.strip().upper()
    try:
        saved_path = save_user_sample(phrase, req.tensor)
        stats = get_dataset_statistics()
        return JSONResponse({
            "status": "success",
            "message": f"Stored movement sample in dataset/{phrase.replace(' ', '_')}",
            "file_path": saved_path,
            "word_count": stats["per_word_counts"].get(phrase, 0),
            "total_samples": stats["total_samples"]
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/api/dataset_stats")
async def dataset_stats():
    return JSONResponse(get_dataset_statistics())


@app.post("/api/train_100_epochs")
async def trigger_training():
    """
    Trains the neural network for 100 epochs on ALL folders in dataset/ and reloads model.
    """
    if not HAS_TORCH or train_custom_100_epochs is None:
        return JSONResponse({
            "status": "error",
            "message": "Local 100-Epoch training requires PyTorch. Please run training locally in VS Code with 'python models/train_custom.py' (or 'pip install -r requirements-train.txt'). On Vercel serverless functions, execution is optimized for sub-10ms pure NumPy inference."
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
    return {"status": "online", "system": "SilentVoice AI", "version": "3.0.0", "classes": len(active_vocabulary)}


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 65)
    print("  [OPEN-VOCABULARY] SILENTVOICE AI 3.0 RUNNING")
    print("  Access Live Web-App: http://127.0.0.1:8000")
    print("=" * 65 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
