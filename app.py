import os
import sys
import json
import time
from typing import Dict, List, Optional
import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
sys.path.append(MODELS_DIR)

from lip_model import VisemeTemporalNet
from agent_synthesizer import agent_synthesizer
from dataset_manager import (
    save_user_sample,
    get_dataset_statistics,
    get_dynamic_vocabulary,
    add_custom_phrase,
    VOCAB_FILE
)
from train_custom import train_custom_100_epochs

app = FastAPI(
    title="SilentVoice AI • Dynamic Neural Speech Interface",
    description="Arbitrary Word & Full Sentence Lip-Reading for Samsung Innovation Campus",
    version="3.0.0"
)

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

DEVICE = torch.device("cpu")
WEIGHTS_PATH = os.path.join(MODELS_DIR, "lip_model.pth")

# Dynamic Vocabulary state
active_vocabulary: List[str] = get_dynamic_vocabulary()
model: Optional[VisemeTemporalNet] = None


def initialize_or_reload_model():
    global model, active_vocabulary
    active_vocabulary = get_dynamic_vocabulary()
    num_classes = len(active_vocabulary)

    model = VisemeTemporalNet(input_dim=12, hidden_dim=64, num_classes=num_classes).to(DEVICE)

    if os.path.exists(WEIGHTS_PATH):
        try:
            state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
            # Check if saved weights match current class count
            if state_dict["fc2.weight"].shape[0] == num_classes:
                model.load_state_dict(state_dict)
                print(f"[MODEL LOADED] Successfully loaded weights for {num_classes} classes from {WEIGHTS_PATH}")
            else:
                print(f"[NOTICE] Architecture changed to {num_classes} classes. Ready for 100-epoch training.")
        except Exception as e:
            print(f"[WARNING] Weight loading notice: {e}")
    else:
        print("[NOTICE] No saved weights found. Ready for initial training.")

    model.eval()


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

    input_tensor = torch.tensor(data).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probabilities, attn_weights = model(input_tensor)
        probs = probabilities[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        pred_word = active_vocabulary[pred_idx] if pred_idx < len(active_vocabulary) else "RESTING"

    inference_latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    agent_output = agent_synthesizer.process_decoded_word(pred_word, confidence)
    prob_dict = {active_vocabulary[i]: round(float(probs[i]), 4) for i in range(len(active_vocabulary))}

    return JSONResponse({
        "status": "success",
        "predicted_word": pred_word,
        "confidence": round(confidence, 4),
        "synthesized_sentence": agent_output["synthesized_sentence"] or pred_word,
        "is_active_speech": agent_output["is_active_speech"],
        "compound_detected": agent_output["compound_detected"],
        "probabilities": prob_dict,
        "latency_ms": inference_latency_ms,
        "attention_weights": attn_weights[0].squeeze().cpu().numpy().tolist()
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
