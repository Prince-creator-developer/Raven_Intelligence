<div align="center">

# 🎙️ SilentVoice AI
### *Neural Silent Speech Interface & Open-Vocabulary Visual Speech Recognition*

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-WebGL_GPU-00C7B7?style=for-the-badge&logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Latency](https://img.shields.io/badge/Latency-%3C15ms_BiLSTM-00F2FE?style=for-the-badge&logo=speedtest&logoColor=black)](https://github.com/Prince-creator-developer/Raven_Intelligence)
[![Training](https://img.shields.io/badge/Training-100_Epochs_Local-A855F7?style=for-the-badge)](https://github.com/Prince-creator-developer/Raven_Intelligence)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

<br/>

**Synthesizing Audible Natural Speech Purely from Camera-Tracked Silent Lip Kinematics in Real Time**

[📖 VS Code Guide](VSCODE_GUIDE.md) • [⚡ Quick Start](#-quick-start) • [🧠 Neural Architecture](#-neural-architecture) • [🎯 Custom Studio & 100 Epochs](#-open-vocabulary-studio--100-epoch-training)

</div>

---

## 📌 Overview

**SilentVoice AI** is an assistive silent-speech neural interface engineered to convert silent, soundless mouth and lip movements into fluent, audible human speech in real time. Designed for patients with aphonia, ALS, laryngectomy, tracheostomy, ICU intubation, or individuals in covert silent environments.

Unlike traditional closed-vocabulary models limited to a fixed set of pre-programmed words, **SilentVoice AI introduces an Open-Vocabulary Dynamic Directory Architecture**:
* **Any Word or Full Sentence:** Users can type and train any custom words, medical phrases, or complete sentences (e.g. `"I WANT TO GO HOME"`, `"PLEASE CALL THE DOCTOR"`, `"HELLO WORLD"`).
* **Personal Movement Studio:** Records real mouth kinematics directly into dedicated directories: `dataset/<PHRASE>/sample_<timestamp>.npy`.
* **100-Epoch Local Training Engine:** Trains a PyTorch BiLSTM + Self-Attention model with biological augmentations (temporal jitter, scale variance, gaussian noise) and hot-reloads weights with zero server downtime.
* **Isotropic Pixel-Space Metric Normalization:** Computes lip aperture, width, and velocity independently of webcam aspect ratio or distance.
* **Agentic Speech Synthesizer:** Expands decoded articulatory visemes into natural sentences and speaks them aloud via Neural Text-to-Speech (`window.speechSynthesis`).

---

## 🧠 Neural Architecture

```
┌────────────────────────────────────────────────────────┐
│            VIDEO STREAM (WebGL MediaPipe GPU)          │
│          468 Facial Landmarks @ 30 - 60 FPS            │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│     12D ARTICULATORY KINEMATICS EXTRACTOR              │
│  • Vertical Aperture (Inner Upper 13 to Lower 14)      │
│  • Horizontal Stretch (Left Corner 78 to Right 308)   │
│  • Aperture Aspect Ratio (H / W)                       │
│  • Protrusion Depth Metric (Z-dispersion)              │
│  • 1st & 2nd Kinematic Derivatives (ΔH/Δt, Δ²H/Δt²)   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│         ROLLING TEMPORAL WINDOW: Tensor(30, 12)        │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│            VISEMETEMPORALNET (PyTorch)                 │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 1D Spatial Feature Convolution + BatchNorm1d     │  │
│  ├──────────────────────────────────────────────────┤  │
│  │ 2-Layer Bidirectional LSTM (Hidden: 64)          │  │
│  ├──────────────────────────────────────────────────┤  │
│  │ Scaled Dot-Product Temporal Self-Attention Head  │  │
│  ├──────────────────────────────────────────────────┤  │
│  │ Dynamic Open-Vocabulary Softmax Head (N-Classes) │  │
│  └──────────────────────────────────────────────────┘  │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│       AGENTIC BIONIC CONTEXTUAL SYNTHESIZER            │
│  • Temporal Hysteresis & Viseme Compound Chaining      │
│  • Full Sentence Neural Speech Synthesis (Audible TTS) │
└────────────────────────────────────────────────────────┘
```

---

## 🎯 Open-Vocabulary Studio & 100-Epoch Training

SilentVoice AI stores physical lip movement arrays into disk-backed subdirectories:

```text
silent_voice/
  └── dataset/
        ├── RESTING/              # Baseline resting posture
        ├── HELP/                 # "I need immediate assistance, please help me."
        ├── WATER/                # "Could I please have a glass of water?"
        ├── DOCTOR/               # "Please call the doctor for an evaluation."
        ├── I_WANT_TO_GO_HOME/    # <-- User-added custom sentence!
        └── PLEASE_CALL_MY_FAMILY/# <-- Any arbitrary phrase!
```

### Adding Any Word or Full Sentence
1. In the Web UI, locate the **🎯 Custom Movement Studio**.
2. Type any custom word or full sentence in the input box and click **➕ Add Phrase**.
3. A new directory `dataset/<YOUR_PHRASE>/` is automatically created, baseline seed files are generated, and the phrase is added to the real-time vocabulary heatmap.

### Recording Your Movement
1. Select your target phrase from the dropdown.
2. Click **⏺️ Record 1s Movement** and mouth the phrase in front of the camera.
3. The raw 30-frame $30 \times 12$ kinematic trajectory is automatically stored as:
   ```text
   dataset/<PHRASE>/sample_<timestamp>.npy
   ```

### 100-Epoch Training Engine
Click **⚡ Train Model on Stored Movements (100 Epochs)** in the UI, or run in the terminal:
```bash
python models/train_custom.py
```
* Applies 6x biological augmentation (temporal jitter, scale, noise).
* Runs 100 epochs using `AdamW` + `CosineAnnealingLR`.
* Saves weights to `models/lip_model.pth` and hot-reloads into active memory.

---

## 🚀 Quick Start

### 1. Clone & Navigate
```bash
git clone https://github.com/Prince-creator-developer/Raven_Intelligence.git
cd Raven_Intelligence
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch SilentVoice AI
```bash
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
Open your browser at:
👉 **`http://localhost:8000`**

---

## 💻 Running in Visual Studio Code (VS Code)

For detailed step-by-step instructions on running, debugging, and training inside VS Code, see the [**VS Code Manual (`VSCODE_GUIDE.md`)**](VSCODE_GUIDE.md).

```powershell
# Open the project folder in VS Code
code .

# Open integrated terminal (Ctrl + ~)
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

---

## 📡 API Reference

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/` | `GET` | Interactive Visual Speech HUD & Custom Studio |
| `/health` | `GET` | System telemetry, model status, and active class count |
| `/api/predict_silent_speech` | `POST` | Accepts a $30 \times 12$ tensor, returns decoded word/sentence and probabilities |
| `/api/add_custom_word` | `POST` | Creates a new class folder in `dataset/` for any custom word or full sentence |
| `/api/record_sample` | `POST` | Saves a recorded 30-frame movement array into `dataset/<PHRASE>/` |
| `/api/train_100_epochs` | `POST` | Triggers 100-epoch local PyTorch training and hot-reloads model weights |
| `/api/dataset_stats` | `GET` | Returns stored sample counts per word/sentence across all directories |

---

## 📁 Repository Structure

```text
SilentVoice/
├── app.py                      # FastAPI server with dynamic model reloading & API routes
├── requirements.txt            # System dependencies (PyTorch, FastAPI, NumPy, SciPy)
├── VSCODE_GUIDE.md             # Complete step-by-step VS Code & local training guide
├── LICENSE                     # MIT License
├── README.md                   # System documentation & architecture
├── dataset/                    # Stored movement files (.npy) per word/sentence
│   ├── RESTING/
│   ├── HELP/
│   ├── WATER/
│   ├── DOCTOR/
│   └── I_WANT_WATER/           # Custom phrase directories
├── models/
│   ├── lip_model.py            # VisemeTemporalNet PyTorch architecture (BiLSTM + Attention)
│   ├── lip_model.pth           # 100-epoch trained neural network weights
│   ├── vocab.json              # Dynamic vocabulary registry
│   ├── dataset_manager.py      # Dynamic directory scanner & dataset loader
│   ├── train_custom.py         # 100-epoch local training engine
│   └── agent_synthesizer.py    # Agentic Context & Grammar Reconstruction
├── static/
│   ├── css/style.css           # Cyber-bionic dark HUD styling
│   ├── js/mediapipe_lips.js    # MediaPipe FaceMesh GPU tracking & 12D metric extraction
│   └── js/app.js               # Client orchestrator, visualizer & studio manager
└── templates/
    └── index.html              # Real-time HUD, billboard, telemetry & movement studio
```

---

## 👨‍💻 Author

**Prince Kumar**  
GitHub: [@Prince-creator-developer](https://github.com/Prince-creator-developer)

---

## ⚖️ License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more details.
