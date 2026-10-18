# 🎙️ SilentVoice AI — VS Code & Local Training Manual

Welcome to **SilentVoice AI v3.0**, a neural silent-speech interface that reads silent lip movements in real time using **MediaPipe FaceMesh GPU tracking**, a **PyTorch BiLSTM + Self-Attention Neural Network**, and an **Agentic Speech Synthesizer**.

---

## 🚀 1. How to Open and Run in VS Code

### Step 1: Open VS Code
1. Launch **Visual Studio Code**.
2. Click **File** ➔ **Open Folder...** (or press `Ctrl + K`, `Ctrl + O`).
3. Select the folder:
   ```text
   C:\Users\Prince kumar\OneDrive\Desktop\dsih\silent_voice
   ```

### Step 2: Open Integrated Terminal
1. Press `Ctrl + ~` (tilde) or go to **Terminal** ➔ **New Terminal**.
2. Verify you are inside the `silent_voice` directory:
   ```powershell
   pwd
   # Output should end with \silent_voice
   ```

### Step 3: Run the Application
In the VS Code terminal, execute:
```powershell
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
Open your browser and navigate to:
👉 **`http://localhost:8000`**

---

## 🔤 2. How to Add ANY Custom Word or Full Sentence

SilentVoice AI is built with an **Open-Vocabulary Dynamic Directory Architecture**. You are **not limited to a fixed set of words**. You can add single words, medical terms, emergency phrases, or complete sentences (e.g., `"I WANT TO GO HOME"`, `"PLEASE CALL MY FAMILY"`, `"WATER PLEASE"`).

### Method A: From the Browser UI (Easiest)
1. On the web app, find the **🎯 Custom Movement Studio** panel.
2. Type any custom word or full sentence into the input box:
   ```text
   I WANT TO GO HOME
   ```
3. Click **➕ Add Phrase** (or press `Enter`).
4. SilentVoice AI will automatically:
   - Create a dedicated folder: `dataset/I_WANT_TO_GO_HOME/`
   - Seed baseline kinematic trajectories
   - Register the phrase into `models/vocab.json`
   - Add it to the real-time probability heatmap

### Method B: From VS Code / File Explorer
You can also manually create a folder inside `dataset/`:
```text
silent_voice/
  └── dataset/
        ├── RESTING/
        ├── HELP/
        ├── WATER/
        ├── I_WANT_TO_GO_HOME/       <-- Your new folder!
        └── PLEASE_CALL_DOCTOR/      <-- Another custom sentence!
```
SilentVoice AI dynamically discovers all folders in `dataset/` on startup and during training.

---

## 📹 3. How to Record Your Actual Lip Movements

To train the model on your personal mouth mechanics:
1. Click **⚡ Engage Bionic Camera** to activate your webcam.
2. Look directly into the camera with a relaxed, closed mouth and click **🎯 Calibrate Neutral Mouth**.
3. In the **Record Movement** dropdown, select the word or sentence you want to practice.
4. Click **⏺️ Record 1s Movement**:
   - You have a 1-second preparation buffer.
   - When the red countdown appears (**Mouth Now! 30..20..10..0**), mouth the word or sentence silently.
   - The app captures a $30 \times 12$ float matrix of your genuine lip kinematics and sends it to `POST /api/record_sample`.
   - The sample is automatically saved as:
     ```text
     dataset/<PHRASE>/sample_<timestamp>.npy
     ```
5. Record 3 to 10 repetitions for each custom phrase for optimal accuracy.

---

## ⚡ 4. How to Train Locally for 100 Epochs

### Option A: From the Browser UI (1-Click Hot-Reload)
1. In the **🎯 Custom Movement Studio**, click:
   ```text
   [⚡ Train Model on Stored Movements (100 Epochs)]
   ```
2. The server executes 100 epochs on your CPU/GPU, displays the validation accuracy, saves the new weights to `models/lip_model.pth`, and **hot-reloads the model into memory with zero downtime**.

### Option B: From the VS Code Terminal
1. In your VS Code terminal, run:
   ```powershell
   python models/train_custom.py
   ```
2. You will see the training progress printed every 10 epochs:
   ```text
   =================================================================
     [TRAINING ENGINE] RUNNING 100 EPOCHS ON LOCAL DATASET
   =================================================================
   Dynamic Vocabulary Size: 13 classes
   Active Classes: RESTING, BREATHE, DOCTOR, HELP, I WANT WATER, ...
   Loaded 185 recorded kinematic files across 13 directories.
   Total augmented training tensors: 1295
     Epoch 010/100 | Loss: 0.1250 | Train Acc: 96.2% | Val Acc: 98.1%
     Epoch 020/100 | Loss: 0.0420 | Train Acc: 98.9% | Val Acc: 99.4%
     ...
     Epoch 100/100 | Loss: 0.0018 | Train Acc: 100.0% | Val Acc: 100.0%
   =================================================================
     Training Complete! Model saved to: models/lip_model.pth
   ```

---

## 🧠 5. How Inference & Speech Synthesis Works

1. **MediaPipe FaceMesh (GPU):** 468 facial landmarks tracked at 30–60 FPS in browser WebGL.
2. **Kinematic Feature Extraction:** Computes true isotropic physical metrics (vertical aperture, lip spread, aspect ratio, protrusion, velocity $\frac{\Delta h}{\Delta t}$, acceleration $\frac{\Delta^2 h}{\Delta t^2}$).
3. **BiLSTM + Temporal Attention:** PyTorch model processes a rolling 30-frame window ($30 \times 12$) in under **15 milliseconds**.
4. **Agentic Sentence Expansion & TTS:**
   - Single words are expanded into grammatically natural sentences.
   - Full sentences (e.g. `"I WANT TO GO HOME"`) are spoken aloud directly via the browser's Neural Speech Synthesis (`window.speechSynthesis`).
