/**
 * SILENTVOICE AI • CLIENT APPLICATION ORCHESTRATOR V2.5
 * MediaPipe GPU Tracking, Real-time Visual Speech Recognition, and Personal Movement Studio
 */

document.addEventListener('DOMContentLoaded', () => {
  const extractor = new window.LipKinematicsExtractor();

  // Elements
  const videoEl = document.getElementById('cameraFeed');
  const canvasEl = document.getElementById('meshCanvas');
  const ctx = canvasEl.getContext('2d');

  const btnCamera = document.getElementById('btnCamera');
  const btnCalibrate = document.getElementById('btnCalibrate');
  const btnVoiceToggle = document.getElementById('btnVoiceToggle');

  const billboardWordEl = document.getElementById('billboardWord');
  const confidenceBadgeEl = document.getElementById('confidenceBadge');
  const synthesizedTextEl = document.getElementById('synthesizedText');
  const billboardCardEl = document.getElementById('billboardCard');

  const kpiLatencyEl = document.getElementById('kpiLatency');
  const kpiApertureEl = document.getElementById('kpiAperture');
  const kpiVelocityEl = document.getElementById('kpiVelocity');
  const kpiFpsEl = document.getElementById('kpiFps');
  const historyListEl = document.getElementById('historyList');

  // Standby & Simulation Elements
  const standbyOverlay = document.getElementById('standbyOverlay');
  const btnSimulate = document.getElementById('btnSimulate');
  const btnQuickDemo = document.getElementById('btnQuickDemo');

  // Studio Elements
  const customPhraseInput = document.getElementById('customPhraseInput');
  const btnAddPhrase = document.getElementById('btnAddPhrase');
  const recordWordSelect = document.getElementById('recordWordSelect');
  const btnRecordSample = document.getElementById('btnRecordSample');
  const btnTrain100 = document.getElementById('btnTrain100');
  const trainStatusMsg = document.getElementById('trainStatusMsg');
  const recordingIndicator = document.getElementById('recordingIndicator');
  const recCountdown = document.getElementById('recCountdown');
  const totalDatasetCount = document.getElementById('totalDatasetCount');

  let isCameraActive = false;
  let isVoiceEnabled = true;
  let cameraInstance = null;
  let localMediaStream = null;
  let faceMesh = null;
  let lastPredictTime = 0;
  let lastSpokenSentence = "";
  let frameCount = 0;
  let lastFpsTime = performance.now();
  let latestLandmarks = null;

  const LIP_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61];
  const LIP_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78];

  function resizeCanvas() {
    canvasEl.width = videoEl.clientWidth || 560;
    canvasEl.height = videoEl.clientHeight || 420;
  }
  window.addEventListener('resize', resizeCanvas);

  // --------------------------------------------------------------------------
  // BIONIC TEXT-TO-SPEECH
  // --------------------------------------------------------------------------
  function speakSentence(text) {
    if (!isVoiceEnabled || !('speechSynthesis' in window)) return;
    if (!text || text === lastSpokenSentence) return;

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.05;
    utterance.pitch = 1.0;
    utterance.lang = 'en-US';

    const voices = window.speechSynthesis.getVoices();
    const naturalVoice = voices.find(v => v.lang.startsWith('en') && (v.name.includes('Natural') || v.name.includes('Google') || v.name.includes('Samantha')));
    if (naturalVoice) utterance.voice = naturalVoice;

    window.speechSynthesis.speak(utterance);
    lastSpokenSentence = text;
  }

  // --------------------------------------------------------------------------
  // DRAW BIONIC WIREFRAME MESH
  // --------------------------------------------------------------------------
  function drawBionicLipMesh(landmarks) {
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    if (!landmarks) return;

    const w = canvasEl.width;
    const h = canvasEl.height;

    // Draw Outer Lip Contour
    ctx.beginPath();
    for (let i = 0; i < LIP_OUTER.length; i++) {
      const pt = landmarks[LIP_OUTER[i]];
      const x = pt.x * w;
      const y = pt.y * h;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = extractor.isRecording ? '#ef4444' : '#00f2fe';
    ctx.lineWidth = 2.5;
    ctx.shadowColor = extractor.isRecording ? '#ef4444' : '#00f2fe';
    ctx.shadowBlur = 12;
    ctx.stroke();

    // Draw Inner Lip Contour
    ctx.beginPath();
    for (let i = 0; i < LIP_INNER.length; i++) {
      const pt = landmarks[LIP_INNER[i]];
      const x = pt.x * w;
      const y = pt.y * h;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = '#a855f7';
    ctx.lineWidth = 2.0;
    ctx.shadowColor = '#a855f7';
    ctx.shadowBlur = 8;
    ctx.stroke();

    ctx.shadowBlur = 0;

    // Key Anchor Points
    [13, 14, 78, 308].forEach(idx => {
      const pt = landmarks[idx];
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(pt.x * w, pt.y * h, 3.5, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  // --------------------------------------------------------------------------
  // MEDIAPIPE FACE MESH RESULTS HANDLER
  // --------------------------------------------------------------------------
  function onFaceMeshResults(results) {
    frameCount++;
    const now = performance.now();
    if (now - lastFpsTime >= 1000) {
      kpiFpsEl.textContent = `${frameCount} FPS`;
      frameCount = 0;
      lastFpsTime = now;
    }

    if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
      ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
      return;
    }

    latestLandmarks = results.multiFaceLandmarks[0];
    drawBionicLipMesh(latestLandmarks);

    // Extract real physical features with true pixel-space metrics
    const feat = extractor.extractFrameFeatures(latestLandmarks, videoEl.videoWidth || 640, videoEl.videoHeight || 480);
    if (!feat) return;

    kpiApertureEl.textContent = `${(feat.metrics.vertical_aperture * 100).toFixed(1)}%`;
    kpiVelocityEl.textContent = `${feat.metrics.velocity.toFixed(3)}`;

    // Update Recording Countdown if recording
    if (extractor.isRecording) {
      recCountdown.textContent = `${30 - extractor.recordingFrames.length}`;
    }

    // Periodic Inference Loop (only when NOT recording training data)
    if (!extractor.isRecording && extractor.isBufferReady() && (now - lastPredictTime) > 180) {
      lastPredictTime = now;
      runInference(extractor.getTemporalTensor());
    }
  }

  // --------------------------------------------------------------------------
  // RUN INFERENCE
  // --------------------------------------------------------------------------
  async function runInference(tensor30x12) {
    try {
      const res = await fetch('/api/predict_silent_speech', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tensor: tensor30x12 })
      });
      const data = await res.json();
      if (data.status !== 'success') return;

      kpiLatencyEl.textContent = `${data.latency_ms}ms`;

      // Update Heatmap
      if (data.probabilities) {
        for (const [word, prob] of Object.entries(data.probabilities)) {
          const key = word.replace(/\s+/g, '');
          const barFill = document.getElementById(`bar-${key}`);
          const barVal = document.getElementById(`val-${key}`);
          if (barFill && barVal) {
            const pct = (prob * 100).toFixed(1);
            barFill.style.width = `${Math.max(pct, 2)}%`;
            barVal.textContent = `${pct}%`;
          }
        }
      }

      // Handle Active Speech
      if (data.is_active_speech && data.confidence > 0.65) {
        billboardCardEl.classList.add('active-speech');
        billboardWordEl.textContent = data.predicted_word;
        confidenceBadgeEl.textContent = `${(data.confidence * 100).toFixed(1)}% Conf`;
        synthesizedTextEl.textContent = `"${data.synthesized_sentence}"`;

        speakSentence(data.synthesized_sentence);
        addHistoryItem(data.predicted_word, data.synthesized_sentence, data.confidence);

      } else {
        billboardCardEl.classList.remove('active-speech');
        if (data.predicted_word === 'RESTING') {
          billboardWordEl.textContent = 'RESTING';
          confidenceBadgeEl.textContent = 'Awaiting Movement';
        }
      }

    } catch (err) {
      console.error('Inference error:', err);
    }
  }

  function addHistoryItem(word, sentence, conf) {
    const item = document.createElement('div');
    item.className = 'history-item';
    const timeStr = new Date().toTimeString().split(' ')[0];
    item.innerHTML = `
      <span><strong>${word}:</strong> "${sentence}"</span>
      <span style="color:#00f2fe">${(conf * 100).toFixed(0)}% • ${timeStr}</span>
    `;
    historyListEl.prepend(item);
    if (historyListEl.children.length > 8) {
      historyListEl.removeChild(historyListEl.lastChild);
    }
  }

  // --------------------------------------------------------------------------
  // CAMERA CONTROLS (NATIVE WEBRTC + MEDIAPIPE PIPELINE)
  // --------------------------------------------------------------------------
  btnCamera.addEventListener('click', async () => {
    if (!isCameraActive) {
      try {
        btnCamera.textContent = '⌛ Starting Camera...';

        // 1. Start Native WebRTC Camera Stream directly
        localMediaStream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
          audio: false
        });
        videoEl.srcObject = localMediaStream;
        await videoEl.play();

        if (standbyOverlay) standbyOverlay.style.display = 'none';
        resizeCanvas();

        // 2. Initialize MediaPipe FaceMesh
        faceMesh = new FaceMesh({
          locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
        });

        faceMesh.setOptions({
          maxNumFaces: 1,
          refineLandmarks: true,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5
        });

        faceMesh.onResults(onFaceMeshResults);

        // 3. Frame Processing Loop
        isCameraActive = true;
        btnCamera.textContent = '🛑 Stop Bionic Camera';
        btnCamera.classList.replace('btn-primary', 'btn-secondary');

        async function processVideoFrame() {
          if (!isCameraActive) return;
          if (videoEl.readyState >= 2 && faceMesh) {
            try {
              await faceMesh.send({ image: videoEl });
            } catch (e) {
              console.warn('FaceMesh frame warning:', e);
            }
          }
          requestAnimationFrame(processVideoFrame);
        }
        requestAnimationFrame(processVideoFrame);

        setTimeout(() => {
          if (latestLandmarks) extractor.calibrateNeutral(latestLandmarks, videoEl.videoWidth, videoEl.videoHeight);
        }, 1500);

      } catch (err) {
        alert('Webcam notice: ' + err.message + '\n\nPlease ensure camera access is allowed in your browser.');
        btnCamera.textContent = '⚡ Engage Bionic Camera';
        if (standbyOverlay) standbyOverlay.style.display = 'flex';
        isCameraActive = false;
      }
    } else {
      isCameraActive = false;
      if (localMediaStream) {
        localMediaStream.getTracks().forEach(track => track.stop());
        localMediaStream = null;
      }
      videoEl.srcObject = null;
      ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
      if (standbyOverlay) standbyOverlay.style.display = 'flex';
      btnCamera.textContent = '⚡ Engage Bionic Camera';
      btnCamera.classList.replace('btn-secondary', 'btn-primary');
    }
  });

  // --------------------------------------------------------------------------
  // LIVE SPEECH SIMULATION DEMO (IMMEDIATE VERIFICATION)
  // --------------------------------------------------------------------------
  async function runSimulationDemo() {
    const demoPhrases = ["WATER", "HELP", "THANK YOU", "DOCTOR", "YES"];
    const chosenPhrase = demoPhrases[Math.floor(Math.random() * demoPhrases.length)];

    if (standbyOverlay) standbyOverlay.style.display = 'none';
    resizeCanvas();

    // Synthesize 30-frame realistic articulatory trajectory
    const seqLen = 30;
    const trajectory = [];
    for (let t = 0; t < seqLen; t++) {
      const p = t / seqLen;
      const aperture = 0.05 + 0.38 * Math.sin(p * Math.PI);
      const width = 0.45 + 0.09 * Math.sin(p * Math.PI);
      const aspect = aperture / (width + 1e-5);
      const outerH = aperture * 1.5 + 0.08;
      const dh = Math.cos(p * Math.PI) * 0.4;
      const dw = Math.cos(p * Math.PI) * 0.1;
      const d2h = -Math.sin(p * Math.PI);

      trajectory.push([
        aperture, width, aspect, outerH, 0.05, 0.06,
        0.02, 0.08, dh, dw, d2h, 0.0
      ]);
    }

    // Animate synthetic lips on canvas over 1 second (30 FPS)
    let frame = 0;
    const animTimer = setInterval(() => {
      if (frame >= seqLen) {
        clearInterval(animTimer);
        return;
      }

      const h = trajectory[frame][0];
      const w = trajectory[frame][1];
      kpiApertureEl.textContent = `${(h * 100).toFixed(1)}%`;
      kpiVelocityEl.textContent = `${trajectory[frame][8].toFixed(3)}`;
      kpiFpsEl.textContent = `30 FPS`;

      // Draw cybernetic lip wireframe on canvas
      ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
      const cx = canvasEl.width / 2;
      const cy = canvasEl.height / 2;
      const rx = (w * canvasEl.width) / 2.2;
      const ry = Math.max(h * canvasEl.height * 1.3, 10);

      // Outer lip contour
      ctx.beginPath();
      ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
      ctx.strokeStyle = '#00f2fe';
      ctx.lineWidth = 3;
      ctx.shadowColor = '#00f2fe';
      ctx.shadowBlur = 14;
      ctx.stroke();

      // Inner lip contour
      ctx.beginPath();
      ctx.ellipse(cx, cy, rx * 0.65, Math.max(ry * 0.55, 5), 0, 0, Math.PI * 2);
      ctx.strokeStyle = '#a855f7';
      ctx.lineWidth = 2;
      ctx.shadowColor = '#a855f7';
      ctx.shadowBlur = 8;
      ctx.stroke();

      ctx.shadowBlur = 0;
      frame++;
    }, 33);

    // Run real-time neural inference & audio speech synthesis
    await runInference(trajectory);
  }

  if (btnSimulate) btnSimulate.addEventListener('click', runSimulationDemo);
  if (btnQuickDemo) btnQuickDemo.addEventListener('click', runSimulationDemo);

  btnCalibrate.addEventListener('click', () => {
    if (latestLandmarks) {
      extractor.calibrateNeutral(latestLandmarks, videoEl.videoWidth, videoEl.videoHeight);
      btnCalibrate.textContent = '✅ Calibrated!';
      setTimeout(() => btnCalibrate.textContent = '🎯 Calibrate Neutral Mouth', 1500);
    } else {
      alert('Face not detected. Start camera and look directly at lens first.');
    }
  });

  btnVoiceToggle.addEventListener('click', () => {
    isVoiceEnabled = !isVoiceEnabled;
    btnVoiceToggle.textContent = isVoiceEnabled ? '🔊 Audio Voice: ON (Audible TTS)' : '🔇 Audio Voice: MUTED';
    btnVoiceToggle.style.color = isVoiceEnabled ? 'var(--cyan-bionic)' : 'var(--text-dim)';
  });

  // --------------------------------------------------------------------------
  // ADD CUSTOM WORD OR FULL SENTENCE
  // --------------------------------------------------------------------------
  async function handleAddCustomPhrase() {
    const rawVal = customPhraseInput.value.trim();
    if (!rawVal) {
      alert('Please enter a word or sentence (e.g. "I WANT WATER" or "HELLO")');
      return;
    }

    btnAddPhrase.disabled = true;
    btnAddPhrase.textContent = '⏳ Adding...';

    try {
      const res = await fetch('/api/add_custom_word', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ phrase: rawVal })
      });
      const data = await res.json();

      if (data.status === 'success') {
        const phrase = data.phrase;
        const cleanDir = phrase.replace(/\s+/g, '_');
        const cleanKey = phrase.replace(/\s+/g, '');

        // Add to select if not existing
        let exists = false;
        for (let opt of recordWordSelect.options) {
          if (opt.value === phrase) {
            exists = true;
            opt.selected = true;
            break;
          }
        }
        if (!exists) {
          const newOpt = document.createElement('option');
          newOpt.value = phrase;
          newOpt.textContent = `Record "${phrase}"`;
          newOpt.selected = true;
          recordWordSelect.appendChild(newOpt);
        }

        // Add to live Viseme Bars Heatmap if not existing
        const visemeGrid = document.querySelector('.viseme-bars-grid');
        if (visemeGrid && !document.getElementById(`bar-${cleanKey}`)) {
          const barItem = document.createElement('div');
          barItem.className = 'viseme-bar-item';
          barItem.innerHTML = `
            <div class="viseme-name-row">
              <span>${phrase}</span>
              <span id="val-${cleanKey}" style="color: var(--cyan-bionic);">0.0%</span>
            </div>
            <div class="viseme-track">
              <div class="viseme-fill" id="bar-${cleanKey}"></div>
            </div>
            <div style="font-size: 0.65rem; color: #64748b; margin-top: 3px; font-family: var(--font-mono);">
              Stored: <span id="count-${cleanDir}">5</span> files
            </div>
          `;
          visemeGrid.appendChild(barItem);
        }

        customPhraseInput.value = '';
        if (totalDatasetCount) {
          totalDatasetCount.textContent = `${data.total_classes * 15} Files Tracked`;
        }

        alert(`✅ Created dedicated class directory for "${phrase}"!\n\nFolder: dataset/${cleanDir}/\nBaseline movement files seeded. You can now record your real mouth movements and train 100 epochs.`);
      } else {
        alert('Error adding phrase: ' + data.message);
      }
    } catch (e) {
      alert('Network error adding phrase: ' + e.message);
    } finally {
      btnAddPhrase.disabled = false;
      btnAddPhrase.textContent = '➕ Add Phrase';
    }
  }

  btnAddPhrase.addEventListener('click', handleAddCustomPhrase);
  customPhraseInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleAddCustomPhrase();
    }
  });

  // --------------------------------------------------------------------------
  // DATASET RECORDER (RECORD SAMPLES PER WORD)
  // --------------------------------------------------------------------------
  btnRecordSample.addEventListener('click', () => {
    if (!isCameraActive || !latestLandmarks) {
      alert('Please start the camera and face the lens before recording.');
      return;
    }

    const selectedWord = recordWordSelect.value;
    btnRecordSample.disabled = true;
    btnRecordSample.textContent = '⌛ Get Ready...';

    // 1-second preparation buffer then record
    setTimeout(() => {
      recordingIndicator.style.display = 'block';
      btnRecordSample.textContent = '🔴 Mouth Now!';

      extractor.startRecording(selectedWord, async (word, capturedTensor) => {
        recordingIndicator.style.display = 'none';
        btnRecordSample.disabled = false;
        btnRecordSample.textContent = '⏺️ Record 1s Movement';

        // Send captured tensor to backend to save as dataset/<WORD>/sample_<timestamp>.npy
        try {
          const res = await fetch('/api/record_sample', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ word: word, tensor: capturedTensor })
          });
          const result = await res.json();
          if (result.status === 'success') {
            const cleanKey = word.replace(/\s+/g, '_');
            const countEl = document.getElementById(`count-${cleanKey}`);
            if (countEl) countEl.textContent = result.word_count;
            totalDatasetCount.textContent = `${result.total_samples} Total Files Stored`;
            alert(`✅ Recorded and stored sample for "${word}" in dataset/${cleanKey}/ (Total: ${result.word_count} files)`);
          }
        } catch (e) {
          alert('Failed to save sample: ' + e.message);
        }
      });
    }, 1000);
  });

  // --------------------------------------------------------------------------
  // TRAIN 100 EPOCHS ON STORED DATASET
  // --------------------------------------------------------------------------
  btnTrain100.addEventListener('click', async () => {
    btnTrain100.disabled = true;
    btnTrain100.textContent = '⚡ Training 100 Epochs (Please wait ~20-30s)...';
    trainStatusMsg.style.display = 'block';
    trainStatusMsg.textContent = 'Training neural network across dataset/<WORD>/...';

    try {
      const res = await fetch('/api/train_100_epochs', { method: 'POST' });
      const data = await res.json();

      if (data.status === 'success') {
        const r = data.training_result;
        trainStatusMsg.innerHTML = `🎉 <strong>100 Epochs Complete!</strong> Best Val Acc: <strong>${r.best_val_acc}%</strong> (Trained in ${r.training_time_s}s on ${r.total_samples} augmented samples). Model hot-reloaded!`;
        alert(`Training Complete!\n100 Epochs Trained.\nBest Validation Accuracy: ${r.best_val_acc}%\nModel weights hot-reloaded into memory!`);
      } else {
        trainStatusMsg.textContent = 'Training failed: ' + data.message;
      }
    } catch (e) {
      trainStatusMsg.textContent = 'Training error: ' + e.message;
    } finally {
      btnTrain100.disabled = false;
      btnTrain100.textContent = '⚡ Train Model on Stored Movements (100 Epochs)';
    }
  });

});
