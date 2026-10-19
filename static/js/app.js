/**
 * SILENTVOICE AI • CLIENT APPLICATION ORCHESTRATOR V4.1
 * Unified Vision Pipeline, 10-Slot Visual Movement Studio,
 * Real-Time Continuous Conversational Decoder & AI Care Partner
 */

document.addEventListener('DOMContentLoaded', () => {
  const extractor = new window.LipKinematicsExtractor();
  const REQUIRED_SAMPLES = 10;

  // DOM Elements — Permanent Vision Column
  const videoEl = document.getElementById('cameraFeed');
  const canvasEl = document.getElementById('meshCanvas');
  const ctx = canvasEl.getContext('2d');

  const btnCamera = document.getElementById('btnCamera');
  const btnCalibrate = document.getElementById('btnCalibrate');
  const btnVoiceToggle = document.getElementById('btnVoiceToggle');
  const btnSimulate = document.getElementById('btnSimulate');
  const btnQuickDemo = document.getElementById('btnQuickDemo');
  const standbyOverlay = document.getElementById('standbyOverlay');
  const recordingIndicator = document.getElementById('recordingIndicator');
  const recCountdown = document.getElementById('recCountdown');

  // KPIs
  const kpiLatencyEl = document.getElementById('kpiLatency');
  const kpiApertureEl = document.getElementById('kpiAperture');
  const kpiVelocityEl = document.getElementById('kpiVelocity');
  const kpiFpsEl = document.getElementById('kpiFps');

  // Navigation Tabs
  const modeTabBtns = document.querySelectorAll('.mode-tab-btn');
  const tabPanels = {
    studio: document.getElementById('tab-studio'),
    live: document.getElementById('tab-live')
  };
  let activeTab = 'studio';

  // Studio Elements
  const wordsGrid = document.getElementById('wordsGrid');
  const customPhraseInput = document.getElementById('customPhraseInput');
  const btnAddPhrase = document.getElementById('btnAddPhrase');
  const btnResetAll = document.getElementById('btnResetAll');
  const btnTrain100 = document.getElementById('btnTrain100');
  const trainStatusBanner = document.getElementById('trainStatusBanner');
  const studioSampleCounter = document.getElementById('studioSampleCounter');
  const navDatasetBadge = document.getElementById('navDatasetBadge');
  const datasetProgressFill = document.getElementById('datasetProgressFill');
  const epochTrainingProgress = document.getElementById('epochTrainingProgress');
  const epochProgressBar = document.getElementById('epochProgressBar');
  const epochProgressLabel = document.getElementById('epochProgressLabel');
  const epochProgressValAcc = document.getElementById('epochProgressValAcc');

  // Recording Modal Elements
  const recordingModal = document.getElementById('recordingModal');
  const modalWordTitle = document.getElementById('modalWordTitle');
  const modalSampleSubtitle = document.getElementById('modalSampleSubtitle');
  const modalCurrentSampleNum = document.getElementById('modalCurrentSampleNum');
  const modalCountdown = document.getElementById('modalCountdown');
  const modalInstruction = document.getElementById('modalInstruction');
  const btnCancelModal = document.getElementById('btnCancelModal');

  // Live Testing Elements
  const billboardCardEl = document.getElementById('billboardCard');
  const billboardWordEl = document.getElementById('billboardWord');
  const confidenceBadgeEl = document.getElementById('confidenceBadge');
  const synthesizedTextEl = document.getElementById('synthesizedText');
  const continuousSentenceText = document.getElementById('continuousSentenceText');
  const chatScrollArea = document.getElementById('chatScrollArea');
  const suggestionsChips = document.getElementById('suggestionsChips');
  const historyListEl = document.getElementById('historyList');

  // Application State
  let isCameraActive = false;
  let isVoiceEnabled = true;
  let localMediaStream = null;
  let faceMesh = null;
  let latestLandmarks = null;
  let frameCount = 0;
  let lastFpsTime = performance.now();
  let lastPredictTime = 0;
  let isProcessingInference = false;
  let lastSpokenWord = "";
  let lastSpokenTime = 0;
  let isModalActive = false;
  let cancelRecordingFlag = false;

  // Local word sample tracker
  let wordSampleCounts = {};

  function resizeCanvas() {
    canvasEl.width = videoEl.clientWidth || 500;
    canvasEl.height = videoEl.clientHeight || 380;
  }
  window.addEventListener('resize', resizeCanvas);

  // --------------------------------------------------------------------------
  // TAB NAVIGATION (CAMERA REMAINS UNIFIED & ACTIVE ACROSS BOTH TABS)
  // --------------------------------------------------------------------------
  modeTabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTab = btn.dataset.tab;
      activeTab = targetTab;

      modeTabBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      Object.entries(tabPanels).forEach(([key, panel]) => {
        if (key === targetTab) panel.classList.add('active');
        else panel.classList.remove('active');
      });
    });
  });

  // --------------------------------------------------------------------------
  // ROBUST NON-BLOCKING DUAL-VOICE TTS QUEUE (CHROMIUM SAFE)
  // --------------------------------------------------------------------------
  class RobustSpeechQueue {
    constructor() {
      this.queue = [];
      this.isSpeaking = false;
      this.userVoice = null;
      this.aiVoice = null;
      this.initVoices();

      // Chromium watchdog: Prevent Web Speech API freezing
      setInterval(() => {
        if (window.speechSynthesis && window.speechSynthesis.speaking) {
          window.speechSynthesis.pause();
          window.speechSynthesis.resume();
        }
      }, 5000);
    }

    initVoices() {
      if (!('speechSynthesis' in window)) return;
      const load = () => {
        const voices = window.speechSynthesis.getVoices();
        if (!voices || voices.length === 0) return;
        this.userVoice = voices.find(v => v.lang.startsWith('en') && (v.name.includes('David') || v.name.includes('Google US English') || v.name.includes('Natural'))) || voices[0];
        this.aiVoice = voices.find(v => v.lang.startsWith('en') && (v.name.includes('Zira') || v.name.includes('Samantha') || v.name.includes('Victoria') || v.name.includes('Google UK English Female'))) || voices[0];
      };
      load();
      if (window.speechSynthesis.onvoiceschanged !== undefined) {
        window.speechSynthesis.onvoiceschanged = load;
      }
    }

    speak(text, sender = 'patient') {
      if (!isVoiceEnabled || !('speechSynthesis' in window) || !text) return;
      this.queue.push({ text, sender });
      this.process();
    }

    process() {
      if (this.isSpeaking || this.queue.length === 0) return;
      const item = this.queue.shift();
      this.isSpeaking = true;

      try {
        const utterance = new SpeechSynthesisUtterance(item.text);
        utterance.rate = item.sender === 'ai' ? 1.05 : 1.0;
        utterance.pitch = item.sender === 'ai' ? 1.15 : 0.95;
        utterance.lang = 'en-US';

        if (item.sender === 'ai' && this.aiVoice) utterance.voice = this.aiVoice;
        else if (item.sender === 'patient' && this.userVoice) utterance.voice = this.userVoice;

        utterance.onend = () => {
          this.isSpeaking = false;
          setTimeout(() => this.process(), 100);
        };
        utterance.onerror = () => {
          this.isSpeaking = false;
          setTimeout(() => this.process(), 100);
        };

        window.speechSynthesis.speak(utterance);
      } catch (e) {
        this.isSpeaking = false;
      }
    }
  }

  const speechQueue = new RobustSpeechQueue();

  // --------------------------------------------------------------------------
  // BIONIC LIP MESH RENDERING
  // --------------------------------------------------------------------------
  const LIP_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61];
  const LIP_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78];

  function drawBionicLipMesh(landmarks) {
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    if (!landmarks) return;

    const w = canvasEl.width;
    const h = canvasEl.height;

    // Outer Lip Contour
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

    // Inner Lip Contour
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

    // Extract real physical features with true pixel-space metrics & VAD
    const feat = extractor.extractFrameFeatures(latestLandmarks, videoEl.videoWidth || 640, videoEl.videoHeight || 480);
    if (!feat) return;

    kpiApertureEl.textContent = `${(feat.metrics.vertical_aperture * 100).toFixed(1)}%`;
    kpiVelocityEl.textContent = `${feat.metrics.velocity.toFixed(3)}`;

    // Reset duplicate lockout when mouth returns to resting closure
    if (feat.metrics.consecutive_rest >= 5) {
      lastSpokenWord = "";
    }

    // Update Recording Countdown if recording
    if (extractor.isRecording) {
      recCountdown.textContent = `${30 - extractor.recordingFrames.length}`;
    }

    // Continuous Real-Time Inference (active in Live Testing tab when not recording training data)
    if (!extractor.isRecording && extractor.isBufferReady() && !isProcessingInference && (now - lastPredictTime) > 160) {
      lastPredictTime = now;
      runInference(extractor.getTemporalTensor());
    }
  }

  // --------------------------------------------------------------------------
  // UNIFIED CAMERA CONTROLS (NATIVE WEBRTC + MEDIAPIPE PIPELINE)
  // --------------------------------------------------------------------------
  btnCamera.addEventListener('click', async () => {
    if (!isCameraActive) {
      try {
        btnCamera.textContent = '⌛ Starting Camera...';

        localMediaStream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
          audio: false
        });
        videoEl.srcObject = localMediaStream;
        await videoEl.play();

        if (standbyOverlay) standbyOverlay.style.display = 'none';
        resizeCanvas();

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

        isCameraActive = true;
        btnCamera.textContent = '🛑 Stop Bionic Camera';
        btnCamera.classList.replace('btn-primary', 'btn-secondary');

        async function processVideoFrame() {
          if (!isCameraActive) return;
          if (videoEl.readyState >= 2 && faceMesh) {
            try {
              await faceMesh.send({ image: videoEl });
            } catch (e) {
              // frame drop graceful skip
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

  btnCalibrate.addEventListener('click', () => {
    if (latestLandmarks) {
      extractor.calibrateNeutral(latestLandmarks, videoEl.videoWidth, videoEl.videoHeight);
      btnCalibrate.textContent = '✅ Calibrated!';
      setTimeout(() => btnCalibrate.textContent = '🎯 Calibrate', 1500);
    } else {
      alert('Face not detected. Start camera and face the lens first.');
    }
  });

  btnVoiceToggle.addEventListener('click', () => {
    isVoiceEnabled = !isVoiceEnabled;
    btnVoiceToggle.textContent = isVoiceEnabled ? '🔊 Audio Voice: ON (Audible TTS)' : '🔇 Audio Voice: MUTED';
    btnVoiceToggle.style.color = isVoiceEnabled ? 'var(--cyan-bionic)' : 'var(--text-dim)';
  });

  // --------------------------------------------------------------------------
  // INFERENCE ENGINE (REAL-TIME CONTINUOUS DECODER)
  // --------------------------------------------------------------------------
  async function runInference(tensor30x12) {
    if (isProcessingInference) return;
    isProcessingInference = true;

    try {
      const res = await fetch('/api/predict_silent_speech', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tensor: tensor30x12 })
      });
      const data = await res.json();
      if (data.status !== 'success') {
        isProcessingInference = false;
        return;
      }

      kpiLatencyEl.textContent = `${data.latency_ms}ms`;

      // Update Probability Bars
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

      const now = performance.now();

      // Handle Active Speech Detection
      if (data.is_active_speech && data.confidence >= 0.60 && data.predicted_word !== 'RESTING') {
        const isDuplicateRecent = (data.predicted_word === lastSpokenWord && (now - lastSpokenTime) < 1000);

        if (!isDuplicateRecent) {
          lastSpokenWord = data.predicted_word;
          lastSpokenTime = now;

          // 1. Update Billboard
          billboardCardEl.classList.add('active-speech');
          billboardWordEl.textContent = data.predicted_word;
          confidenceBadgeEl.textContent = `${(data.confidence * 100).toFixed(1)}% Conf`;
          synthesizedTextEl.textContent = `"${data.synthesized_sentence}"`;

          // 2. Continuous Sentence Accumulator
          if (continuousSentenceText) {
            continuousSentenceText.textContent = `"${data.continuous_sentence || data.synthesized_sentence}"`;
          }

          // 3. Two-Way Live Conversation Stream (Patient Bubble)
          appendChatMessage('patient', data.synthesized_sentence, data.predicted_word, data.confidence);
          speechQueue.speak(data.synthesized_sentence, 'patient');

          // 4. AI Care Partner Dialogue Response
          if (data.ai_dialogue_reply) {
            setTimeout(() => {
              appendChatMessage('ai', data.ai_dialogue_reply);
              speechQueue.speak(data.ai_dialogue_reply, 'ai');
            }, 320);
          }

          // 5. Contextual Next-Word Suggestions
          if (data.suggested_next_words) {
            updateSuggestionChips(data.suggested_next_words);
          }

          // 6. Clinical History Record
          addHistoryItem(data.predicted_word, data.synthesized_sentence, data.confidence);

          // 7. FLUID BUFFER STRIDE: Discard completed word to enable next word recognition!
          extractor.flushBufferAfterDetection(6);
        }

      } else {
        billboardCardEl.classList.remove('active-speech');
        if (data.predicted_word === 'RESTING') {
          billboardWordEl.textContent = 'RESTING';
          confidenceBadgeEl.textContent = 'Awaiting Movement';
        }
      }

    } catch (err) {
      // quiet inference catch
    } finally {
      isProcessingInference = false;
    }
  }

  function appendChatMessage(sender, text, rawWord = '', conf = 0) {
    if (!chatScrollArea || !text) return;

    const bubble = document.createElement('div');
    const timeStr = new Date().toTimeString().split(' ')[0];

    if (sender === 'patient') {
      bubble.className = 'chat-bubble chat-patient';
      bubble.innerHTML = `
        <div class="chat-meta">
          <span>👤 Patient (${rawWord || 'Silent Speech'})</span>
          <span>${conf > 0 ? (conf * 100).toFixed(0) + '% • ' : ''}${timeStr}</span>
        </div>
        <div>"${text}"</div>
      `;
    } else {
      bubble.className = 'chat-bubble chat-ai';
      bubble.innerHTML = `
        <div class="chat-meta">
          <span>🤖 AI Care Partner</span>
          <span>${timeStr}</span>
        </div>
        <div>${text}</div>
      `;
    }

    chatScrollArea.appendChild(bubble);
    chatScrollArea.scrollTop = chatScrollArea.scrollHeight;
  }

  function updateSuggestionChips(suggestions) {
    if (!suggestionsChips || !suggestions || suggestions.length === 0) return;
    suggestionsChips.innerHTML = '';
    suggestions.forEach(word => {
      const btn = document.createElement('button');
      btn.className = 'suggestion-chip';
      btn.dataset.word = word;
      btn.textContent = word;
      btn.addEventListener('click', () => {
        runWordSimulation(word);
      });
      suggestionsChips.appendChild(btn);
    });
  }

  function addHistoryItem(word, sentence, conf) {
    if (!historyListEl) return;
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
  // 10-SLOT DATASET STUDIO CONTROLLERS
  // --------------------------------------------------------------------------
  function updateCardSlotsUI(word, count) {
    const cleanDir = word.replace(/\s+/g, '_');
    wordSampleCounts[word] = count;

    const card = document.getElementById(`card-${cleanDir}`);
    const badge = document.getElementById(`badge-${cleanDir}`);
    if (badge) badge.textContent = `${count} / ${REQUIRED_SAMPLES}`;

    if (card) {
      if (count >= REQUIRED_SAMPLES) card.classList.add('complete');
      else card.classList.remove('complete');
    }

    for (let i = 1; i <= REQUIRED_SAMPLES; i++) {
      const slot = document.getElementById(`slot-${cleanDir}-${i}`);
      if (slot) {
        if (i <= count) {
          slot.classList.add('filled');
          slot.classList.remove('current');
          slot.textContent = '✓';
        } else {
          slot.classList.remove('filled', 'current');
          slot.textContent = i;
        }
      }
    }

    checkTrainingReadiness();
  }

  function checkTrainingReadiness() {
    let totalCollected = 0;
    let readyWords = 0;
    const words = Object.keys(wordSampleCounts);

    words.forEach(w => {
      const cnt = wordSampleCounts[w] || 0;
      totalCollected += cnt;
      if (cnt >= REQUIRED_SAMPLES) readyWords++;
    });

    if (studioSampleCounter) studioSampleCounter.textContent = `${totalCollected} Total Recordings`;
    if (navDatasetBadge) navDatasetBadge.textContent = `${totalCollected} Samples`;

    const maxExpected = Math.max(words.length * REQUIRED_SAMPLES, 20);
    const pct = Math.min((totalCollected / maxExpected) * 100, 100);
    if (datasetProgressFill) datasetProgressFill.style.width = `${pct}%`;

    if (readyWords >= 2 || totalCollected >= 20) {
      trainStatusBanner.className = 'train-status-pill ready';
      trainStatusBanner.innerHTML = `<span>✅ <strong>${readyWords} word(s) ready</strong> with recordings! Click below to run 100-epoch training.</span>`;
      btnTrain100.disabled = false;
    } else {
      trainStatusBanner.className = 'train-status-pill warn';
      trainStatusBanner.innerHTML = `<span>⚠ Record 10 samples for at least 2 words to enable 100-epoch training (${readyWords}/2 ready).</span>`;
      btnTrain100.disabled = true;
    }
  }

  // Record 1 Single Sample for a Word
  window.recordSingleSample = async function(word) {
    if (!isCameraActive || !latestLandmarks) {
      alert('Please start the camera and face the lens before recording.');
      return;
    }

    const currentCount = wordSampleCounts[word] || 0;
    if (currentCount >= REQUIRED_SAMPLES) {
      alert(`Already collected all ${REQUIRED_SAMPLES} samples for "${word}". Use the trash button to clear and re-record.`);
      return;
    }

    const nextSlot = currentCount + 1;
    const cleanDir = word.replace(/\s+/g, '_');
    const slotEl = document.getElementById(`slot-${cleanDir}-${nextSlot}`);
    if (slotEl) slotEl.classList.add('current');

    // 1-second prepare then record
    recordingIndicator.style.display = 'block';
    recordingIndicator.classList.add('recording-indicator-hud');

    extractor.startRecording(word, async (w, capturedTensor) => {
      recordingIndicator.style.display = 'none';
      recordingIndicator.classList.remove('recording-indicator-hud');

      try {
        const res = await fetch('/api/record_sample', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ word: w, tensor: capturedTensor, sample_num: nextSlot })
        });
        const result = await res.json();
        if (result.status === 'success') {
          updateCardSlotsUI(w, result.current_count);
        } else {
          alert('Recording note: ' + result.message);
        }
      } catch (e) {
        alert('Network error saving sample: ' + e.message);
      }
    });
  };

  // Guided 10x Recording Sequence
  window.record10xSequence = async function(word) {
    if (!isCameraActive || !latestLandmarks) {
      alert('Please start the camera and face the lens first.');
      return;
    }

    let count = wordSampleCounts[word] || 0;
    if (count >= REQUIRED_SAMPLES) {
      if (!confirm(`Word "${word}" already has 10 samples. Do you want to wipe and re-record all 10?`)) return;
      await window.clearWordSamples(word);
      count = 0;
    }

    isModalActive = true;
    cancelRecordingFlag = false;
    recordingModal.classList.add('active');
    modalWordTitle.textContent = word;

    for (let sampleIdx = count + 1; sampleIdx <= REQUIRED_SAMPLES; sampleIdx++) {
      if (cancelRecordingFlag) break;

      modalCurrentSampleNum.textContent = sampleIdx;
      modalInstruction.textContent = `Get ready for sample ${sampleIdx} of ${REQUIRED_SAMPLES}...`;

      // 3-second countdown
      for (let c = 3; c >= 1; c--) {
        if (cancelRecordingFlag) break;
        modalCountdown.textContent = c;
        modalCountdown.style.color = '#f59e0b';
        await new Promise(r => setTimeout(r, 850));
      }

      if (cancelRecordingFlag) break;

      modalCountdown.textContent = '🔴 MOUTH NOW';
      modalCountdown.style.color = '#ef4444';
      modalInstruction.textContent = `Silently mouth "${word}" clearly now! (1s)`;

      recordingIndicator.style.display = 'block';
      recordingIndicator.classList.add('recording-indicator-hud');

      const capturedTensor = await new Promise(resolve => {
        extractor.startRecording(word, (w, tensor) => {
          resolve(tensor);
        });
      });

      recordingIndicator.style.display = 'none';
      recordingIndicator.classList.remove('recording-indicator-hud');

      if (cancelRecordingFlag) break;

      // Save sample to backend
      try {
        const res = await fetch('/api/record_sample', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ word: word, tensor: capturedTensor, sample_num: sampleIdx })
        });
        const result = await res.json();
        if (result.status === 'success') {
          updateCardSlotsUI(word, result.current_count);
        }
      } catch (e) {
        console.error('Save error:', e);
      }

      modalCountdown.textContent = '✅';
      modalInstruction.textContent = `Sample ${sampleIdx} captured! Brief pause before next...`;
      await new Promise(r => setTimeout(r, 900));
    }

    recordingModal.classList.remove('active');
    isModalActive = false;
  };

  btnCancelModal.addEventListener('click', () => {
    cancelRecordingFlag = true;
    extractor.isRecording = false;
    recordingIndicator.style.display = 'none';
    recordingIndicator.classList.remove('recording-indicator-hud');
    recordingModal.classList.remove('active');
  });

  // Clear single word samples
  window.clearWordSamples = async function(word) {
    if (!confirm(`Are you sure you want to empty all recordings for "${word}"?`)) return;

    try {
      const res = await fetch('/api/clear_word_samples', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ word })
      });
      const data = await res.json();
      if (data.status === 'success') {
        updateCardSlotsUI(word, 0);
      } else {
        alert('Error clearing: ' + data.message);
      }
    } catch (e) {
      alert('Network error: ' + e.message);
    }
  };

  // Reset ALL Trained Boxes
  btnResetAll.addEventListener('click', async () => {
    const ok = confirm('🚨 EMPTY ALL TRAINED BOXES?\n\nThis will completely wipe all recorded movements across all words so you can start 100% fresh.');
    if (!ok) return;

    try {
      const res = await fetch('/api/reset_all_dataset', { method: 'POST' });
      const data = await res.json();
      if (data.status === 'success') {
        Object.keys(wordSampleCounts).forEach(w => {
          updateCardSlotsUI(w, 0);
        });
        alert(`✅ All trained boxes are now completely empty! (Wiped ${data.deleted_count} files). You can now record 10 samples per word.`);
      }
    } catch (e) {
      alert('Error resetting dataset: ' + e.message);
    }
  });

  // Add Custom Word / Full Sentence
  btnAddPhrase.addEventListener('click', async () => {
    const rawVal = customPhraseInput.value.trim();
    if (!rawVal) {
      alert('Please enter a word or sentence (e.g. "I WANT WATER" or "DOCTOR")');
      return;
    }

    btnAddPhrase.disabled = true;
    btnAddPhrase.textContent = '⏳ Adding...';

    try {
      const res = await fetch('/api/add_word', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ phrase: rawVal })
      });
      const data = await res.json();

      if (data.status === 'success') {
        const phrase = data.phrase;
        const cleanDir = phrase.replace(/\s+/g, '_');
        const cleanKey = phrase.replace(/\s+/g, '');

        // Add Card to Studio Grid if not present
        if (!document.getElementById(`card-${cleanDir}`)) {
          const card = document.createElement('div');
          card.className = 'word-card-studio';
          card.id = `card-${cleanDir}`;
          card.dataset.word = phrase;

          let slotsHtml = '';
          for (let i = 1; i <= REQUIRED_SAMPLES; i++) {
            slotsHtml += `<div class="slot-box" id="slot-${cleanDir}-${i}">${i}</div>`;
          }

          card.innerHTML = `
            <div class="word-card-header">
              <span class="word-card-name">${phrase}</span>
              <span class="word-count-badge" id="badge-${cleanDir}">0 / ${REQUIRED_SAMPLES}</span>
            </div>
            <div class="slots-container" id="slots-${cleanDir}">
              ${slotsHtml}
            </div>
            <div class="word-card-actions">
              <button class="btn-bionic btn-primary" onclick="window.record10xSequence('${phrase}')">
                <span>● Record 10x</span>
              </button>
              <button class="btn-bionic btn-secondary" onclick="window.recordSingleSample('${phrase}')">
                <span>⏺ 1x</span>
              </button>
              <button class="btn-bionic btn-secondary" onclick="window.clearWordSamples('${phrase}')" title="Clear recordings">
                <span>🗑</span>
              </button>
            </div>
          `;
          wordsGrid.appendChild(card);
          wordSampleCounts[phrase] = 0;
        }

        // Add to heatmap if not present
        const heatmapGrid = document.getElementById('visemeBarsGrid');
        if (heatmapGrid && !document.getElementById(`bar-${cleanKey}`)) {
          const item = document.createElement('div');
          item.className = 'viseme-bar-item';
          item.innerHTML = `
            <div class="viseme-name-row">
              <span title="${phrase}">${phrase}</span>
              <span id="val-${cleanKey}" style="color: var(--cyan-bionic);">0.0%</span>
            </div>
            <div class="viseme-track">
              <div class="viseme-fill" id="bar-${cleanKey}"></div>
            </div>
          `;
          heatmapGrid.appendChild(item);
        }

        customPhraseInput.value = '';
        checkTrainingReadiness();
        alert(`✅ Created empty box for "${phrase}"! Folder: dataset/${cleanDir}/\nYou can now record your 10 movements.`);
      } else {
        alert('Error: ' + data.message);
      }
    } catch (e) {
      alert('Network error: ' + e.message);
    } finally {
      btnAddPhrase.disabled = false;
      btnAddPhrase.textContent = '➕ Add Word Box';
    }
  });

  // 100-Epoch Neural Training Trigger
  btnTrain100.addEventListener('click', async () => {
    btnTrain100.disabled = true;
    btnTrain100.textContent = '⚡ Training 100 Epochs (Please wait ~20-30s)...';
    epochTrainingProgress.style.display = 'block';

    let simulatedEpoch = 0;
    const epochTicker = setInterval(() => {
      simulatedEpoch = Math.min(simulatedEpoch + 1, 95);
      epochProgressBar.style.width = `${simulatedEpoch}%`;
      epochProgressLabel.textContent = `Training Epoch ${simulatedEpoch} / 100...`;
    }, 450);

    try {
      const res = await fetch('/api/train_100_epochs', { method: 'POST' });
      clearInterval(epochTicker);
      const data = await res.json();

      if (data.status === 'success') {
        const r = data.training_result;
        epochProgressBar.style.width = '100%';
        epochProgressLabel.textContent = `🎉 100 Epochs Complete (${r.training_time_s}s)!`;
        epochProgressValAcc.textContent = `Best Val Acc: ${r.best_val_acc}%`;
        trainStatusBanner.className = 'train-status-pill ready';
        trainStatusBanner.innerHTML = `<span>🎉 <strong>Model Trained for 100 Epochs!</strong> Val Accuracy: <strong>${r.best_val_acc}%</strong> across ${r.num_classes} classes. Model hot-reloaded!</span>`;
        alert(`Training Complete!\n100 Epochs Trained.\nBest Validation Accuracy: ${r.best_val_acc}%\nClasses: ${r.vocabulary.join(', ')}\nModel weights hot-reloaded!`);
      } else {
        epochProgressBar.style.width = '0%';
        epochProgressLabel.textContent = 'Training stopped.';
        alert('Training Notice: ' + data.message);
      }
    } catch (e) {
      clearInterval(epochTicker);
      alert('Training error: ' + e.message);
    } finally {
      btnTrain100.disabled = false;
      btnTrain100.textContent = '⚡ Train Neural Network on Stored Movements (100 Epochs)';
    }
  });

  // --------------------------------------------------------------------------
  // LIVE SPEECH SIMULATION DEMO (INTERACTIVE 3-TURN CONVERSATION)
  // --------------------------------------------------------------------------
  async function runWordSimulation(targetPhrase) {
    if (standbyOverlay) standbyOverlay.style.display = 'none';
    resizeCanvas();

    const seqLen = 30;
    const trajectory = [];
    const numWords = Math.max(1, targetPhrase.split(' ').length);

    for (let t = 0; t < seqLen; t++) {
      const p = t / seqLen;
      let aperture = 0.05;
      for (let s = 0; s < numWords; s++) {
        const center = (s + 1) / (numWords + 1);
        const sigma = 0.12 / Math.sqrt(numWords);
        aperture += 0.35 * Math.exp(-Math.pow(p - center, 2) / (2 * sigma * sigma));
      }
      const width = 0.45 + 0.08 * Math.sin(p * Math.PI * numWords);
      const aspect = aperture / (width + 1e-5);
      const outerH = aperture * 1.5 + 0.08;
      const dh = Math.cos(p * Math.PI * numWords) * 0.4;
      const dw = Math.cos(p * Math.PI * numWords) * 0.1;
      const d2h = -Math.sin(p * Math.PI * numWords);

      trajectory.push([
        aperture, width, aspect, outerH, 0.05, 0.06,
        0.02, 0.08, dh, dw, d2h, 0.0
      ]);
    }

    // Animate lips on canvas
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

      ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
      const cx = canvasEl.width / 2;
      const cy = canvasEl.height / 2;
      const rx = (w * canvasEl.width) / 2.2;
      const ry = Math.max(h * canvasEl.height * 1.3, 10);

      // Outer
      ctx.beginPath();
      ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
      ctx.strokeStyle = '#00f2fe';
      ctx.lineWidth = 3;
      ctx.shadowColor = '#00f2fe';
      ctx.shadowBlur = 14;
      ctx.stroke();

      // Inner
      ctx.beginPath();
      ctx.ellipse(cx, cy, rx * 0.65, Math.max(ry * 0.55, 5), 0, 0, Math.PI * 2);
      ctx.strokeStyle = '#a855f7';
      ctx.lineWidth = 2;
      ctx.shadowColor = '#a855f7';
      ctx.shadowBlur = 8;
      ctx.stroke();

      ctx.shadowBlur = 0;
      frame++;
    }, 24);

    await runInference(trajectory);
  }

  async function runInteractiveConversationDemo() {
    // Switch to Live tab to show visual results
    const liveTabBtn = document.querySelector('.mode-tab-btn[data-tab="live"]');
    if (liveTabBtn) liveTabBtn.click();

    // Turn 1: "PRINCE KUMAR"
    await runWordSimulation("PRINCE KUMAR");

    // Turn 2: "HELP" after 2.6 seconds
    setTimeout(async () => {
      await runWordSimulation("HELP");

      // Turn 3: "THANK YOU" after another 2.8 seconds
      setTimeout(async () => {
        await runWordSimulation("THANK YOU");
      }, 2800);
    }, 2600);
  }

  if (btnSimulate) btnSimulate.addEventListener('click', runInteractiveConversationDemo);
  if (btnQuickDemo) btnQuickDemo.addEventListener('click', runInteractiveConversationDemo);

  // Initialize Suggestion Chips click listeners
  document.querySelectorAll('.suggestion-chip').forEach(btn => {
    btn.addEventListener('click', () => {
      runWordSimulation(btn.dataset.word || btn.textContent.trim());
    });
  });

  // --------------------------------------------------------------------------
  // INITIAL DATASET LOADING
  // --------------------------------------------------------------------------
  async function loadInitialStats() {
    try {
      const res = await fetch('/api/dataset_stats');
      const data = await res.json();
      const realCounts = data.real_recordings || {};

      Object.entries(realCounts).forEach(([word, count]) => {
        updateCardSlotsUI(word, count);
      });
      checkTrainingReadiness();
    } catch (e) {
      console.warn('Stats fetch notice:', e);
    }
  }

  loadInitialStats();
});
