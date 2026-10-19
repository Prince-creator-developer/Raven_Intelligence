/**
 * SILENTVOICE AI • CLIENT APPLICATION ORCHESTRATOR V4.3
 * Pure Direct Silent Speech Articulation (Zero Fake AI Chatbot Text)
 * Instant Auto-Model Enhancement on Word Recording Completion
 * Strict Articulatory Gesture Lifecycle (Zero Repeating Looping Audio)
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
  const btnClearSentence = document.getElementById('btnClearSentence');
  const btnSpeakSentence = document.getElementById('btnSpeakSentence');
  const chatScrollArea = document.getElementById('chatScrollArea');
  const suggestionsChips = document.getElementById('suggestionsChips');
  const historyListEl = document.getElementById('historyList');
  const lipStatePill = document.getElementById('lipStatePill');

  // Application State
  let isCameraActive = false;
  let isVoiceEnabled = true;
  let localMediaStream = null;
  let faceMesh = null;
  let latestLandmarks = null;
  let frameCount = 0;
  let lastFpsTime = performance.now();
  let isProcessingInference = false;
  let isModalActive = false;
  let cancelRecordingFlag = false;

  // Active Model Classes & Word Tracking
  let activeModelClasses = ['RESTING', 'I WANT WATER'];
  let wordSampleCounts = {};
  let sentenceWords = [];

  // Fluid Articulatory Gesture Lifecycle (Zero Repeat Echo, Continuous Flow)
  let gestureState = 'REST'; // 'REST' | 'ARTICULATING' | 'COOLDOWN'
  let gestureFrames = 0;
  let peakAperture = 0;
  let cooldownFrames = 0;
  let lastEmittedWord = "";
  let lastEmittedTime = 0;

  function resizeCanvas() {
    canvasEl.width = videoEl.clientWidth || 500;
    canvasEl.height = videoEl.clientHeight || 380;
  }
  window.addEventListener('resize', resizeCanvas);

  // --------------------------------------------------------------------------
  // TAB NAVIGATION
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
  // CLEAN SINGLE-VOICE SPEECH QUEUE (ZERO CANNED AI CHATTER)
  // --------------------------------------------------------------------------
  class RobustSpeechQueue {
    constructor() {
      this.queue = [];
      this.isSpeaking = false;
      this.voice = null;
      this.lastSpokenText = "";
      this.lastSpokenTime = 0;
      this.initVoice();

      // Chromium watchdog: Prevent Web Speech API freezing
      setInterval(() => {
        if (window.speechSynthesis && window.speechSynthesis.speaking) {
          window.speechSynthesis.pause();
          window.speechSynthesis.resume();
        }
      }, 4000);
    }

    initVoice() {
      if (!('speechSynthesis' in window)) return;
      const load = () => {
        const voices = window.speechSynthesis.getVoices();
        if (!voices || voices.length === 0) return;
        this.voice = voices.find(v => v.lang.startsWith('en') && (v.name.includes('David') || v.name.includes('Google US English') || v.name.includes('Natural') || v.name.includes('Guy'))) || voices[0];
      };
      load();
      if (window.speechSynthesis.onvoiceschanged !== undefined) {
        window.speechSynthesis.onvoiceschanged = load;
      }
    }

    speak(text) {
      if (!isVoiceEnabled || !('speechSynthesis' in window) || !text) return;

      const now = performance.now();
      // Strict anti-repetition: Ignore duplicate audio within 3 seconds
      if (text === this.lastSpokenText && (now - this.lastSpokenTime) < 3000) {
        return;
      }
      this.lastSpokenText = text;
      this.lastSpokenTime = now;

      this.queue.push(text);
      this.process();
    }

    process() {
      if (this.isSpeaking || this.queue.length === 0 || !isVoiceEnabled) return;
      const text = this.queue.shift();
      this.isSpeaking = true;

      try {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        utterance.lang = 'en-US';
        if (this.voice) utterance.voice = this.voice;

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

    stopAll() {
      if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
      }
      this.queue = [];
      this.isSpeaking = false;
      this.lastSpokenText = "";
    }
  }

  const speechQueue = new RobustSpeechQueue();

  // --------------------------------------------------------------------------
  // BIONIC LIP MESH RENDERING
  // --------------------------------------------------------------------------
  const LIP_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61];
  const LIP_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78];

  function drawBionicLipMesh(landmarks, feat = null) {
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    if (!landmarks) return;

    const w = canvasEl.width;
    const h = canvasEl.height;

    const p13 = landmarks[13]; // Upper inner lip center
    const p14 = landmarks[14]; // Lower inner lip center

    // Direct pixel separation of inner anchor dots 13 and 14
    const innerDistPx = (p13 && p14) ? Math.hypot((p13.x - p14.x) * w, (p13.y - p14.y) * h) : 0;
    const innerDyPx = (p13 && p14) ? Math.abs(p13.y - p14.y) * h : 0;

    const vel = (feat && feat.metrics) ? feat.metrics.velocity : 0;
    const ap = (feat && feat.metrics) ? feat.metrics.vertical_aperture : 0;
    const neutral = (extractor && extractor.calibratedNeutralH) ? extractor.calibratedNeutralH : 0.045;

    // PHYSICAL REALITY:
    // Lips are active whenever:
    // 1) Inner anchor dots 13 & 14 separate (innerDyPx >= 3.6px or innerDistPx >= 4.0px or ap > neutral * 1.15)
    // 2) OR lips are actively moving (velocity > 0.008)
    // 3) OR gesture state is currently articulating
    const isSeparated = innerDyPx >= 3.6 || innerDistPx >= 4.0 || ap > (neutral * 1.15);
    const isMoving = vel > 0.008;
    const isLipsActive = isSeparated || isMoving || (gestureState === 'ARTICULATING');

    // Outer Lip Boundary Color:
    // MUST BE GREEN (#10b981) whenever mouth moves or inner dots separate!
    // Blue (#00f2fe) ONLY when lips are completely closed/resting!
    // Red (#ef4444) when actively recording dataset samples!
    let strokeColor = '#00f2fe';
    let glowColor = '#00f2fe';
    let lineWidth = 2.4;

    if (extractor.isRecording) {
      strokeColor = '#ef4444';
      glowColor = '#ef4444';
      lineWidth = 3.2;
    } else if (isLipsActive) {
      strokeColor = '#10b981'; // VIBRANT EMERALD GREEN ON MOVEMENT & OPENING
      glowColor = '#10b981';
      lineWidth = 3.2;
    } else {
      strokeColor = '#00f2fe'; // BIONIC BLUE/CYAN ONLY AT REST
      glowColor = '#00f2fe';
      lineWidth = 2.2;
    }

    // Outer Lip Contour
    ctx.beginPath();
    for (let i = 0; i < LIP_OUTER.length; i++) {
      const pt = landmarks[LIP_OUTER[i]];
      const x = pt.x * w;
      const y = pt.y * h;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = lineWidth;
    ctx.shadowColor = glowColor;
    ctx.shadowBlur = 14;
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
    ctx.strokeStyle = isLipsActive ? '#34d399' : '#a855f7';
    ctx.lineWidth = 2.0;
    ctx.shadowColor = isLipsActive ? '#34d399' : '#a855f7';
    ctx.shadowBlur = 8;
    ctx.stroke();

    ctx.shadowBlur = 0;

    // Corner Anchor Points (78, 308)
    [78, 308].forEach(idx => {
      const pt = landmarks[idx];
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(pt.x * w, pt.y * h, 3.5, 0, Math.PI * 2);
      ctx.fill();
    });

    // INNER ANCHOR DOTS (13 & 14) - Center of Upper & Lower Inner Lips
    if (p13 && p14) {
      const x13 = p13.x * w;
      const y13 = p13.y * h;
      const x14 = p14.x * w;
      const y14 = p14.y * h;

      if (isLipsActive) {
        // Bright Emerald Green Dots when Separated or Moving
        ctx.fillStyle = '#10b981';
        ctx.shadowColor = '#10b981';
        ctx.shadowBlur = 12;

        ctx.beginPath();
        ctx.arc(x13, y13, 5.0, 0, Math.PI * 2);
        ctx.fill();

        ctx.beginPath();
        ctx.arc(x14, y14, 5.0, 0, Math.PI * 2);
        ctx.fill();

        // Dynamic aperture connecting guide
        ctx.strokeStyle = 'rgba(16, 185, 129, 0.75)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(x13, y13);
        ctx.lineTo(x14, y14);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.shadowBlur = 0;
      } else {
        // Resting / Closed Cyan Dots touching
        ctx.fillStyle = '#00f2fe';
        ctx.shadowColor = '#00f2fe';
        ctx.shadowBlur = 6;

        ctx.beginPath();
        ctx.arc(x13, y13, 3.5, 0, Math.PI * 2);
        ctx.fill();

        ctx.beginPath();
        ctx.arc(x14, y14, 3.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
      }
    }

    // Update Live HUD Pill
    if (lipStatePill) {
      if (isLipsActive) {
        lipStatePill.textContent = '● Moving / Open (Green)';
        lipStatePill.style.color = '#34d399';
        lipStatePill.style.borderColor = '#10b981';
        lipStatePill.style.boxShadow = '0 0 10px rgba(16, 185, 129, 0.4)';
      } else {
        lipStatePill.textContent = '● Resting / Closed (Blue)';
        lipStatePill.style.color = '#00f2fe';
        lipStatePill.style.borderColor = 'rgba(0, 242, 254, 0.4)';
        lipStatePill.style.boxShadow = 'none';
      }
    }
  }

  // --------------------------------------------------------------------------
  // MEDIAPIPE FACE MESH & FLUID CONTINUOUS GESTURE LIFECYCLE
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

    // Extract kinematics
    const feat = extractor.extractFrameFeatures(latestLandmarks, videoEl.videoWidth || 640, videoEl.videoHeight || 480);
    if (!feat) return;

    // Render Bionic Lip Mesh with live kinematics
    drawBionicLipMesh(latestLandmarks, feat);

    const ap = feat.metrics.vertical_aperture;
    const vel = feat.metrics.velocity;
    const neutral = extractor.calibratedNeutralH || 0.045;

    kpiApertureEl.textContent = `${(ap * 100).toFixed(1)}%`;
    kpiVelocityEl.textContent = `${vel.toFixed(3)}`;

    // Update Recording Countdown if recording training data
    if (extractor.isRecording) {
      recCountdown.textContent = `${30 - extractor.recordingFrames.length}`;
      return;
    }

    // ─────────────────────────────────────────────────────────────
    // CONTINUOUS ARTICULATORY SENTENCE LIFECYCLE (FLOW ACROSS WORDS)
    // ─────────────────────────────────────────────────────────────
    const p13 = latestLandmarks[13];
    const p14 = latestLandmarks[14];
    const w = canvasEl.width;
    const h = canvasEl.height;
    const innerDyPx = (p13 && p14) ? Math.abs(p13.y - p14.y) * h : 0;
    const innerDistPx = (p13 && p14) ? Math.hypot((p13.x - p14.x) * w, (p13.y - p14.y) * h) : 0;

    const isInnerSeparated = innerDyPx >= 3.6 || innerDistPx >= 4.0 || ap > (neutral * 1.15);
    const isMoving = vel > 0.008;
    const isLipsActive = isInnerSeparated || isMoving;

    if (gestureState === 'REST') {
      if (isLipsActive) {
        // Speech onset: lips separated or started moving
        gestureState = 'ARTICULATING';
        gestureFrames = 1;
        peakAperture = ap;
      }
    } else if (gestureState === 'ARTICULATING') {
      gestureFrames++;
      if (ap > peakAperture) peakAperture = ap;

      // Word / gesture completion cue:
      // Minimum duration 12 frames (~400ms) with closure/velocity dip, OR maximum chunk duration 26 frames (~850ms)
      const gestureCompleted = (gestureFrames >= 12 && (ap < peakAperture * 0.78 || vel < 0.010)) || (gestureFrames >= 26);

      if (gestureCompleted) {
        // Trigger inference for this word/chunk
        if (extractor.isBufferReady() && !isProcessingInference) {
          runInference(extractor.getTemporalTensor());
        }

        // Non-blocking transition cooldown: 5 frames (~160ms) to allow buffer stride
        gestureState = 'COOLDOWN';
        cooldownFrames = 5;
      }
    } else if (gestureState === 'COOLDOWN') {
      cooldownFrames--;
      if (cooldownFrames <= 0) {
        if (isLipsActive) {
          // Continuous flow: lips are still open or mouthing the next word in flow!
          gestureState = 'ARTICULATING';
          gestureFrames = 1;
          peakAperture = ap;
        } else {
          // Lips returned to closed resting
          gestureState = 'REST';
        }
      }
    }
  }

  // --------------------------------------------------------------------------
  // UNIFIED CAMERA CONTROLS
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
              // frame skip
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
    if (!isVoiceEnabled) {
      speechQueue.stopAll();
    }
    btnVoiceToggle.textContent = isVoiceEnabled ? '🔊 Audio Voice: ON (Audible TTS)' : '🔇 Audio Voice: MUTED';
    btnVoiceToggle.style.color = isVoiceEnabled ? 'var(--cyan-bionic)' : 'var(--text-dim)';
  });

  // --------------------------------------------------------------------------
  // INFERENCE ENGINE (REAL-TIME DIRECT RECOGNITION)
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

      if (data.model_classes && Array.isArray(data.model_classes)) {
        activeModelClasses = data.model_classes;
      }

      // Update Probability Heatmap for active model classes
      if (data.probabilities) {
        document.querySelectorAll('.viseme-fill').forEach(bar => {
          bar.style.width = '1%';
          bar.style.background = 'rgba(255, 255, 255, 0.08)';
        });
        document.querySelectorAll('[id^="val-"]').forEach(val => {
          val.textContent = '--';
          val.style.color = 'var(--text-dim)';
        });

        for (const [word, prob] of Object.entries(data.probabilities)) {
          const key = word.replace(/\s+/g, '');
          const barFill = document.getElementById(`bar-${key}`);
          const barVal = document.getElementById(`val-${key}`);
          if (barFill && barVal) {
            const pct = (prob * 100).toFixed(1);
            barFill.style.width = `${Math.max(pct, 2)}%`;
            barFill.style.background = (word === data.predicted_word) ? 'linear-gradient(90deg, #10b981, #34d399)' : 'linear-gradient(90deg, var(--cyan-bionic), var(--blue-glow))';
            barVal.textContent = `${pct}%`;
            barVal.style.color = (word === data.predicted_word) ? '#34d399' : 'var(--cyan-bionic)';
          }
        }
      }

      const now = performance.now();

      // Handle Active Speech Recognition
      if (data.is_active_speech && data.confidence >= 0.50 && data.predicted_word !== 'RESTING') {
        const isDuplicateRecent = (data.predicted_word === lastEmittedWord && (now - lastEmittedTime) < 2200);

        if (!isDuplicateRecent) {
          lastEmittedWord = data.predicted_word;
          lastEmittedTime = now;

          // 1. Update Billboard
          billboardCardEl.classList.add('active-speech');
          billboardWordEl.textContent = data.predicted_word;
          confidenceBadgeEl.textContent = `${(data.confidence * 100).toFixed(1)}% Conf`;
          synthesizedTextEl.textContent = `"${data.predicted_word}"`;

          // 2. Append to clean continuous sentence stream
          appendSentenceWord(data.predicted_word);

          // 3. Spoken Audio: Speak the clean word ONCE!
          speechQueue.speak(data.predicted_word);

          // 4. Clinical Speech Activity Stream
          appendSpeechLog(data.predicted_word, data.confidence);

          // 5. Fluid buffer stride (keeps 4 frames transition context for immediate next word)
          extractor.flushBufferAfterDetection(4);
        }

      } else {
        billboardCardEl.classList.remove('active-speech');
        if (data.predicted_word === 'RESTING') {
          billboardWordEl.textContent = 'RESTING';
          confidenceBadgeEl.textContent = 'Resting Baseline';
          synthesizedTextEl.textContent = '"(Resting baseline - mouth closed)"';
        }
      }

    } catch (err) {
      // quiet inference catch
    } finally {
      isProcessingInference = false;
    }
  }

  // Continuous Sentence Stream Management
  function appendSentenceWord(word) {
    if (sentenceWords.length === 0 || sentenceWords[sentenceWords.length - 1] !== word) {
      sentenceWords.push(word);
      if (continuousSentenceText) {
        continuousSentenceText.textContent = sentenceWords.join(' • ');
      }
    }
  }

  function clearSentence() {
    sentenceWords = [];
    if (continuousSentenceText) {
      continuousSentenceText.textContent = '(Awaiting speech...)';
    }
  }

  if (btnClearSentence) btnClearSentence.addEventListener('click', clearSentence);
  if (btnSpeakSentence) {
    btnSpeakSentence.addEventListener('click', () => {
      if (sentenceWords.length > 0) {
        speechQueue.speak(sentenceWords.join(' '));
      }
    });
  }

  // Contextual Suggestion Chips (Click to append directly to sentence stream)
  if (suggestionsChips) {
    suggestionsChips.addEventListener('click', (e) => {
      const chip = e.target.closest('.suggestion-chip');
      if (chip && chip.dataset.word) {
        const word = chip.dataset.word;
        appendSentenceWord(word);
        speechQueue.speak(word);
        appendSpeechLog(word, 1.0);
        billboardCardEl.classList.add('active-speech');
        billboardWordEl.textContent = word;
        synthesizedTextEl.textContent = `"${word}"`;
        confidenceBadgeEl.textContent = '100% (Selected)';
      }
    });
  }

  function appendSpeechLog(word, conf = 0) {
    if (!chatScrollArea || !word) return;

    const bubble = document.createElement('div');
    const timeStr = new Date().toTimeString().split(' ')[0];

    bubble.className = 'chat-bubble chat-patient';
    bubble.style.alignSelf = 'flex-start';
    bubble.style.maxWidth = '100%';
    bubble.innerHTML = `
      <div class="chat-meta">
        <span>👤 Spoken Articulation: <strong>${word}</strong></span>
        <span>${conf > 0 ? (conf * 100).toFixed(0) + '% Conf • ' : ''}${timeStr}</span>
      </div>
      <div style="font-size: 1.1rem; font-weight: 800; color: #f8fafc;">"${word}"</div>
    `;

    chatScrollArea.appendChild(bubble);
    chatScrollArea.scrollTop = chatScrollArea.scrollHeight;
  }

  // --------------------------------------------------------------------------
  // AUTO-ENHANCE NEURAL NETWORK AFTER EACH WORD TRAINING
  // --------------------------------------------------------------------------
  async function autoEnhanceModel(wordTrained) {
    if (trainStatusBanner) {
      trainStatusBanner.className = 'train-status-pill warn';
      trainStatusBanner.innerHTML = `<span>⚡ Auto-enhancing neural network with your new "${wordTrained}" movements...</span>`;
    }

    try {
      const res = await fetch('/api/train', { method: 'POST' });
      const data = await res.json();
      if (data.status === 'success') {
        const r = data.training_result;
        activeModelClasses = r.vocabulary;

        if (trainStatusBanner) {
          trainStatusBanner.className = 'train-status-pill ready';
          trainStatusBanner.innerHTML = `<span>🎉 <strong>Model Enhanced in ${r.training_time_s}s!</strong> Accuracy: <strong>${r.best_val_acc}%</strong>. Active words: <strong>${activeModelClasses.join(', ')}</strong></span>`;
        }

        if (modalInstruction) {
          modalInstruction.textContent = `🎉 Enhanced in ${r.training_time_s}s! "${wordTrained}" is now active and ready for testing!`;
        }
      }
    } catch (e) {
      console.error('Auto enhance error:', e);
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

    if (readyWords >= 1 || totalCollected >= 10) {
      trainStatusBanner.className = 'train-status-pill ready';
      trainStatusBanner.innerHTML = `<span>✅ <strong>${readyWords} word(s) ready</strong> with recordings! Click below to train or re-train.</span>`;
      btnTrain100.disabled = false;
    } else {
      trainStatusBanner.className = 'train-status-pill warn';
      trainStatusBanner.innerHTML = `<span>⚠ Record 10 samples for any word to train the neural network.</span>`;
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
          if (result.current_count >= REQUIRED_SAMPLES) {
            await autoEnhanceModel(w);
          }
        }
      } catch (e) {
        alert('Network error saving sample: ' + e.message);
      }
    });
  };

  // Guided 10x Recording Sequence with Instant Auto-Enhance
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
      await new Promise(r => setTimeout(r, 850));

      // When the 10th sample completes, AUTO-ENHANCE THE NEURAL MODEL IMMEDIATELY!
      if (sampleIdx === REQUIRED_SAMPLES) {
        modalCountdown.textContent = '⚡';
        modalInstruction.textContent = `All 10 samples captured! Auto-enhancing neural network now...`;
        await autoEnhanceModel(word);
        await new Promise(r => setTimeout(r, 1200));
      }
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
        clearSentence();
        alert(`✅ All trained boxes are now completely empty! You can now record 10 samples per word.`);
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

        const heatmapGrid = document.getElementById('visemeBarsGrid');
        if (heatmapGrid && !document.getElementById(`bar-${cleanKey}`)) {
          const item = document.createElement('div');
          item.className = 'viseme-bar-item';
          item.innerHTML = `
            <div class="viseme-name-row">
              <span title="${phrase}">${phrase}</span>
              <span id="val-${cleanKey}" style="color: var(--cyan-bionic);">--</span>
            </div>
            <div class="viseme-track">
              <div class="viseme-fill" id="bar-${cleanKey}"></div>
            </div>
          `;
          heatmapGrid.appendChild(item);
        }

        customPhraseInput.value = '';
        checkTrainingReadiness();
        alert(`✅ Created box for "${phrase}"! Click [● Record 10x] to record your movements.`);
      }
    } catch (e) {
      alert('Network error: ' + e.message);
    } finally {
      btnAddPhrase.disabled = false;
      btnAddPhrase.textContent = '➕ Add Word Box';
    }
  });

  // Manual 80-Epoch Neural Training Trigger
  btnTrain100.addEventListener('click', async () => {
    btnTrain100.disabled = true;
    btnTrain100.textContent = '⚡ Training Neural Model...';
    epochTrainingProgress.style.display = 'block';

    let simulatedEpoch = 0;
    const epochTicker = setInterval(() => {
      simulatedEpoch = Math.min(simulatedEpoch + 2, 95);
      epochProgressBar.style.width = `${simulatedEpoch}%`;
      epochProgressLabel.textContent = `Optimizing Neural Weights (Epoch ${simulatedEpoch})...`;
    }, 200);

    try {
      const res = await fetch('/api/train', { method: 'POST' });
      clearInterval(epochTicker);
      const data = await res.json();

      if (data.status === 'success') {
        const r = data.training_result;
        epochProgressBar.style.width = '100%';
        epochProgressLabel.textContent = `🎉 Training Complete in ${r.training_time_s}s!`;
        epochProgressValAcc.textContent = `Accuracy: ${r.best_val_acc}%`;
        trainStatusBanner.className = 'train-status-pill ready';
        trainStatusBanner.innerHTML = `<span>🎉 <strong>Model Enhanced!</strong> Accuracy: <strong>${r.best_val_acc}%</strong> across <strong>${r.vocabulary.join(', ')}</strong>. Model hot-reloaded!</span>`;

        if (r.vocabulary) {
          activeModelClasses = r.vocabulary;
        }

        alert(`Training Complete in ${r.training_time_s}s!\nAccuracy: ${r.best_val_acc}%\nActive Model Classes: ${r.vocabulary.join(', ')}\nModel weights hot-reloaded!`);
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
      btnTrain100.textContent = '⚡ Train Neural Network on Stored Movements';
    }
  });

  // --------------------------------------------------------------------------
  // LIVE SPEECH DEMO
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

      // Outer Lip - Green during simulated speech movement!
      const isSimMoving = Math.abs(trajectory[frame][8]) > 0.005 || h > 0.055;
      ctx.beginPath();
      ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
      ctx.strokeStyle = isSimMoving ? '#10b981' : '#00f2fe';
      ctx.lineWidth = 3;
      ctx.shadowColor = ctx.strokeStyle;
      ctx.shadowBlur = 14;
      ctx.stroke();

      // Inner
      ctx.beginPath();
      ctx.ellipse(cx, cy, rx * 0.65, Math.max(ry * 0.55, 5), 0, 0, Math.PI * 2);
      ctx.strokeStyle = isSimMoving ? '#34d399' : '#a855f7';
      ctx.lineWidth = 2;
      ctx.shadowColor = ctx.strokeStyle;
      ctx.shadowBlur = 8;
      ctx.stroke();

      // Anchor dots
      ctx.fillStyle = isSimMoving ? '#10b981' : '#00f2fe';
      ctx.beginPath();
      ctx.arc(cx, cy - ry * 0.55, 4.5, 0, Math.PI * 2);
      ctx.arc(cx, cy + ry * 0.55, 4.5, 0, Math.PI * 2);
      ctx.fill();

      if (lipStatePill) {
        lipStatePill.textContent = isSimMoving ? '● Moving / Open (Green)' : '● Resting / Closed (Blue)';
        lipStatePill.style.color = isSimMoving ? '#34d399' : '#00f2fe';
        lipStatePill.style.borderColor = isSimMoving ? '#10b981' : 'rgba(0, 242, 254, 0.4)';
      }

      ctx.shadowBlur = 0;
      frame++;
    }, 24);

    await runInference(trajectory);
  }

  async function runInteractiveConversationDemo() {
    const liveTabBtn = document.querySelector('.mode-tab-btn[data-tab="live"]');
    if (liveTabBtn) liveTabBtn.click();
    await runWordSimulation("I WANT WATER");
  }

  if (btnSimulate) btnSimulate.addEventListener('click', runInteractiveConversationDemo);
  if (btnQuickDemo) btnQuickDemo.addEventListener('click', runInteractiveConversationDemo);

  // --------------------------------------------------------------------------
  // INITIAL DATASET & MODEL CLASS LOADING
  // --------------------------------------------------------------------------
  async function loadInitialStats() {
    try {
      const res = await fetch('/api/dataset_stats');
      const data = await res.json();
      const realCounts = data.real_recordings || {};

      if (data.model_classes && Array.isArray(data.model_classes)) {
        activeModelClasses = data.model_classes;
      }

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
