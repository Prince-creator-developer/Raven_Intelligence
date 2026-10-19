/**
 * SILENTVOICE AI v4.0 — Application Controller
 * Training Studio + Live Testing orchestrator
 */

// ════════════════════════════════════════════════
// SHARED STATE
// ════════════════════════════════════════════════
const REQUIRED_SAMPLES = 10;
let isCameraActive = false;
let isTestCameraActive = false;
let isVoiceEnabled = true;
let localStream = null;
let testStream = null;
let faceMesh = null;
let testFaceMesh = null;
let latestLandmarks = null;
let latestTestLandmarks = null;
let frameCount = 0;
let lastFpsTime = performance.now();
let lastInferenceTime = 0;
let lastSpokenWord = '';
let lastSpokenTime = 0;
let isInferring = false;
let wordCounts = {};   // word → real recording count

// DOM refs — Training tab
const videoEl    = document.getElementById('cameraFeed');
const canvasEl   = document.getElementById('meshCanvas');
const ctx        = canvasEl.getContext('2d');
const btnCamera  = document.getElementById('btnCamera');
const btnCal     = document.getElementById('btnCalibrate');
const btnVoice   = document.getElementById('btnVoiceToggle');
const standby    = document.getElementById('standbyOverlay');
const recPill    = document.getElementById('recordingPill');
const recCount   = document.getElementById('recCountdown');
const kpiAp      = document.getElementById('kpiAperture');
const kpiVel     = document.getElementById('kpiVelocity');
const kpiFps     = document.getElementById('kpiFps');
const kpiLat     = document.getElementById('kpiLatency');
const btnAddWord = document.getElementById('btnAddWord');
const newWordIn  = document.getElementById('newWordInput');
const btnTrain   = document.getElementById('btnTrain');
const trainReady = document.getElementById('trainReadyStatus');
const trainLog   = document.getElementById('trainLog');
const epochBar   = document.getElementById('epochProgressBar');
const epochOuter = document.getElementById('epochProgressOuter');

// DOM refs — Test tab
const videoElT   = document.getElementById('cameraFeedTest');
const canvasElT  = document.getElementById('meshCanvasTest');
const ctxT       = canvasElT ? canvasElT.getContext('2d') : null;
const btnCameraT = document.getElementById('btnCameraTest');
const btnCalT    = document.getElementById('btnCalibrateTest');
const btnVoiceT  = document.getElementById('btnVoiceTest');
const btnDemo    = document.getElementById('btnDemo');
const btnDemoT   = document.getElementById('btnDemoTest');
const standbyT   = document.getElementById('standbyOverlayTest');
const billboard  = document.getElementById('billboard');
const bbWord     = document.getElementById('billboardWord');
const bbSentence = document.getElementById('billboardSentence');
const confBadge  = document.getElementById('confBadge');
const contText   = document.getElementById('continuousText');
const chatArea   = document.getElementById('chatArea');
const chipRow    = document.getElementById('chipRow');
const histList   = document.getElementById('historyList');
const kpiApT     = document.getElementById('kpiApertureT');
const kpiVelT    = document.getElementById('kpiVelocityT');
const kpiFpsT    = document.getElementById('kpiFpsT');
const kpiLatT    = document.getElementById('kpiLatencyT');

// ════════════════════════════════════════════════
// TAB SWITCHING
// ════════════════════════════════════════════════
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
  });
});

// ════════════════════════════════════════════════
// KINEMATIC EXTRACTOR — shared between both tabs
// ════════════════════════════════════════════════
const extractor  = new window.LipKinematicsExtractor();   // Training tab
const extractorT = new window.LipKinematicsExtractor();   // Test tab

// ════════════════════════════════════════════════
// CANVAS RESIZE
// ════════════════════════════════════════════════
function resizeCanvas(vid, cvs) {
  cvs.width  = vid.clientWidth  || 520;
  cvs.height = vid.clientHeight || 400;
}
window.addEventListener('resize', () => {
  if (isCameraActive)     resizeCanvas(videoEl, canvasEl);
  if (isTestCameraActive) resizeCanvas(videoElT, canvasElT);
});

// ════════════════════════════════════════════════
// BIONIC LIP MESH DRAWING
// ════════════════════════════════════════════════
const LIP_OUTER = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146,61];
const LIP_INNER = [78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95,78];

function drawMesh(landmarks, c, cvs, isRec) {
  c.clearRect(0, 0, cvs.width, cvs.height);
  if (!landmarks) return;
  const w = cvs.width, h = cvs.height;
  const drawPath = (pts, color, lw) => {
    c.beginPath();
    pts.forEach((idx, i) => {
      const pt = landmarks[idx];
      const x = pt.x * w, y = pt.y * h;
      i === 0 ? c.moveTo(x, y) : c.lineTo(x, y);
    });
    c.strokeStyle = color; c.lineWidth = lw;
    c.shadowColor = color; c.shadowBlur = 10;
    c.stroke(); c.shadowBlur = 0;
  };
  drawPath(LIP_OUTER, isRec ? '#ef4444' : '#00f2fe', 2.5);
  drawPath(LIP_INNER, '#a855f7', 2);
  [13, 14, 78, 308].forEach(idx => {
    const pt = landmarks[idx];
    c.fillStyle = '#fff';
    c.beginPath(); c.arc(pt.x * w, pt.y * h, 3, 0, Math.PI * 2); c.fill();
  });
}

// ════════════════════════════════════════════════
// MEDIAPIPE SETUP (shared helper)
// ════════════════════════════════════════════════
function buildFaceMesh(onResults) {
  const fm = new FaceMesh({
    locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${f}`
  });
  fm.setOptions({ maxNumFaces: 1, refineLandmarks: true, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
  fm.onResults(onResults);
  return fm;
}

// ════════════════════════════════════════════════
// ── TRAINING TAB CAMERA ──
// ════════════════════════════════════════════════
btnCamera.addEventListener('click', async () => {
  if (!isCameraActive) {
    try {
      btnCamera.textContent = '⌛ Starting…';
      localStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' }, audio: false });
      videoEl.srcObject = localStream;
      await videoEl.play();
      if (standby) standby.style.display = 'none';
      resizeCanvas(videoEl, canvasEl);
      faceMesh = buildFaceMesh(onTrainFrame);
      isCameraActive = true;
      btnCamera.textContent = '🛑 Stop Camera';
      startFrameLoop(videoEl, faceMesh);
      setTimeout(() => {
        if (latestLandmarks) extractor.calibrateNeutral(latestLandmarks, videoEl.videoWidth || 640, videoEl.videoHeight || 480);
      }, 1500);
    } catch (e) {
      alert('Camera error: ' + e.message);
      btnCamera.textContent = '⚡ Start Camera';
    }
  } else {
    stopCamera();
  }
});

function stopCamera() {
  isCameraActive = false;
  if (localStream) { localStream.getTracks().forEach(t => t.stop()); localStream = null; }
  videoEl.srcObject = null;
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  if (standby) standby.style.display = 'flex';
  btnCamera.textContent = '⚡ Start Camera';
}

function startFrameLoop(vid, fm) {
  const loop = async () => {
    if (!isCameraActive) return;
    if (vid.readyState >= 2 && fm) {
      try { await fm.send({ image: vid }); } catch(e) {}
    }
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

function onTrainFrame(results) {
  frameCount++;
  const now = performance.now();
  if (now - lastFpsTime >= 1000) {
    kpiFps.textContent = frameCount + ' FPS';
    frameCount = 0; lastFpsTime = now;
  }
  if (!results.multiFaceLandmarks?.length) { ctx.clearRect(0, 0, canvasEl.width, canvasEl.height); return; }
  latestLandmarks = results.multiFaceLandmarks[0];
  drawMesh(latestLandmarks, ctx, canvasEl, extractor.isRecording);
  const feat = extractor.extractFrameFeatures(latestLandmarks, videoEl.videoWidth || 640, videoEl.videoHeight || 480);
  if (!feat) return;
  kpiAp.textContent  = (feat.metrics.vertical_aperture * 100).toFixed(1) + '%';
  kpiVel.textContent = feat.metrics.velocity.toFixed(3);
  if (extractor.isRecording) recCount.textContent = String(30 - extractor.recordingFrames.length);
}

btnCal.addEventListener('click', () => {
  if (latestLandmarks) {
    extractor.calibrateNeutral(latestLandmarks, videoEl.videoWidth || 640, videoEl.videoHeight || 480);
    btnCal.textContent = '✅ Done!';
    setTimeout(() => btnCal.textContent = '🎯 Calibrate', 1500);
  } else { alert('Start camera first.'); }
});

// ════════════════════════════════════════════════
// ── RECORDING SESSION (10 samples per word) ──
// ════════════════════════════════════════════════
let recordingWord = '';
let recordingSampleNum = 0;
let isRecordingSession = false;

const modal         = document.getElementById('recordModal');
const modalWord     = document.getElementById('modalWord');
const modalInstr    = document.getElementById('modalInstruction');
const modalCountdown= document.getElementById('modalCountdown');
const sampleDots    = document.getElementById('sampleDots');
const curSampleEl   = document.getElementById('curSample');
const btnModalClose = document.getElementById('btnModalClose');

function startRecordSession(word) {
  if (!isCameraActive) { alert('Please start the camera first.'); return; }
  recordingWord = word;
  recordingSampleNum = wordCounts[word] || 0;
  if (recordingSampleNum >= REQUIRED_SAMPLES) { alert(`Already have ${REQUIRED_SAMPLES} recordings for "${word}". Clear first if you want to re-record.`); return; }
  modal.classList.add('open');
  modalWord.textContent = word;
  updateModalDots();
  runNextSample();
}

function updateModalDots() {
  sampleDots.innerHTML = '';
  for (let i = 0; i < REQUIRED_SAMPLES; i++) {
    const dot = document.createElement('div');
    dot.className = 'sample-dot';
    dot.textContent = i + 1;
    if (i < recordingSampleNum) dot.classList.add('done');
    else if (i === recordingSampleNum) dot.classList.add('active');
    sampleDots.appendChild(dot);
  }
  curSampleEl.textContent = recordingSampleNum;
}

async function runNextSample() {
  if (recordingSampleNum >= REQUIRED_SAMPLES) {
    modalInstr.textContent = `✅ All ${REQUIRED_SAMPLES} recordings complete for "${recordingWord}"!`;
    modalCountdown.style.display = 'none';
    updateWordCardUI(recordingWord, REQUIRED_SAMPLES);
    checkTrainReady();
    setTimeout(() => {
      modal.classList.remove('open');
      isRecordingSession = false;
    }, 1800);
    return;
  }

  isRecordingSession = true;
  modalInstr.textContent = `Sample ${recordingSampleNum + 1} of ${REQUIRED_SAMPLES} — Get ready!`;
  updateModalDots();

  // 3-second countdown
  modalCountdown.style.display = 'block';
  for (let i = 3; i >= 1; i--) {
    modalCountdown.textContent = String(i);
    await sleep(900);
  }
  modalCountdown.textContent = '🔴';
  modalInstr.textContent = `Mouth "${recordingWord}" clearly now!`;

  // Show recording pill
  recPill.style.display = 'block';
  recPill.classList.add('recording-pill');

  // Capture 30 frames (~1 second)
  extractor.startRecording(recordingWord, async (word, tensor) => {
    recPill.style.display = 'none';

    // Send to backend
    try {
      const res = await fetch('/api/record_sample', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ word, tensor })
      });
      const data = await res.json();
      if (data.status === 'success') {
        recordingSampleNum = data.current_count;
        wordCounts[word] = recordingSampleNum;
        updateWordCardUI(word, recordingSampleNum);
        checkTrainReady();
        await sleep(400);
        runNextSample();
      } else {
        modalInstr.textContent = '⚠ ' + data.message;
        isRecordingSession = false;
      }
    } catch(e) {
      modalInstr.textContent = 'Network error: ' + e.message;
      isRecordingSession = false;
    }
  });
}

btnModalClose.addEventListener('click', () => {
  extractor.isRecording = false;
  isRecordingSession = false;
  recPill.style.display = 'none';
  modal.classList.remove('open');
});

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// expose globally for onclick attributes
window.startRecordSession = startRecordSession;

// ════════════════════════════════════════════════
// CLEAR WORD SAMPLES
// ════════════════════════════════════════════════
async function clearWordSamples(word) {
  if (!confirm(`Clear all ${REQUIRED_SAMPLES} recordings for "${word}"? Baseline files will be kept.`)) return;
  try {
    const res = await fetch('/api/clear_word_samples', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ word })
    });
    const data = await res.json();
    if (data.status === 'success') {
      wordCounts[word] = 0;
      updateWordCardUI(word, 0);
      checkTrainReady();
    } else { alert('Error: ' + data.message); }
  } catch(e) { alert('Network error: ' + e.message); }
}
window.clearWordSamples = clearWordSamples;

// ════════════════════════════════════════════════
// WORD CARD UI UPDATE
// ════════════════════════════════════════════════
function updateWordCardUI(word, count) {
  const folder = word.replace(/\s+/g, '_');
  const card   = document.getElementById('wcard-' + folder);
  const fill   = document.getElementById('wfill-' + folder);
  const cntEl  = document.getElementById('wcount-' + folder);
  if (!card) return;
  const pct = Math.min((count / REQUIRED_SAMPLES) * 100, 100);
  if (fill) { fill.style.width = pct + '%'; fill.classList.toggle('complete', count >= REQUIRED_SAMPLES); }
  if (cntEl) cntEl.textContent = `${count} / ${REQUIRED_SAMPLES} recordings`;
  card.classList.toggle('complete', count >= REQUIRED_SAMPLES);
  const icon = card.querySelector('.word-status-icon');
  if (icon) icon.textContent = count >= REQUIRED_SAMPLES ? '✅' : count > 0 ? '🔄' : '⭕';
}

// ════════════════════════════════════════════════
// TRAIN READINESS CHECK
// ════════════════════════════════════════════════
function checkTrainReady() {
  const wordCards = document.querySelectorAll('.word-card');
  let allReady = true;
  let totalWords = 0;
  let readyWords = 0;
  wordCards.forEach(card => {
    const word = card.dataset.word;
    if (!word || word === 'RESTING') return;
    totalWords++;
    const cnt = wordCounts[word] || 0;
    if (cnt >= REQUIRED_SAMPLES) readyWords++;
    else allReady = false;
  });

  if (totalWords === 0) return;

  if (readyWords === 0) {
    trainReady.className = 'status-bar warn';
    trainReady.textContent = `⚠ Record ${REQUIRED_SAMPLES} samples per word to enable training.`;
    btnTrain.disabled = true;
  } else if (!allReady) {
    trainReady.className = 'status-bar warn';
    trainReady.textContent = `🔄 ${readyWords}/${totalWords} words ready. Complete all words for best accuracy.`;
    btnTrain.disabled = false;
  } else {
    trainReady.className = 'status-bar';
    trainReady.textContent = `✅ All ${totalWords} words have ${REQUIRED_SAMPLES} recordings. Ready to train!`;
    btnTrain.disabled = false;
  }
}

// ════════════════════════════════════════════════
// ADD CUSTOM WORD
// ════════════════════════════════════════════════
btnAddWord.addEventListener('click', async () => {
  const phrase = newWordIn.value.trim();
  if (!phrase) { alert('Enter a word or sentence.'); return; }
  btnAddWord.disabled = true; btnAddWord.textContent = '…';
  try {
    const res = await fetch('/api/add_word', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ phrase })
    });
    const data = await res.json();
    if (data.status === 'success') {
      newWordIn.value = '';
      addWordCard(data.phrase);
      updateProbBars(data.vocabulary);
    } else { alert('Error: ' + data.message); }
  } catch(e) { alert('Network error: ' + e.message); }
  finally { btnAddWord.disabled = false; btnAddWord.textContent = 'Add'; }
});
newWordIn.addEventListener('keydown', e => { if (e.key === 'Enter') btnAddWord.click(); });

function addWordCard(phrase) {
  const grid   = document.getElementById('wordGrid');
  const folder = phrase.replace(/\s+/g, '_');
  if (document.getElementById('wcard-' + folder)) return;
  const card = document.createElement('div');
  card.className = 'word-card'; card.id = 'wcard-' + folder; card.dataset.word = phrase;
  card.innerHTML = `
    <div class="word-card-name"><span>${phrase}</span><span class="word-status-icon">⭕</span></div>
    <div class="word-progress-bar"><div class="word-progress-fill" id="wfill-${folder}" style="width:0%"></div></div>
    <div class="word-count" id="wcount-${folder}">0 / ${REQUIRED_SAMPLES} recordings</div>
    <div class="word-actions">
      <button class="btn btn-red" onclick="startRecordSession('${phrase}')">● Record</button>
      <button class="btn btn-ghost" onclick="clearWordSamples('${phrase}')" title="Clear recordings">🗑</button>
    </div>
  `;
  grid.appendChild(card);
  wordCounts[phrase] = 0;
}

function updateProbBars(vocabulary) {
  const grid = document.getElementById('probGrid');
  if (!grid || !vocabulary) return;
  vocabulary.forEach(word => {
    const key = word.replace(/\s+/g, '');
    if (!document.getElementById('bar-' + key)) {
      const row = document.createElement('div');
      row.className = 'prob-row';
      row.innerHTML = `
        <div class="prob-label" title="${word}">${word}</div>
        <div class="prob-track"><div class="prob-fill" id="bar-${key}" style="width:2%"></div></div>
        <div class="prob-pct" id="val-${key}">0.0%</div>
      `;
      grid.appendChild(row);
    }
  });
}

// ════════════════════════════════════════════════
// TRAIN MODEL
// ════════════════════════════════════════════════
btnTrain.addEventListener('click', async () => {
  if (!confirm('Start 100-epoch training? This may take 30–60 seconds on CPU.')) return;
  btnTrain.disabled = true; btnTrain.textContent = '⚡ Training…';
  trainReady.className = 'status-bar warn'; trainReady.textContent = '🔄 Training in progress…';
  epochOuter.style.display = 'block'; epochBar.style.width = '0%';
  trainLog.style.display = 'block'; trainLog.textContent = 'Launching training engine…';

  // Animate progress bar fake (real progress in result)
  let fakeEpoch = 0;
  const ticker = setInterval(() => {
    fakeEpoch = Math.min(fakeEpoch + 1, 92);
    epochBar.style.width = fakeEpoch + '%';
    trainLog.textContent = `Epoch ${fakeEpoch}/100 — training…`;
  }, 600);

  try {
    const res = await fetch('/api/train', { method: 'POST' });
    clearInterval(ticker);
    const data = await res.json();
    if (data.status === 'success') {
      const r = data.training_result;
      epochBar.style.width = '100%';
      trainLog.textContent = `✅ Done! Best Val Acc: ${r.best_val_acc}% | Time: ${r.training_time_s}s | Samples: ${r.total_samples}`;
      trainReady.className = 'status-bar'; trainReady.textContent = `🎉 Model trained! Val Accuracy: ${r.best_val_acc}%`;
      btnTrain.textContent = '✅ Trained — Re-Train';
    } else {
      epochBar.style.width = '0%';
      trainLog.textContent = '❌ ' + data.message;
      trainReady.className = 'status-bar err'; trainReady.textContent = '❌ Training failed: ' + data.message;
      btnTrain.textContent = '⚡ Train 100 Epochs on All Recordings';
    }
  } catch(e) {
    clearInterval(ticker);
    trainLog.textContent = '❌ Network error: ' + e.message;
    trainReady.className = 'status-bar err'; trainReady.textContent = '❌ ' + e.message;
    btnTrain.textContent = '⚡ Train 100 Epochs on All Recordings';
  }
  btnTrain.disabled = false;
});

// ════════════════════════════════════════════════
// VOICE TOGGLE (both tabs)
// ════════════════════════════════════════════════
btnVoice.addEventListener('click', () => {
  isVoiceEnabled = !isVoiceEnabled;
  btnVoice.textContent = isVoiceEnabled ? '🔊 Voice ON' : '🔇 Voice OFF';
});
if (btnVoiceT) btnVoiceT.addEventListener('click', () => {
  isVoiceEnabled = !isVoiceEnabled;
  btnVoiceT.textContent = isVoiceEnabled ? '🔊 Voice ON' : '🔇 Voice OFF';
  btnVoice.textContent  = isVoiceEnabled ? '🔊 Voice ON' : '🔇 Voice OFF';
});

// ════════════════════════════════════════════════
// ── LIVE TESTING TAB ──
// ════════════════════════════════════════════════
if (btnCameraT) btnCameraT.addEventListener('click', async () => {
  if (!isTestCameraActive) {
    try {
      btnCameraT.textContent = '⌛ Starting…';
      testStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' }, audio: false });
      videoElT.srcObject = testStream;
      await videoElT.play();
      if (standbyT) standbyT.style.display = 'none';
      resizeCanvas(videoElT, canvasElT);
      testFaceMesh = buildFaceMesh(onTestFrame);
      isTestCameraActive = true;
      btnCameraT.textContent = '🛑 Stop Camera';
      startTestFrameLoop();
      setTimeout(() => {
        if (latestTestLandmarks) extractorT.calibrateNeutral(latestTestLandmarks, videoElT.videoWidth || 640, videoElT.videoHeight || 480);
      }, 1500);
    } catch(e) { alert('Camera error: ' + e.message); btnCameraT.textContent = '⚡ Start Camera'; }
  } else {
    isTestCameraActive = false;
    if (testStream) { testStream.getTracks().forEach(t => t.stop()); testStream = null; }
    videoElT.srcObject = null;
    ctxT.clearRect(0, 0, canvasElT.width, canvasElT.height);
    if (standbyT) standbyT.style.display = 'flex';
    btnCameraT.textContent = '⚡ Start Camera';
  }
});

function startTestFrameLoop() {
  let fcTest = 0, lastFpsTest = performance.now();
  const loop = async () => {
    if (!isTestCameraActive) return;
    if (videoElT.readyState >= 2 && testFaceMesh) {
      try { await testFaceMesh.send({ image: videoElT }); } catch(e) {}
    }
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

function onTestFrame(results) {
  const now = performance.now();
  if (!results.multiFaceLandmarks?.length) { ctxT.clearRect(0, 0, canvasElT.width, canvasElT.height); return; }
  latestTestLandmarks = results.multiFaceLandmarks[0];
  drawMesh(latestTestLandmarks, ctxT, canvasElT, false);
  const feat = extractorT.extractFrameFeatures(latestTestLandmarks, videoElT.videoWidth || 640, videoElT.videoHeight || 480);
  if (!feat) return;
  kpiApT.textContent  = (feat.metrics.vertical_aperture * 100).toFixed(1) + '%';
  kpiVelT.textContent = feat.metrics.velocity.toFixed(3);

  if (feat.metrics.consecutive_rest >= 5) lastSpokenWord = '';
  if (!isInferring && extractorT.isBufferReady() && (now - lastInferenceTime) > 160) {
    lastInferenceTime = now;
    runInference(extractorT.getTemporalTensor());
  }
}

if (btnCalT) btnCalT.addEventListener('click', () => {
  if (latestTestLandmarks) {
    extractorT.calibrateNeutral(latestTestLandmarks, videoElT.videoWidth || 640, videoElT.videoHeight || 480);
    btnCalT.textContent = '✅ Done!'; setTimeout(() => btnCalT.textContent = '🎯 Calibrate', 1500);
  } else { alert('Start camera first.'); }
});

// ════════════════════════════════════════════════
// INFERENCE ENGINE
// ════════════════════════════════════════════════
async function runInference(tensor) {
  if (isInferring) return;
  isInferring = true;
  const t0 = performance.now();
  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tensor })
    });
    const data = await res.json();
    if (data.status !== 'success') return;

    const latency = Math.round(performance.now() - t0);
    kpiLatT.textContent = latency + 'ms';

    // Update probability bars
    if (data.probabilities) {
      let topWord = '', topProb = 0;
      for (const [word, prob] of Object.entries(data.probabilities)) {
        const key = word.replace(/\s+/g, '');
        const fill = document.getElementById('bar-' + key);
        const val  = document.getElementById('val-' + key);
        const pct  = (prob * 100).toFixed(1);
        if (fill) { fill.style.width = Math.max(pct, 1) + '%'; fill.classList.toggle('top', word === data.predicted_word); }
        if (val)  val.textContent = pct + '%';
        if (prob > topProb) { topProb = prob; topWord = word; }
      }
    }

    const now = performance.now();
    if (data.is_active_speech && data.confidence >= 0.60 && data.predicted_word !== 'RESTING') {
      const isDupe = data.predicted_word === lastSpokenWord && (now - lastSpokenTime) < 1000;
      if (!isDupe) {
        lastSpokenWord = data.predicted_word;
        lastSpokenTime = now;

        // Update billboard
        billboard.classList.add('active');
        bbWord.textContent = data.predicted_word;
        bbSentence.textContent = '"' + data.synthesized_sentence + '"';
        confBadge.textContent = (data.confidence * 100).toFixed(1) + '% Conf';

        // Running sentence
        if (contText && data.continuous_sentence) contText.textContent = '"' + data.continuous_sentence + '"';

        // TTS
        speak(data.synthesized_sentence, 'patient');

        // Chat
        appendBubble('patient', data.predicted_word, data.synthesized_sentence, data.confidence);
        if (data.ai_dialogue_reply) {
          setTimeout(() => {
            appendBubble('ai', '', data.ai_dialogue_reply, 0);
            speak(data.ai_dialogue_reply, 'ai');
          }, 350);
        }

        // Suggestions
        if (data.suggested_next_words?.length) renderChips(data.suggested_next_words);

        // History
        addHistory(data.predicted_word, data.synthesized_sentence, data.confidence);

        // Flush buffer
        extractorT.flushBufferAfterDetection(6);
      }
    } else {
      billboard.classList.remove('active');
      if (data.predicted_word === 'RESTING') {
        bbWord.textContent = 'RESTING';
        confBadge.textContent = 'Awaiting Movement';
      }
    }
  } catch(e) { console.error('Inference error:', e); }
  finally { isInferring = false; }
}

// ════════════════════════════════════════════════
// TTS QUEUE (Chrome-safe, non-blocking)
// ════════════════════════════════════════════════
const ttsQueue = [];
let ttsBusy = false;
let ttsVoices = [];
if ('speechSynthesis' in window) {
  const loadVoices = () => { ttsVoices = window.speechSynthesis.getVoices(); };
  loadVoices();
  window.speechSynthesis.onvoiceschanged = loadVoices;
  setInterval(() => { if (window.speechSynthesis.speaking) { window.speechSynthesis.pause(); window.speechSynthesis.resume(); } }, 5000);
}
function speak(text, sender) {
  if (!isVoiceEnabled || !text || !('speechSynthesis' in window)) return;
  ttsQueue.push({ text, sender });
  processTTS();
}
function processTTS() {
  if (ttsBusy || !ttsQueue.length) return;
  ttsBusy = true;
  const { text, sender } = ttsQueue.shift();
  const utt = new SpeechSynthesisUtterance(text);
  utt.lang = 'en-US';
  utt.rate  = sender === 'ai' ? 1.05 : 1.0;
  utt.pitch = sender === 'ai' ? 1.1  : 0.95;
  const aiVoice = ttsVoices.find(v => v.lang.startsWith('en') && (v.name.includes('Zira') || v.name.includes('Victoria') || v.name.includes('Google UK')));
  const patVoice = ttsVoices.find(v => v.lang.startsWith('en') && (v.name.includes('David') || v.name.includes('Google US')));
  if (sender === 'ai' && aiVoice) utt.voice = aiVoice;
  else if (patVoice) utt.voice = patVoice;
  utt.onend = utt.onerror = () => { ttsBusy = false; setTimeout(processTTS, 100); };
  window.speechSynthesis.speak(utt);
}

// ════════════════════════════════════════════════
// CHAT BUBBLES
// ════════════════════════════════════════════════
function appendBubble(sender, word, text, conf) {
  if (!chatArea || !text) return;
  const b = document.createElement('div');
  const t = new Date().toTimeString().split(' ')[0];
  b.className = 'bubble bubble-' + sender;
  b.innerHTML = sender === 'patient'
    ? `<div class="bubble-meta">👤 Patient (${word}) ${conf > 0 ? '• ' + (conf*100).toFixed(0)+'%' : ''} • ${t}</div><div>"${text}"</div>`
    : `<div class="bubble-meta">🤖 AI Care Partner • ${t}</div><div>${text}</div>`;
  chatArea.appendChild(b);
  if (chatArea.children.length > 20) chatArea.removeChild(chatArea.firstChild);
  chatArea.scrollTop = chatArea.scrollHeight;
}

// ════════════════════════════════════════════════
// SUGGESTION CHIPS
// ════════════════════════════════════════════════
function renderChips(words) {
  if (!chipRow) return;
  chipRow.innerHTML = '';
  words.forEach(w => {
    const c = document.createElement('button');
    c.className = 'chip'; c.dataset.word = w; c.textContent = w;
    c.addEventListener('click', () => runDemo(w));
    chipRow.appendChild(c);
  });
}
if (chipRow) chipRow.addEventListener('click', e => {
  const w = e.target.closest('.chip')?.dataset.word;
  if (w) runDemo(w);
});

// ════════════════════════════════════════════════
// HISTORY LOG
// ════════════════════════════════════════════════
function addHistory(word, sentence, conf) {
  if (!histList) return;
  const item = document.createElement('div');
  item.className = 'history-item';
  const t = new Date().toTimeString().split(' ')[0];
  item.innerHTML = `<span><strong>${word}:</strong> "${sentence}"</span><span class="hi-time">${(conf*100).toFixed(0)}% • ${t}</span>`;
  histList.prepend(item);
  if (histList.children.length > 10) histList.removeChild(histList.lastChild);
}

// ════════════════════════════════════════════════
// DEMO / SIMULATION
// ════════════════════════════════════════════════
async function runDemo(phrase) {
  if (standbyT) standbyT.style.display = 'none';
  resizeCanvas(videoElT, canvasElT);
  const words = phrase.trim().split(/\s+/);
  const n = words.length;
  const seqLen = 30;
  const trajectory = [];
  for (let t = 0; t < seqLen; t++) {
    const p = t / seqLen;
    let h = 0.05;
    for (let s = 0; s < n; s++) {
      const center = (s + 0.5) / n;
      const sigma = 0.10 / Math.sqrt(n);
      h += 0.34 * Math.exp(-((p - center) ** 2) / (2 * sigma ** 2));
    }
    const w = 0.45 + 0.08 * Math.sin(p * Math.PI * n);
    const dh = Math.cos(p * Math.PI * n) * 0.38;
    trajectory.push([h, w, h/(w+1e-5), h*1.5+.08, .05, .06, .02, Math.PI*h/2*w/2, dh, dh*.25, -Math.sin(p*Math.PI*n)*.1, 0]);
  }
  // Animate canvas
  let fr = 0;
  const anim = setInterval(() => {
    if (fr >= seqLen) { clearInterval(anim); return; }
    const [h, w] = trajectory[fr];
    ctxT.clearRect(0, 0, canvasElT.width, canvasElT.height);
    const cx = canvasElT.width/2, cy = canvasElT.height/2;
    const rx = w * canvasElT.width / 2.2, ry = Math.max(h * canvasElT.height * 1.3, 8);
    [['#00f2fe',3,rx,ry], ['#a855f7',2,rx*.65,Math.max(ry*.55,4)]].forEach(([col,lw,rx2,ry2]) => {
      ctxT.beginPath(); ctxT.ellipse(cx,cy,rx2,ry2,0,0,Math.PI*2);
      ctxT.strokeStyle=col; ctxT.lineWidth=lw; ctxT.shadowColor=col; ctxT.shadowBlur=12; ctxT.stroke(); ctxT.shadowBlur=0;
    });
    kpiApT.textContent = (trajectory[fr][0]*100).toFixed(1)+'%';
    fr++;
  }, 24);
  await runInference(trajectory);
}

async function runInteractiveDemo() {
  await runDemo('HELP');
  setTimeout(() => runDemo('WATER'), 2500);
  setTimeout(() => runDemo('THANK YOU'), 5000);
}

if (btnDemo)  btnDemo.addEventListener('click', runInteractiveDemo);
if (btnDemoT) btnDemoT.addEventListener('click', runInteractiveDemo);

// ════════════════════════════════════════════════
// LOAD INITIAL DATASET COUNTS
// ════════════════════════════════════════════════
async function loadDatasetCounts() {
  try {
    const res = await fetch('/api/dataset_stats');
    const data = await res.json();
    const realRec = data.real_recordings || {};
    Object.entries(realRec).forEach(([word, count]) => {
      wordCounts[word] = count;
      updateWordCardUI(word, count);
    });
    checkTrainReady();
  } catch(e) { console.warn('Could not load dataset stats:', e); }
}

loadDatasetCounts();
