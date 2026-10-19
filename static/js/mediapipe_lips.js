/**
 * SILENTVOICE AI • BIOMECHANICAL LIP KINEMATICS EXTRACTOR V3.0
 * Continuous Viseme Activity Detection (VAD) & Fluid Articulatory Streaming
 */

class LipKinematicsExtractor {
  constructor() {
    this.buffer = []; // Sliding window of 30 frames for real-time inference
    this.maxFrames = 30;

    this.prevH = 0.05;
    this.prevW = 0.45;
    this.prevDh = 0.0;

    // Calibration
    this.calibratedNeutralH = 0.05;
    this.calibratedNeutralW = 0.45;
    this.isCalibrated = false;

    // Articulatory Kinematics & Viseme Activity Detection (VAD)
    this.consecutiveRestFrames = 0;
    this.consecutiveActiveFrames = 0;
    this.currentVelocity = 0.0;
    this.currentAperture = 0.05;

    // Data Collection Studio (Record movements per word)
    this.isRecording = false;
    this.recordingFrames = [];
    this.recordingWord = "";
    this.onRecordingComplete = null;
  }

  // True Isotropic Metric Distance in Pixel Space
  distance(p1, p2, width = 640, height = 480) {
    const dx = (p1.x - p2.x) * width;
    const dy = (p1.y - p2.y) * height;
    const dz = ((p1.z || 0) - (p2.z || 0)) * width;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  calibrateNeutral(landmarks, width = 640, height = 480) {
    if (!landmarks || landmarks.length < 468) return;
    const faceH = this.distance(landmarks[10], landmarks[152], width, height) || 1.0;
    const faceW = this.distance(landmarks[234], landmarks[454], width, height) || 1.0;

    this.calibratedNeutralH = this.distance(landmarks[13], landmarks[14], width, height) / faceH;
    this.calibratedNeutralW = this.distance(landmarks[78], landmarks[308], width, height) / faceW;
    this.isCalibrated = true;
    console.log('[CALIBRATION] Baseline set: Height =', this.calibratedNeutralH.toFixed(3), 'Width =', this.calibratedNeutralW.toFixed(3));
  }

  startRecording(word, onComplete) {
    this.isRecording = true;
    this.recordingWord = word;
    this.recordingFrames = [];
    this.onRecordingComplete = onComplete;
    console.log(`[RECORDER] Started recording 30 frames for word: "${word}"`);
  }

  /**
   * Fluid Buffer Flush:
   * Discards the completed word from the sliding window immediately upon emission,
   * retaining only the trailing frames as transition context so the very next word
   * can be recognized without smearing or delay!
   */
  flushBufferAfterDetection(keepFrames = 6) {
    if (this.buffer.length > keepFrames) {
      this.buffer = this.buffer.slice(this.buffer.length - keepFrames);
    }
  }

  resetBufferToRest() {
    this.buffer = [];
  }

  extractFrameFeatures(landmarks, width = 640, height = 480) {
    if (!landmarks || landmarks.length < 468) return null;

    // Isotropic face normalization scale
    const faceH = this.distance(landmarks[10], landmarks[152], width, height) || 1.0;
    const faceW = this.distance(landmarks[234], landmarks[454], width, height) || 1.0;

    // 1. Geometric apertures normalized by facial scale
    const rawH = this.distance(landmarks[13], landmarks[14], width, height) / faceH;
    const rawW = this.distance(landmarks[78], landmarks[308], width, height) / faceW;

    const h = rawH;
    const w = rawW;
    const aspect = h / (w + 1e-5);
    const outerH = this.distance(landmarks[0], landmarks[17], width, height) / faceH;
    const upperThick = this.distance(landmarks[0], landmarks[13], width, height) / faceH;
    const lowerThick = this.distance(landmarks[14], landmarks[17], width, height) / faceH;

    // Corner angle
    const cornerY = ((landmarks[78].y + landmarks[308].y) / 2.0 - landmarks[13].y) * height / faceH;
    const area = Math.PI * (h / 2.0) * (w / 2.0);

    // Kinematic derivatives (velocities and accelerations)
    const dh = (h - this.prevH) * 5.0;
    const dw = (w - this.prevW) * 5.0;
    const d2h = (dh - this.prevDh) * 10.0;

    this.prevH = h;
    this.prevW = w;
    this.prevDh = dh;

    this.currentAperture = h;
    this.currentVelocity = Math.abs(dh);

    const zProtrusion = ((landmarks[13].z || 0) - (landmarks[168].z || 0)) * width / faceW;

    // 12-Dimensional Kinematic Articulatory Vector
    const featureVector = [
      h, w, aspect, outerH, upperThick, lowerThick,
      cornerY, area, dh, dw, d2h, zProtrusion
    ];

    // Viseme Activity Detection (VAD)
    const isMoving = Math.abs(dh) > 0.012 || h > (this.calibratedNeutralH * 1.25);
    if (isMoving) {
      this.consecutiveActiveFrames++;
      this.consecutiveRestFrames = 0;
    } else {
      this.consecutiveRestFrames++;
      this.consecutiveActiveFrames = 0;
    }

    // Handle Active Recording Mode
    if (this.isRecording) {
      this.recordingFrames.push(featureVector);
      if (this.recordingFrames.length >= this.maxFrames) {
        this.isRecording = false;
        const capturedTensor = this.recordingFrames.slice(0, this.maxFrames);
        if (this.onRecordingComplete) {
          this.onRecordingComplete(this.recordingWord, capturedTensor);
        }
      }
    }

    // Sliding window buffer for real-time continuous inference
    this.buffer.push(featureVector);
    if (this.buffer.length > this.maxFrames) {
      this.buffer.shift();
    }

    return {
      vector: featureVector,
      metrics: {
        vertical_aperture: h,
        horizontal_aperture: w,
        velocity: Math.abs(dh),
        is_moving: isMoving,
        is_calibrated: this.isCalibrated,
        consecutive_rest: this.consecutiveRestFrames,
        consecutive_active: this.consecutiveActiveFrames
      },
      recordingProgress: this.isRecording ? (this.recordingFrames.length / this.maxFrames) : 0
    };
  }

  isBufferReady() {
    // Allows fluid prediction once at least 14 frames have accumulated, with zero blocking
    return this.buffer.length >= 14;
  }

  getTemporalTensor() {
    if (this.buffer.length >= this.maxFrames) {
      return this.buffer.slice(-this.maxFrames);
    }
    // Front-pad with earliest frame to always deliver a complete (30, 12) temporal tensor
    const padded = [];
    const missing = this.maxFrames - this.buffer.length;
    const padFrame = this.buffer.length > 0 ? this.buffer[0] : new Array(12).fill(0);
    for (let i = 0; i < missing; i++) {
      padded.push(padFrame);
    }
    return padded.concat(this.buffer);
  }
}

window.LipKinematicsExtractor = LipKinematicsExtractor;
