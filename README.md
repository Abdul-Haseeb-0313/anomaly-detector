
# Audio Anomaly Detector

A lightweight CNN autoencoder that detects anomalous sounds on low‑power edge devices. No raw audio leaves the device.

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Reproducing the Workflow](#reproducing-the-workflow)
  - [1. Dataset Preparation](#1-dataset-preparation)
  - [2. Training the Model](#2-training-the-model)
  - [3. Evaluation & Threshold Calibration](#3-evaluation--threshold-calibration)
  - [4. Convert to TensorFlow Lite](#4-convert-to-tensorflow-lite)
  - [5. Live Inference Demo](#5-live-inference-demo)
- [Model Performance](#model-performance)
- [Edge Deployment Plan](#edge-deployment-plan)
- [Explainability & Failure Modes](#explainability--failure-modes)
- [Social Impact & Privacy](#social-impact--privacy)
- [Credits & License](#credits--license)

---

## Overview

- **Model**: Convolutional autoencoder (3 encoder + 3 decoder layers)
- **Input**: 128×85 mel‑spectrogram (≈1 second of audio at 22.05 kHz)
- **Output**: Reconstruction of the input; high error → anomaly
- **Optimization**: Quantized TensorFlow Lite model, < 3 MB, < 20 ms inference on CPU
- **Use Case**: Detect safety‑critical sounds in community spaces without compromising privacy

## Repository Structure

| File | Description |
|------|-------------|
| `model_training.py` | Script to train the autoencoder on your dataset |
| `evaluate_model.py` | Computes reconstruction errors, sets threshold, estimates AUC/F1 |
| `convert_to_tflite.py` | Converts the trained `.h5` model to quantized TFLite |
| `demo_tflite.py` | Real‑time anomaly detection using microphone + TFLite |
| `autoencoder.h5` | Trained Keras model (for reference) |
| `autoencoder_quant.tflite` | Quantized TFLite model for edge deployment |
| `metrics.txt` | Evaluation metrics (AUC, F1) |
| `benchmark.txt` | TFLite model size and inference latency |
| `requirements.txt` | Python dependencies |

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/anomaly-detector.git
   cd anomaly-detector
   ```

2. **Create and activate a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Reproducing the Workflow

### 1. Dataset Preparation

The model was trained on the [ReaLISED](https://zenodo.org/records/6488321) dataset (Real Life Indoor Sound Event Dataset).  

Place all `.flac` audio files inside a folder named `dataset/`.  
The folder must contain a `silence.flac` file (used for padding).  
Example structure:
```
dataset/
├── silence.flac
├── sound1.flac
├── sound2.flac
└── ...
```

### 2. Training the Model

Run the training script:
```bash
python model_training.py
```
This will:
- Extract mel‑spectrograms from all audio files
- Pad shorter clips with noisy silence (silence padding with small Gaussian noise)
- Normalise each spectrogram to [0,1]
- Train the autoencoder for 10 epochs
- Save the trained model as `autoencoder.h5`

### 3. Evaluation & Threshold Calibration

```bash
python evaluate_model.py
```
- Loads the trained model and dataset
- Splits data into train/validation (80/20)
- Computes reconstruction MSE on validation normal samples
- Sets anomaly threshold = 99th percentile of those errors → saved to `threshold.npy`
- (Optional) Simulates anomalies by adding Gaussian noise to estimate AUC and F1 → saved to `metrics.txt`

### 4. Convert to TensorFlow Lite

```bash
python convert_to_tflite.py
```
- Converts `autoencoder.h5` to a quantized TFLite model
- Uses **SELECT_TF_OPS** to handle dynamic upsampling shapes
- Measures model size and average inference time on CPU → saved to `benchmark.txt`

### 5. Live Inference Demo

Make sure your microphone is accessible, then run:
```bash
python demo_tflite.py
```
- Captures 1‑second audio chunks continuously
- Preprocesses each chunk identically to training
- Runs inference using the quantized TFLite model
- Prints `✅ normal` or `🚨 ANOMALY` based on the pre‑computed threshold

Press `Ctrl+C` to stop.

## Model Performance

| Metric | Value |
|--------|-------|
| **TFLite model size** | `2.3 MB` (see `benchmark.txt`) |
| **Inference latency (CPU)** | `12.5 ms` per 1‑sec chunk |
| **Threshold (99th %ile)** | `0.000324` |
| **Simulated AUC** | `0.91` |
| **Simulated F1 Score** | `0.87` |

*Note: AUC and F1 are estimated using synthetic noise anomalies; real‑world performance will vary.*

## Edge Deployment Plan

- **Runtime**: TensorFlow Lite (C++ API) on a Raspberry Pi 4 or similar ARM Cortex‑A device.
- **Resource Constraints**: 1 CPU core, < 500 MB RAM, < 5 W power budget.
- **Inference Trigger**: Continuous streaming – every 1 second a new spectrogram chunk is processed.
- **Data Flow**:
  1. Microphone → 1‑second audio buffer
  2. On‑device mel‑spectrogram + padding/normalisation
  3. TFLite model inference → reconstruction error
  4. Error > threshold → log alert locally (optional: send anonymised flag)
  5. **No raw audio is stored or transmitted.**

## Explainability & Failure Modes

See EXPLAINABILITY.md for a plain‑language guide (with ASCII diagram) and an honest discussion of what can go wrong (e.g., false alarms on new but harmless sounds) and how to mitigate it.

## Social Impact & Privacy

This detector can be deployed in community centres, shelters, or public spaces to flag sudden loud events without recording conversations.

## Credits & License

- **Dataset**: ReaLISED (see CREDITS.md for full citation)
- **Libraries**: TensorFlow, Librosa, NumPy, scikit‑learn, sounddevice (see `requirements.txt` for versions)

---
*Built for the Socio‑Matics Edge AI Engineer Intern Challenge – April 2026*
