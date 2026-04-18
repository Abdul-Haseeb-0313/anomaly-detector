# Model Design, Edge Deployment & Explainability

## Part 1 — Model Design & Optimization

### Architecture Choice and Rationale

The detector uses a **convolutional autoencoder** trained on mel‑spectrograms of normal environmental audio. An autoencoder learns to compress and reconstruct input data. When presented with abnormal sounds, reconstruction error increases, flagging an anomaly.

**Why a custom CNN autoencoder instead of MobileNet or a pretrained model?**
- The task is **unsupervised** – we only have normal data. Autoencoders are a natural fit.
- Input dimensions are small (128×85×1), so a custom lightweight CNN is more efficient than adapting a larger architecture.
- The model is trained from scratch on domain‑specific spectrograms, avoiding unnecessary complexity.

**Layer summary (encoder → decoder):**
- 3 convolutional layers (32, 64, 128 filters) with max‑pooling
- 3 upsampling + convolutional layers for reconstruction
- Final `sigmoid` activation to output values in [0,1]

### Quantization and Size/Latency Trade‑off

The trained Keras model (`autoencoder.h5`) is converted to **TensorFlow Lite** with **dynamic range quantization** (`tf.lite.Optimize.DEFAULT`). Due to the upsampling operations, we enable **SELECT_TF_OPS** to allow a small number of TensorFlow ops to run alongside TFLite builtins.

| Metric | Value |
|--------|-------|
| TFLite model size | **0.33 MB** (target < 50 MB) |
| Average inference time (CPU) | **14.57 ms** (target < 100 ms) |

**Accuracy trade‑off:**  
Quantization reduces model size by ~4× compared to the float32 `.h5` file. Reconstruction error distribution remains nearly identical; the threshold and anomaly detection performance are unaffected for this application.

### Benchmark Metrics

| Metric | Value |
|--------|-------|
| Threshold (99th percentile of normal MSE) | 0.002099 |
| Simulated AUC (normal vs. Gaussian‑noise anomalies) | 1.0000 |
| Simulated F1 score at threshold | 0.9946 |

*Note: AUC and F1 are estimated by adding synthetic Gaussian noise to validation samples. This gives a rough upper bound; real‑world performance will be lower due to more subtle anomalies.*

### Dataset Citation and Social‑Good Use Case

**Dataset:** ReaLISED (Real Life Indoor Sound Event Dataset)  
- Contains indoor environmental sounds recorded in real homes.  
- License: Creative Commons Attribution 4.0 International (CC BY 4.0).  

**Social‑good connection:** The detector can be deployed in community centers, shelters, or public libraries to identify sudden loud events (crashes, aggressive shouting, equipment failure) without recording conversations. This enables staff to respond quickly while preserving privacy of vulnerable populations.

---

## Part 2 — Edge Deployment Approach

### Target Runtime

**TensorFlow Lite** is chosen because:
- Lightweight interpreter runs on ARM Cortex‑A CPUs (Raspberry Pi, Android, embedded Linux).
- Supports quantization and select TensorFlow ops.
- No cloud dependency – all inference happens locally.

### Resource Constraints Assumed

| Resource | Assumed Limit |
|----------|---------------|
| CPU | 1 ARM Cortex‑A53/A72 core @ 1.0‑1.5 GHz |
| RAM | 512 MB total; < 100 MB available for application |
| Power budget | < 5 W (passively cooled device) |

The TFLite model requires ~10‑20 MB of RAM at runtime and <15 ms per inference, well within these constraints.

### Inference Trigger

**Continuous streaming** – a 1‑second audio buffer is captured from the microphone, converted to a mel‑spectrogram, and fed to the model. This repeats indefinitely. The system can easily be modified to be **event‑driven** (e.g., triggered by a sound level threshold) to save power.

### On‑Device vs. Transmitted Data

| Data Type | Handled On‑Device | Transmitted |
|-----------|-------------------|-------------|
| Raw audio | Processed and discarded immediately | Never |
| Mel‑spectrogram | Used for inference only | Never |
| Reconstruction error | Compared to threshold | No |
| Anomaly flag (binary) | Logged locally | Optionally sent (if consented) |

**Privacy guarantee:** No personally identifiable information (PII) or recoverable audio leaves the device.

---

## Part 3 — Explainability Brief (Plain‑Language)

### How the System Detects Anomalies (For a Non‑Technical Community Liaison)

Imagine you have a machine that knows what “normal” sounds look like as pictures called spectrograms. When a new sound arrives, the machine tries to draw the same picture from memory. If the new sound is familiar, the drawing comes out very similar. If it’s unfamiliar—like a loud bang or a scream—the drawing is messy and different.

The machine compares the original picture with its drawing and calculates a **difference score**. If that score is higher than a preset limit, it says **“Anomaly!”** Otherwise, it says **“Normal.”**

This all happens inside a small computer (like a Raspberry Pi) that sits in the community space. No sound is ever recorded or sent anywhere. Only a simple “normal” or “anomaly” note is logged.

### Simple Data Flow Diagram (ASCII)

```
   Microphone
       │
       ▼
 [ 1‑second sound chunk ]
       │
       ▼
┌─────────────────────┐
│ Turn sound into     │
│ spectrogram picture │
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│ Autoencoder tries   │
│ to redraw picture   │
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│ Compare original    │
│ with redrawn picture│  → Difference Score
└─────────────────────┘
       │
       ▼
 Score > Limit? ──── Yes ──► 🚨 Anomaly logged
       │
       No
       │
       ▼
   ✅ Normal
```

### One Realistic Failure Mode and Mitigation

**Failure mode:** False alarms on new but harmless sounds.  
Example: The system has never heard a vacuum cleaner or a doorbell. It may flag these as anomalies even though they are not safety threats.

**Mitigation plan:**
- **Adaptive threshold:** Over time, the system can slowly adjust its baseline (with community administrator approval) to incorporate new normal sounds.
- **On‑device confirmation:** The device could store a short, encrypted snippet (never transmitted) that a trusted person can review later and label as “normal.” This label can be used to fine‑tune the model locally without exposing any audio.

---

*This document serves as the technical and plain‑language explanation for the Socio‑Matics Edge AI Engineer Intern Challenge.*
