import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, auc

# --- CONFIG ---
DATA_PATH = "./ReaLISED_Dataset"          # path to your .flac files
SILENCE_PATH = "./ReaLISED_Dataset/silence.flac"
SR = 22050
N_MELS = 128
MAX_LEN = 85                     # must match training
MODEL_PATH = "autoencoder.h5"

# --- Helper functions (exactly as you used for training) ---
def extract_spectrogram(file_path):
    y, _ = librosa.load(file_path, sr=SR)
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def load_silence_spec(max_len):
    spec = extract_spectrogram(SILENCE_PATH)
    if spec.shape[1] < max_len:
        pad = max_len - spec.shape[1]
        spec = np.pad(spec, ((0,0),(0,pad)), constant_values=np.min(spec))
    return spec

def get_noisy_silence_slice(silence_spec, width):
    slice_ = silence_spec[:, :width].copy()
    noise = np.random.normal(0.0, 0.01, slice_.shape)
    return slice_ + noise

def preprocess_spec(spec, sil_spec):
    if spec.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - spec.shape[1]
        pad_left = np.random.randint(0, pad_width + 1)
        pad_right = pad_width - pad_left
        left_pad = get_noisy_silence_slice(sil_spec, pad_left)
        right_pad = get_noisy_silence_slice(sil_spec, pad_right)
        spec = np.concatenate([left_pad, spec, right_pad], axis=1)
    else:
        spec = spec[:, :MAX_LEN]
    spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec) + 1e-8)
    return spec

# --- Load and preprocess all files ---
print("Loading dataset...")
sil_spec = load_silence_spec(MAX_LEN)
all_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".flac") and f != "silence.flac"]
specs = []
for f in all_files:
    spec = extract_spectrogram(os.path.join(DATA_PATH, f))
    spec = preprocess_spec(spec, sil_spec)
    spec = spec[..., np.newaxis]   # add channel dim
    specs.append(spec)
X = np.array(specs)
print(f"Total samples: {X.shape}")

# --- Train/Validation split (80/20) ---
np.random.seed(42)
indices = np.random.permutation(len(X))
split = int(0.8 * len(X))
train_idx, val_idx = indices[:split], indices[split:]
X_train, X_val = X[train_idx], X[val_idx]

# --- Load model ---
model = load_model(MODEL_PATH, compile=False)
model.compile(optimizer='adam', loss='mse')  # recompile for evaluation

# --- Compute reconstruction errors on validation normal data ---
print("Computing validation errors...")
reconstructions = model.predict(X_val, batch_size=32, verbose=1)
mse = np.mean(np.square(X_val - reconstructions), axis=(1,2,3))

# --- Set threshold as 99th percentile ---
threshold = np.percentile(mse, 99)
print(f"99th percentile threshold: {threshold:.6f}")

# Save threshold for later
np.save("threshold.npy", threshold)

# --- (Optional) Simulate anomalies by adding Gaussian noise to validation data ---
# This gives a rough AUC estimate if you have no real anomalies.
print("Simulating anomalies for AUC estimation...")
noise_factor = 0.5
X_val_noisy = X_val + noise_factor * np.random.normal(size=X_val.shape)
X_val_noisy = np.clip(X_val_noisy, 0., 1.)

reconstructions_noisy = model.predict(X_val_noisy, batch_size=32, verbose=1)
mse_noisy = np.mean(np.square(X_val_noisy - reconstructions_noisy), axis=(1,2,3))

# Labels: 0 = normal, 1 = anomaly
y_true = np.concatenate([np.zeros(len(mse)), np.ones(len(mse_noisy))])
y_scores = np.concatenate([mse, mse_noisy])

fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
print(f"Simulated AUC: {roc_auc:.4f}")

# Compute F1 score at chosen threshold
preds = (y_scores > threshold).astype(int)
from sklearn.metrics import f1_score
f1 = f1_score(y_true, preds)
print(f"F1 score at threshold {threshold:.6f}: {f1:.4f}")

# Save metrics to a file
with open("metrics.txt", "w") as f:
    f.write(f"Threshold (99th percentile): {threshold:.6f}\n")
    f.write(f"Simulated AUC: {roc_auc:.4f}\n")
    f.write(f"F1 score at threshold: {f1:.4f}\n")
    f.write("Note: AUC and F1 are estimated using synthetic noise anomalies.\n")

print("Evaluation complete. Metrics saved to metrics.txt, threshold to threshold.npy.")