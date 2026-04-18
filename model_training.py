import os
import numpy as np
import librosa
import matplotlib.pyplot as plt

DATA_PATH = "./dataset"
SILENCE_PATH = "./dataset/silence.flac"

SR = 22050
N_MELS = 128

# 🎧 Extract spectrogram

def extract_spectrogram(file_path):
    y, _ = librosa.load(file_path, sr=SR)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db



# Compute max_len (75 percentile)

def compute_max_len(data_path):
    lengths = []

    for file in os.listdir(data_path):
        if file.endswith(".flac") and file != "silence.flac":
            spec = extract_spectrogram(os.path.join(data_path, file))
            lengths.append(spec.shape[1])

    max_len = int(np.percentile(lengths, 75))
    print("Chosen max_len:", max_len)

    return max_len



# Load silence spectrogram
# Since the audio clips are of various length, am using audio of silence for padding

def load_silence_spec(max_len):
    spec = extract_spectrogram(SILENCE_PATH)

    if spec.shape[1] < max_len:
        pad = max_len - spec.shape[1]
        spec = np.pad(
            spec,
            ((0, 0), (0, pad)),
            constant_values=np.min(spec)
        )

    return spec


# Adding slight noise to silence

def get_noisy_silence_slice(silence_spec, width):
    slice_ = silence_spec[:, :width].copy()

    noise = np.random.normal(
        loc=0.0,
        scale=0.01,   # small noise
        size=slice_.shape
    )

    return slice_ + noise



#  Load dataset
def load_dataset(data_path):
    data = []

    max_len = compute_max_len(data_path)
    silence_spec = load_silence_spec(max_len)

    for file in os.listdir(data_path):
        if file.endswith(".flac") and file != "silence.flac":

            path = os.path.join(data_path, file)
            spec = extract_spectrogram(path)

            # Padding
            if spec.shape[1] < max_len:
                pad_width = max_len - spec.shape[1]

                # random split
                pad_left = np.random.randint(0, pad_width + 1)
                pad_right = pad_width - pad_left

                left_pad = get_noisy_silence_slice(silence_spec, pad_left)
                right_pad = get_noisy_silence_slice(silence_spec, pad_right)

                spec = np.concatenate([left_pad, spec, right_pad], axis=1)

            else:
                spec = spec[:, :max_len]

            # per-sample normalization
            spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec) + 1e-8)

            data.append(spec)

    data = np.array(data)
    data = data[..., np.newaxis]

    return data


# Visualization
def plot_spectrogram(spec):
    plt.figure(figsize=(10, 4))
    plt.imshow(spec, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title("Spectrogram")
    plt.show()


X = load_dataset(DATA_PATH)

print("Dataset shape:", X.shape)

# visualize sample
plot_spectrogram(X[0].squeeze())

from tensorflow.keras import layers, models
from tensorflow.keras import layers, models

input_shape = (128, 85, 1)

inputs = layers.Input(shape=input_shape)

# ---------------- ENCODER ----------------
x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
x = layers.MaxPooling2D((2,2), padding='same')(x)

x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2,2), padding='same')(x)

x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2,2), padding='same')(x)

# ---------------- DECODER ----------------
x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2,2))(x)

x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2,2))(x)

x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2,2))(x)

x = layers.Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

# FIX: crop width to 85
x = layers.Cropping2D(cropping=((0, 0), (0, 3)))(x)
# 88 → 85 (removes 3 columns from right)

# ---------------- MODEL ----------------
autoencoder = models.Model(inputs, x)

autoencoder.compile(
    optimizer='adam',
    loss='mse'
)

print(autoencoder.summary())

autoencoder.fit(
    X, X,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    shuffle=True
)

autoencoder.save("autoencoder.h5")

