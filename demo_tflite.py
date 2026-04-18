import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf

# --- CONFIG ---
SR = 22050
DURATION = 1.0
N_MELS = 128
MAX_LEN = 85
MODEL_PATH = "autoencoder_quant.tflite"

# Load threshold
threshold = 0.02

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load silence spectrogram (same as training)
def extract_spectrogram(y):
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

silence_audio, _ = librosa.load("./dataset/silence.flac", sr=SR)
silence_spec = extract_spectrogram(silence_audio)
if silence_spec.shape[1] < MAX_LEN:
    pad = MAX_LEN - silence_spec.shape[1]
    silence_spec = np.pad(silence_spec, ((0,0),(0,pad)), constant_values=np.min(silence_spec))

def get_noisy_silence(width):
    slice_ = silence_spec[:, :width].copy()
    noise = np.random.normal(0, 0.01, slice_.shape)
    return slice_ + noise

def preprocess_audio(audio):
    spec = extract_spectrogram(audio)
    if spec.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - spec.shape[1]
        pad_left = np.random.randint(0, pad_width + 1)
        pad_right = pad_width - pad_left
        left_pad = get_noisy_silence(pad_left)
        right_pad = get_noisy_silence(pad_right)
        spec = np.concatenate([left_pad, spec, right_pad], axis=1)
    else:
        spec = spec[:, :MAX_LEN]
    spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec) + 1e-8)
    spec = spec[np.newaxis, ..., np.newaxis].astype(np.float32)
    return spec

print(f"Listening... (threshold = {threshold:.6f}) Press Ctrl+C to stop")

try:
    while True:
        audio = sd.rec(int(SR * DURATION), samplerate=SR, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()
        spec = preprocess_audio(audio)

        interpreter.set_tensor(input_details[0]['index'], spec)
        interpreter.invoke()
        recon = interpreter.get_tensor(output_details[0]['index'])

        mse = np.mean((spec - recon) ** 2)
        if mse > threshold:
            print(f"🚨 ANOMALY | error: {mse:.6f}")
        else:
            print(f"✅ normal  | error: {mse:.6f}")
except KeyboardInterrupt:
    print("\nStopped.")
