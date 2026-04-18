import tensorflow as tf
import numpy as np
import time
import os

# Load model
model = tf.keras.models.load_model("autoencoder.h5", compile=False)
model.compile(optimizer='adam', loss='mse')

# Get concrete function from model (fixes shape issues)
input_spec = tf.TensorSpec(shape=[1, 128, 85, 1], dtype=tf.float32, name='input')
concrete_func = tf.function(model).get_concrete_function(input_spec)

# Convert with SELECT_TF_OPS
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS   # Allows unsupported ops like Conv2D with dynamic shapes
]
# Optional: reduce size further with float16 quantization
# converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

# Save
with open("autoencoder_quant.tflite", "wb") as f:
    f.write(tflite_model)

# Check size
size_bytes = os.path.getsize("autoencoder_quant.tflite")
size_mb = size_bytes / (1024 * 1024)
print(f"Quantized TFLite model size: {size_mb:.2f} MB")

# Benchmark inference
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Warm-up
dummy_input = np.random.random(input_details[0]['shape']).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], dummy_input)
interpreter.invoke()

times = []
for _ in range(100):
    start = time.time()
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()
    _ = interpreter.get_tensor(output_details[0]['index'])
    times.append(time.time() - start)

avg_time_ms = np.mean(times) * 1000
print(f"Average inference time on CPU: {avg_time_ms:.2f} ms")

# Save benchmark
with open("benchmark.txt", "w") as f:
    f.write(f"TFLite model size: {size_mb:.2f} MB\n")
    f.write(f"Average inference time (CPU): {avg_time_ms:.2f} ms\n")
    f.write("Note: Conversion used SELECT_TF_OPS to handle dynamic upsampling shapes.\n")