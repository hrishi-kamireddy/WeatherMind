"""
Tiny Neural Network Trainer for ESP32 Sensor Prediction
========================================================
Trains a small NN on temperature, humidity, pressure, and lux data
to predict values 30-60 minutes ahead. Exports weights as a C header
file for Arduino/ESP32 deployment.

Dependencies:
    pip install kagglehub pandas numpy scikit-learn tensorflow

Usage:
    python train_model.py
"""

import numpy as np
import pandas as pd
import os

# ─── 1. Load Dataset ─────────────────────────────────────────────────────────
# print("=" * 60)
# print("Step 1: Loading dataset from Kaggle...")
# print("=" * 60)

# try:
#     import kagglehub
#     from kagglehub import KaggleDatasetAdapter

#     df = kagglehub.load_dataset(
#         KaggleDatasetAdapter.PANDAS,
#         "patrickfleith/temperature-humidity-pressure-illuminance",
#         "",
#     )
# except Exception as e:
#     print(f"Kaggle download failed: {e}")
#     print("Trying local fallback...")
#     # Fallback: if you downloaded manually, place CSV in same directory
#     df = pd.read_csv("sensor_data.csv")

# print(f"Dataset shape: {df.shape}")
# print(f"Columns: {list(df.columns)}")
# print(df.head())
import kagglehub
import os

print("Downloading dataset...")
dataset_path = kagglehub.dataset_download(
    "patrickfleith/temperature-humidity-pressure-illuminance"
)
print(f"Dataset downloaded to: {dataset_path}")

df = pd.read_csv(os.path.join(dataset_path, "DATA-large.CSV"))

# ─── 2. Preprocess ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 2: Preprocessing...")
print("=" * 60)

# Identify columns (adapt names to actual dataset)
# The dataset has: time, temperature, humidity, pressure, lux
col_map = {}
for col in df.columns:
    cl = col.lower().strip()
    if "temp" in cl:
        col_map["temperature"] = col
    elif "hum" in cl:
        col_map["humidity"] = col
    elif "press" in cl:
        col_map["pressure"] = col
    elif "lux" in cl or "illum" in cl or "light" in cl:
        col_map["lux"] = col
    elif "time" in cl or "date" in cl:
        col_map["time"] = col

print(f"Column mapping: {col_map}")

FEATURES = ["temperature", "humidity", "pressure", "lux"]
feature_cols = [col_map[f] for f in FEATURES]

# Extract feature data
data = df[feature_cols].values.astype(np.float32)

# Remove any NaN rows
mask = ~np.isnan(data).any(axis=1)
data = data[mask]
print(f"Clean samples: {data.shape[0]}")

# ─── 3. Determine sampling interval and prediction horizon ───────────────────
# The dataset samples every ~5 seconds. For 30-min prediction: 30*60/5 = 360 steps
# For 60-min prediction: 60*60/5 = 720 steps
# We'll predict at 30-min horizon (360 steps ahead)

# But we'll use a WINDOWED approach: take last N samples as input context,
# predict the value PREDICTION_STEPS ahead.

LOOKBACK = 12       # Use last 12 readings (~1 minute of context)
PREDICTION_STEP = 360  # Predict 360 steps ahead (~30 minutes)

print(f"Lookback window: {LOOKBACK} samples")
print(f"Prediction horizon: {PREDICTION_STEP} steps (~{PREDICTION_STEP * 5 / 60:.0f} min)")

# ─── 4. Normalize data ──────────────────────────────────────────────────────
# Save min/max for each feature (needed on ESP32 for denormalization)
feat_min = data.min(axis=0)
feat_max = data.max(axis=0)
feat_range = feat_max - feat_min
feat_range[feat_range == 0] = 1.0  # avoid division by zero

data_norm = (data - feat_min) / feat_range  # scale to [0, 1]

print(f"Feature mins:   {feat_min}")
print(f"Feature maxes:  {feat_max}")

# ─── 5. Create training sequences ───────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 3: Creating training sequences...")
print("=" * 60)

X_list = []
Y_list = []

for i in range(len(data_norm) - LOOKBACK - PREDICTION_STEP):
    # Input: flattened window of LOOKBACK * 4 features
    window = data_norm[i : i + LOOKBACK]
    # Output: the 4 feature values PREDICTION_STEP ahead of the window end
    target = data_norm[i + LOOKBACK + PREDICTION_STEP - 1]
    X_list.append(window.flatten())
    Y_list.append(target)

X = np.array(X_list, dtype=np.float32)
Y = np.array(Y_list, dtype=np.float32)

print(f"X shape: {X.shape}  (samples, features)")
print(f"Y shape: {Y.shape}  (samples, outputs)")

# Shuffle and split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.15, random_state=42
)
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# ─── 6. Build a TINY model ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 4: Building & training tiny NN...")
print("=" * 60)

import tensorflow as tf
from tensorflow import keras

INPUT_DIM = LOOKBACK * 4  # 12 * 4 = 48
HIDDEN1 = 16
HIDDEN2 = 8
OUTPUT_DIM = 4

model = keras.Sequential([
    keras.layers.Dense(HIDDEN1, activation="relu", input_shape=(INPUT_DIM,), name="hidden1"),
    keras.layers.Dense(HIDDEN2, activation="relu", name="hidden2"),
    keras.layers.Dense(OUTPUT_DIM, activation="sigmoid", name="output"),  # sigmoid since targets in [0,1]
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"],
)

model.summary()

# Calculate total parameters for ESP32 memory estimation
total_params = model.count_params()
memory_bytes = total_params * 4  # float32
print(f"\nTotal parameters: {total_params}")
print(f"Estimated memory: {memory_bytes} bytes ({memory_bytes/1024:.1f} KB)")

# Train
history = model.fit(
    X_train, Y_train,
    validation_split=0.15,
    epochs=50,
    batch_size=64,
    verbose=1,
)

# Evaluate
loss, mae = model.evaluate(X_test, Y_test, verbose=0)
print(f"\nTest Loss (MSE): {loss:.6f}")
print(f"Test MAE (normalized): {mae:.6f}")

# Denormalized MAE per feature
Y_pred = model.predict(X_test)
abs_errors = np.abs(Y_pred - Y_test) * feat_range
mean_abs_errors = abs_errors.mean(axis=0)
print(f"\nDenormalized MAE per feature:")
for i, name in enumerate(FEATURES):
    unit = ["°C", "%", "Pa", "lux"][i]
    print(f"  {name:>12s}: {mean_abs_errors[i]:.2f} {unit}")

# ─── 7. Export weights as C header ───────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 5: Exporting C header for ESP32...")
print("=" * 60)


def array_to_c(name, arr):
    """Convert numpy array to C array string."""
    flat = arr.flatten()
    lines = []
    lines.append(f"const float {name}[{len(flat)}] PROGMEM = {{")
    # Format in rows of 8
    for i in range(0, len(flat), 8):
        chunk = flat[i : i + 8]
        row = ", ".join(f"{v:.8f}f" for v in chunk)
        lines.append(f"    {row},")
    lines.append("};")
    return "\n".join(lines)


header_lines = []
header_lines.append("// ═══════════════════════════════════════════════════════════")
header_lines.append("// AUTO-GENERATED — Tiny NN Weights for ESP32")
header_lines.append("// Model: 48 -> 16 (ReLU) -> 8 (ReLU) -> 4 (Sigmoid)")
header_lines.append(f"// Prediction horizon: ~{PREDICTION_STEP * 5 / 60:.0f} minutes")
header_lines.append(f"// Total parameters: {total_params}")
header_lines.append("// ═══════════════════════════════════════════════════════════")
header_lines.append("#pragma once")
header_lines.append("#include <Arduino.h>")
header_lines.append("")
header_lines.append(f"#define NN_LOOKBACK    {LOOKBACK}")
header_lines.append(f"#define NN_INPUT_DIM   {INPUT_DIM}")
header_lines.append(f"#define NN_HIDDEN1     {HIDDEN1}")
header_lines.append(f"#define NN_HIDDEN2     {HIDDEN2}")
header_lines.append(f"#define NN_OUTPUT_DIM  {OUTPUT_DIM}")
header_lines.append("")

# Normalization parameters
header_lines.append("// Normalization parameters (min, range per feature)")
header_lines.append(f"const float FEAT_MIN[4] PROGMEM = {{{', '.join(f'{v:.6f}f' for v in feat_min)}}};")
header_lines.append(f"const float FEAT_RANGE[4] PROGMEM = {{{', '.join(f'{v:.6f}f' for v in feat_range)}}};")
header_lines.append("")

# Extract and export each layer
for layer in model.layers:
    w, b = layer.get_weights()
    lname = layer.name.upper()
    header_lines.append(f"// Layer: {layer.name}  shape: {w.shape}")
    header_lines.append(array_to_c(f"W_{lname}", w))
    header_lines.append("")
    header_lines.append(array_to_c(f"B_{lname}", b))
    header_lines.append("")

header_content = "\n".join(header_lines)

output_path = "nn_weights.h"
with open(output_path, "w") as f:
    f.write(header_content)

print(f"Saved: {output_path}")
print(f"File size: {os.path.getsize(output_path)} bytes")
print("\nDone! Copy nn_weights.h into your Arduino sketch folder.")