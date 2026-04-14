"""
WeatherMind — Tiny NN Trainer for ESP32
=======================================
Trains a small NN on temperature, humidity, pressure, and lux
to predict values ~30 minutes ahead. Exports weights as a C
header for ESP32 deployment. Computes feature importance.

pip install kagglehub pandas numpy scikit-learn tensorflow
python train_model.py
"""

import numpy as np
import pandas as pd
import os
import json
import glob

# ─── 1. Load Dataset ─────────────────────────────────────────
print("=" * 60)
print("Step 1: Loading dataset from Kaggle...")
print("=" * 60)

import kagglehub

print("Downloading dataset...")
dataset_path = kagglehub.dataset_download(
    "patrickfleith/temperature-humidity-pressure-illuminance"
)
print(f"Dataset downloaded to: {dataset_path}")

df = pd.read_csv(os.path.join(dataset_path, "DATA-large.CSV"))

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(df.head())

# ─── 2. Preprocess ───────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 2: Preprocessing...")
print("=" * 60)

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

data = df[feature_cols].values.astype(np.float32)
mask = ~np.isnan(data).any(axis=1)
data = data[mask]
print(f"Clean samples: {data.shape[0]}")

# ─── 3. Windowing ────────────────────────────────────────────
LOOKBACK = 12          # 12 readings = 1 minute of context
PREDICTION_STEP = 360  # 360 steps * 5s = 30 minutes ahead

print(f"Lookback window: {LOOKBACK} samples (~{LOOKBACK * 5}s)")
print(f"Prediction horizon: {PREDICTION_STEP} steps (~{PREDICTION_STEP * 5 / 60:.0f} min)")

# ─── 4. Normalize ────────────────────────────────────────────
feat_min = data.min(axis=0)
feat_max = data.max(axis=0)
feat_range = feat_max - feat_min
feat_range[feat_range == 0] = 1.0

data_norm = (data - feat_min) / feat_range

print(f"Feature mins:   {feat_min}")
print(f"Feature maxes:  {feat_max}")

# ─── 5. Create sequences ─────────────────────────────────────
print("\n" + "=" * 60)
print("Step 3: Creating training sequences...")
print("=" * 60)

X_list = []
Y_list = []
for i in range(len(data_norm) - LOOKBACK - PREDICTION_STEP):
    window = data_norm[i : i + LOOKBACK]
    target = data_norm[i + LOOKBACK + PREDICTION_STEP - 1]
    X_list.append(window.flatten())
    Y_list.append(target)

X = np.array(X_list, dtype=np.float32)
Y = np.array(Y_list, dtype=np.float32)
print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.15, random_state=42
)
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# ─── 6. Build & train ────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 4: Building & training tiny NN...")
print("=" * 60)

import tensorflow as tf
from tensorflow import keras

INPUT_DIM = LOOKBACK * 4  # 48
HIDDEN1 = 16
HIDDEN2 = 8
OUTPUT_DIM = 4

model = keras.Sequential([
    keras.layers.Dense(HIDDEN1, activation="relu", input_shape=(INPUT_DIM,), name="hidden1"),
    keras.layers.Dense(HIDDEN2, activation="relu", name="hidden2"),
    keras.layers.Dense(OUTPUT_DIM, activation="sigmoid", name="output"),
])

model.compile(optimizer=keras.optimizers.Adam(0.001), loss="mse", metrics=["mae"])
model.summary()

total_params = model.count_params()
print(f"\nTotal parameters: {total_params}")
print(f"Estimated ESP32 memory: {total_params * 4} bytes ({total_params * 4 / 1024:.1f} KB)")

history = model.fit(
    X_train, Y_train,
    validation_split=0.15,
    epochs=50,
    batch_size=64,
    verbose=1,
)

loss, mae = model.evaluate(X_test, Y_test, verbose=0)
print(f"\nTest Loss (MSE): {loss:.6f}")
print(f"Test MAE (normalized): {mae:.6f}")

Y_pred = model.predict(X_test)
abs_errors = np.abs(Y_pred - Y_test) * feat_range
mean_abs_errors = abs_errors.mean(axis=0)
print(f"\nDenormalized MAE per feature:")
for i, name in enumerate(FEATURES):
    unit = ["C", "%", "Pa", "lux"][i]
    print(f"  {name:>12s}: {mean_abs_errors[i]:.2f} {unit}")

# ─── 7. Feature importance ───────────────────────────────────
print("\n" + "=" * 60)
print("Step 5: Computing feature importance...")
print("=" * 60)

baseline_mse = np.mean((Y_pred - Y_test) ** 2, axis=0)
importance_matrix = np.zeros((4, 4))

for input_feat_idx in range(4):
    cols = [t * 4 + input_feat_idx for t in range(LOOKBACK)]
    X_shuffled = X_test.copy()
    for col in cols:
        np.random.shuffle(X_shuffled[:, col])
    Y_shuffled = model.predict(X_shuffled, verbose=0)
    shuffled_mse = np.mean((Y_shuffled - Y_test) ** 2, axis=0)
    importance_matrix[input_feat_idx] = (shuffled_mse - baseline_mse) / (baseline_mse + 1e-10)

print("\nFeature importance (input -> output):")
print(f"{'':>12s}  {'Pred Temp':>10s}  {'Pred Hum':>10s}  {'Pred Press':>10s}  {'Pred Lux':>10s}")
for i, name in enumerate(FEATURES):
    row = "  ".join(f"{importance_matrix[i, j]:10.4f}" for j in range(4))
    print(f"{name:>12s}  {row}")

imp_max = importance_matrix.max()
importance_normalized = importance_matrix / imp_max if imp_max > 0 else importance_matrix

overall_importance = importance_matrix.mean(axis=1)
overall_total = overall_importance.sum()
overall_pct = (overall_importance / overall_total * 100) if overall_total > 0 else np.zeros(4)

print(f"\nOverall feature importance:")
for i, name in enumerate(FEATURES):
    print(f"  {name:>12s}: {overall_pct[i]:.1f}%")

importance_data = {
    "features": FEATURES,
    "matrix": importance_normalized.tolist(),
    "raw_matrix": importance_matrix.tolist(),
    "overall_pct": overall_pct.tolist(),
    "mae_per_output": mean_abs_errors.tolist(),
    "mae_units": ["C", "%", "Pa", "lux"],
    "training_history": {
        "loss": [float(v) for v in history.history["loss"]],
        "val_loss": [float(v) for v in history.history["val_loss"]],
    }
}
with open("feature_importance.json", "w") as f:
    json.dump(importance_data, f, indent=2)
print("Saved: feature_importance.json")

# ─── 8. Export C header ──────────────────────────────────────
print("\n" + "=" * 60)
print("Step 6: Exporting C header for ESP32...")
print("=" * 60)


def array_to_c(name, arr):
    flat = arr.flatten()
    lines = [f"const float {name}[{len(flat)}] PROGMEM = {{"]
    for i in range(0, len(flat), 8):
        chunk = flat[i : i + 8]
        row = ", ".join(f"{v:.8f}f" for v in chunk)
        lines.append(f"    {row},")
    lines.append("};")
    return "\n".join(lines)


header_lines = [
    "// AUTO-GENERATED - WeatherMind NN Weights",
    "// Model: 48 -> 16 (ReLU) -> 8 (ReLU) -> 4 (Sigmoid)",
    f"// Prediction horizon: ~{PREDICTION_STEP * 5 / 60:.0f} minutes",
    f"// Total parameters: {total_params}",
    "#pragma once",
    "#include <Arduino.h>",
    "",
    f"#define NN_LOOKBACK    {LOOKBACK}",
    f"#define NN_INPUT_DIM   {INPUT_DIM}",
    f"#define NN_HIDDEN1     {HIDDEN1}",
    f"#define NN_HIDDEN2     {HIDDEN2}",
    f"#define NN_OUTPUT_DIM  {OUTPUT_DIM}",
    "",
    "// Normalization parameters (min, range per feature)",
    f"const float FEAT_MIN[4] PROGMEM = {{{', '.join(f'{v:.6f}f' for v in feat_min)}}};",
    f"const float FEAT_RANGE[4] PROGMEM = {{{', '.join(f'{v:.6f}f' for v in feat_range)}}};",
    "",
]

for layer in model.layers:
    w, b = layer.get_weights()
    lname = layer.name.upper()
    header_lines.append(f"// Layer: {layer.name}  shape: {w.shape}")
    header_lines.append(array_to_c(f"W_{lname}", w))
    header_lines.append("")
    header_lines.append(array_to_c(f"B_{lname}", b))
    header_lines.append("")

with open("nn_weights.h", "w") as f:
    f.write("\n".join(header_lines))

print(f"Saved: nn_weights.h ({os.path.getsize('nn_weights.h')} bytes)")
print("\nDone! Copy nn_weights.h into your esp32_sensor_nn/ folder.")