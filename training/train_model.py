# # """
# # WeatherMind — Tiny NN Trainer for ESP32
# # =======================================
# # Trains a small NN on temperature, humidity, pressure, and lux
# # to predict values ~30 minutes ahead. Exports weights as a C
# # header for ESP32 deployment. Computes feature importance.

# # pip install kagglehub pandas numpy scikit-learn tensorflow
# # python train_model.py
# # """

# import numpy as np
# import pandas as pd
# import os
# import json
# import glob

# # ─── 1. Load Dataset ─────────────────────────────────────────
# print("=" * 60)
# print("Step 1: Loading dataset from Kaggle...")
# print("=" * 60)

# import kagglehub

# print("Downloading dataset...")
# dataset_path = kagglehub.dataset_download(
#     "patrickfleith/temperature-humidity-pressure-illuminance"
# )
# print(f"Dataset downloaded to: {dataset_path}")

# df = pd.read_csv(os.path.join(dataset_path, "DATA-large.CSV"))

# print(f"Dataset shape: {df.shape}")
# print(f"Columns: {list(df.columns)}")
# print(df.head())
# print(df["lux"].describe())


# # ─── 2. Preprocess ───────────────────────────────────────────
# print("\n" + "=" * 60)
# print("Step 2: Preprocessing...")
# print("=" * 60)

# col_map = {}
# for col in df.columns:
#     cl = col.lower().strip()
#     if "temp" in cl:
#         col_map["temperature"] = col
#     elif "hum" in cl:
#         col_map["humidity"] = col
#     elif "press" in cl:
#         col_map["pressure"] = col
#     elif "lux" in cl or "illum" in cl or "light" in cl:
#         col_map["lux"] = col
#     elif "time" in cl or "date" in cl:
#         col_map["time"] = col

# print(f"Column mapping: {col_map}")

# FEATURES = ["temperature", "humidity", "pressure", "lux"]
# feature_cols = [col_map[f] for f in FEATURES]

# data = df[feature_cols].values.astype(np.float32)
# mask = ~np.isnan(data).any(axis=1)
# data = data[mask]
# print(f"Clean samples: {data.shape[0]}")

# # ─── 3. Windowing ────────────────────────────────────────────
# LOOKBACK = 12          # 12 readings = 1 minute of context
# PREDICTION_STEP = 360  # 360 steps * 5s = 30 minutes ahead

# print(f"Lookback window: {LOOKBACK} samples (~{LOOKBACK * 5}s)")
# print(f"Prediction horizon: {PREDICTION_STEP} steps (~{PREDICTION_STEP * 5 / 60:.0f} min)")

# # ─── 4. Normalize ────────────────────────────────────────────
# feat_min = data.min(axis=0)
# feat_max = data.max(axis=0)


# feat_range = feat_max - feat_min
# feat_range[feat_range == 0] = 1.0

# data_norm = (data - feat_min) / feat_range
# # feat_range = feat_max - feat_min
# # feat_range[feat_range == 0] = 1.0

# # data_norm = (data - feat_min) / feat_range

# # print(f"Feature mins:   {feat_min}")
# # print(f"Feature maxes:  {feat_max}")

# # ─── 5. Create sequences ─────────────────────────────────────
# print("\n" + "=" * 60)
# print("Step 3: Creating training sequences...")
# print("=" * 60)

# X_list = []
# Y_list = []
# for i in range(len(data_norm) - LOOKBACK - PREDICTION_STEP):
#     window = data_norm[i : i + LOOKBACK]
#     target = data_norm[i + LOOKBACK + PREDICTION_STEP - 1]
#     X_list.append(window.flatten())
#     Y_list.append(target[:3])

# X = np.array(X_list, dtype=np.float32)
# Y = np.array(Y_list, dtype=np.float32)
# print(f"X shape: {X.shape}")
# print(f"Y shape: {Y.shape}")

# from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test = train_test_split(
#     X, Y, test_size=0.15, random_state=42
# )
# print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# # ─── 6. Build & train ────────────────────────────────────────
# print("\n" + "=" * 60)
# print("Step 4: Building & training tiny NN...")
# print("=" * 60)

# import tensorflow as tf
# from tensorflow import keras

# INPUT_DIM = LOOKBACK * 4  # 48
# HIDDEN1 = 16
# HIDDEN2 = 8
# OUTPUT_DIM = 3

# model = keras.Sequential([
#     keras.layers.Dense(HIDDEN1, activation="relu", input_shape=(INPUT_DIM,), name="hidden1"),
#     keras.layers.Dense(HIDDEN2, activation="relu", name="hidden2"),
#     keras.layers.Dense(OUTPUT_DIM, activation="sigmoid", name="output"),
# ])

# model.compile(optimizer=keras.optimizers.Adam(0.001), loss="mse", metrics=["mae"])
# model.summary()

# total_params = model.count_params()
# print(f"\nTotal parameters: {total_params}")
# print(f"Estimated ESP32 memory: {total_params * 4} bytes ({total_params * 4 / 1024:.1f} KB)")

# history = model.fit(
#     X_train, Y_train,
#     validation_split=0.15,
#     epochs=50,
#     batch_size=64,
#     verbose=1,
# )

# loss, mae = model.evaluate(X_test, Y_test, verbose=0)
# print(f"\nTest Loss (MSE): {loss:.6f}")
# print(f"Test MAE (normalized): {mae:.6f}")

# Y_pred = model.predict(X_test)
# abs_errors = np.abs(Y_pred - Y_test) * feat_range[:3]
# mean_abs_errors = abs_errors.mean(axis=0)
# print(f"\nDenormalized MAE per feature:")
# for i, name in enumerate(FEATURES[:3]):
#     unit = ["C", "%", "Pa"][i]
#     print(f"  {name:>12s}: {mean_abs_errors[i]:.2f} {unit}")

# # ─── 7. Feature importance ───────────────────────────────────
# print("\n" + "=" * 60)
# print("Step 5: Computing feature importance...")
# print("=" * 60)

# baseline_mse = np.mean((Y_pred - Y_test) ** 2, axis=0)
# importance_matrix = np.zeros((4, 3))

# for input_feat_idx in range(4):
#     cols = [t * 4 + input_feat_idx for t in range(LOOKBACK)]
#     X_shuffled = X_test.copy()
#     for col in cols:
#         np.random.shuffle(X_shuffled[:, col])
#     Y_shuffled = model.predict(X_shuffled, verbose=0)
#     shuffled_mse = np.mean((Y_shuffled - Y_test) ** 2, axis=0)
#     importance_matrix[input_feat_idx] = (shuffled_mse - baseline_mse) / (baseline_mse + 1e-10)

# print("\nFeature importance (input -> output):")
# print(f"{'':>12s}  {'Pred Temp':>10s}  {'Pred Hum':>10s}  {'Pred Press':>10s}")
# for i, name in enumerate(FEATURES):
#     row = "  ".join(f"{importance_matrix[i, j]:10.4f}" for j in range(3))
#     print(f"{name:>12s}  {row}")

# imp_max = importance_matrix.max()
# importance_normalized = importance_matrix / imp_max if imp_max > 0 else importance_matrix

# overall_importance = importance_matrix.mean(axis=1)
# overall_total = overall_importance.sum()
# overall_pct = (overall_importance / overall_total * 100) if overall_total > 0 else np.zeros(4)

# print(f"\nOverall feature importance:")
# for i, name in enumerate(FEATURES):
#     print(f"  {name:>12s}: {overall_pct[i]:.1f}%")

# importance_data = {
#     "features": FEATURES,
#     "matrix": importance_normalized.tolist(),
#     "raw_matrix": importance_matrix.tolist(),
#     "overall_pct": overall_pct.tolist(),
#     "mae_per_output": mean_abs_errors.tolist(),
#     "mae_units": ["C", "%", "Pa", "lux"],
#     "training_history": {
#         "loss": [float(v) for v in history.history["loss"]],
#         "val_loss": [float(v) for v in history.history["val_loss"]],
#     }
# }
# with open("feature_importance.json", "w") as f:
#     json.dump(importance_data, f, indent=2)
# print("Saved: feature_importance.json")

# # ─── Accuracy metrics ─────────────────────────────────────────
# print("\n" + "=" * 60)
# print("Model Accuracy Summary")
# print("=" * 60)

# # R² score per feature
# from sklearn.metrics import r2_score
# for i, name in enumerate(FEATURES[:3]):
#     r2 = r2_score(Y_test[:, i], Y_pred[:, i])
#     print(f"  {name:>12s}  R²: {r2:.4f}  ({r2*100:.1f}% variance explained)")

# # Overall R²
# overall_r2 = r2_score(Y_test, Y_pred, multioutput='uniform_average')
# print(f"\n  {'Overall':>12s}  R²: {overall_r2:.4f}  ({overall_r2*100:.1f}%)")

# # MAPE (Mean Absolute Percentage Error)
# Y_test_denorm = Y_test * feat_range[:3] + feat_min[:3]
# Y_pred_denorm = Y_pred * feat_range[:3] + feat_min[:3]
# for i, name in enumerate(FEATURES[:3]):
#     mask = Y_test_denorm[:, i] != 0
#     mape = np.mean(np.abs((Y_test_denorm[mask, i] - Y_pred_denorm[mask, i]) / Y_test_denorm[mask, i])) * 100
#     print(f"  {name:>12s}  MAPE: {mape:.2f}%")

# # ─── 8. Export C header ──────────────────────────────────────
# print("\n" + "=" * 60)
# print("Step 6: Exporting C header for ESP32...")
# print("=" * 60)


# def array_to_c(name, arr):
#     flat = arr.flatten()
#     lines = [f"const float {name}[{len(flat)}] PROGMEM = {{"]
#     for i in range(0, len(flat), 8):
#         chunk = flat[i : i + 8]
#         row = ", ".join(f"{v:.8f}f" for v in chunk)
#         lines.append(f"    {row},")
#     lines.append("};")
#     return "\n".join(lines)


# header_lines = [
#     "// AUTO-GENERATED - WeatherMind NN Weights",
#     "// Model: 48 -> 16 (ReLU) -> 8 (ReLU) -> 3 (Sigmoid)",
#     f"// Prediction horizon: ~{PREDICTION_STEP * 5 / 60:.0f} minutes",
#     f"// Total parameters: {total_params}",
#     "#pragma once",
#     "#include <Arduino.h>",
#     "",
#     f"#define NN_LOOKBACK    {LOOKBACK}",
#     f"#define NN_INPUT_DIM   {INPUT_DIM}",
#     f"#define NN_HIDDEN1     {HIDDEN1}",
#     f"#define NN_HIDDEN2     {HIDDEN2}",
#     f"#define NN_OUTPUT_DIM  {OUTPUT_DIM}",
#     "",
#     "// Normalization parameters (min, range per feature)",
#     f"const float FEAT_MIN[4] PROGMEM = {{{', '.join(f'{v:.6f}f' for v in feat_min)}}};",
#     f"const float FEAT_RANGE[4] PROGMEM = {{{', '.join(f'{v:.6f}f' for v in feat_range)}}};",
#     "",
# ]

# for layer in model.layers:
#     w, b = layer.get_weights()
#     lname = layer.name.upper()
#     header_lines.append(f"// Layer: {layer.name}  shape: {w.shape}")
#     header_lines.append(array_to_c(f"W_{lname}", w))
#     header_lines.append("")
#     header_lines.append(array_to_c(f"B_{lname}", b))
#     header_lines.append("")

# with open("nn_weights.h", "w") as f:
#     f.write("\n".join(header_lines))

# print(f"Saved: nn_weights.h ({os.path.getsize('nn_weights.h')} bytes)")
# print("\nDone! Copy nn_weights.h into your esp32_sensor_nn/ folder.")
"""
FarmSentinel — NN Trainer (USCRN + In-Field Fine-Tune)
=======================================================
Dataset strategy:
  Phase 1 — Pre-train on USCRN Hourly02 data (NOAA public FTP, no login needed).
             Columns used: T_CALC (air °C), RH_HR_AVG (%), SOLAR_RADIATION (W/m²),
             SOIL_MOISTURE_5 (m³/m³ @ 5cm), SOIL_TEMP_5 (°C @ 5cm).
             Pressure is synthesized from T_CALC + RH via a physics-derived
             correction (standard atmosphere + humidity correction) — adequate
             for pre-training since the NN sees pressure *trends*, not absolutes.
  Phase 2 — Fine-tune on data/field_samples.csv collected from your actual modules
             over the first 2–3 weeks of deployment. This makes the NN farm-specific.

Targets (6-hour ahead):
  - T_MIN_NIGHT: overnight minimum temperature (frost risk proxy)
  - T_MAX_DAY:   daytime maximum temperature (heat/drought stress proxy)
  Classification is done on-device based on time-of-day (light sensor) + these values.

Input features (6 timesteps × 6 sensors = 36-dim input):
  [air_temp, humidity, pressure, lux, soil_moisture, soil_temp]

Architecture: 36 → 24 (ReLU) → 12 (ReLU) → 2 (Linear)
  Linear output: [pred_min_temp, pred_max_temp]
  Small enough for ESP32 PROGMEM (~2.5 KB weights).

Usage:
  pip install numpy pandas scikit-learn tensorflow requests tqdm
  python train_model.py               # pre-train only
  python train_model.py --finetune    # pre-train + fine-tune on field data
  python train_model.py --finetune --field-only  # fine-tune only (weights already exist)
"""

import os
import sys
import json
import argparse
import urllib.request
import numpy as np
import pandas as pd
from pathlib import Path

# ─── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--finetune", action="store_true", help="Fine-tune on field data after pre-training")
parser.add_argument("--field-only", action="store_true", help="Skip pre-training, fine-tune only")
parser.add_argument("--field-data", default="data/field_samples.csv", help="Path to field CSV")
parser.add_argument("--station", default="AK_Bethel_87_WNW", help="USCRN station name (see NOAA FTP)")
parser.add_argument("--years", default="2020,2021,2022,2023", help="Comma-separated years to download")
args = parser.parse_args()

# ─── Config ───────────────────────────────────────────────────────────────────
LOOKBACK     = 6          # 6 hourly readings = 6h of context
HORIZON      = 6          # predict 6h ahead
NN_FEATURES  = 6          # air_temp, humidity, pressure, lux, soil_moisture, soil_temp
NN_HIDDEN1   = 24
NN_HIDDEN2   = 12
NN_OUTPUT    = 2          # [min_temp_ahead, max_temp_ahead]

SOLAR_TO_LUX = 120.0      # approximate W/m² → lux conversion factor (clear sky ~120 lux/W/m²)
DATA_DIR     = Path("data/uscrn")
DATA_DIR.mkdir(parents=True, exist_ok=True)

USCRN_FTP_BASE = "https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02"

# USCRN Hourly02 column positions (space-delimited, fixed-width)
# Full header: WBANNO UTC_DATE UTC_TIME LST_DATE LST_TIME CRX_VN LONGITUDE LATITUDE
#   T_CALC T_HR_AVG T_MAX T_MIN P_CALC SOLARAD SOLARAD_FLAG SURSURRAD_FLAG RH_HR_AVG RH_HR_AVG_FLAG
#   SOIL_MOISTURE_5 SOIL_MOISTURE_10 SOIL_MOISTURE_20 SOIL_MOISTURE_50 SOIL_MOISTURE_100
#   SOIL_TEMP_5 SOIL_TEMP_10 SOIL_TEMP_20 SOIL_TEMP_50 SOIL_TEMP_100
# USCRN_COLS = {
#     "T_CALC":           2,   # air temp °C (average of 3 probes)
#     "T_MAX":            4,   # hourly max air temp °C
#     "T_MIN":            5,   # hourly min air temp °C
#     "RH_HR_AVG":       12,   # relative humidity %
#     "SOLARAD":         13,   # solar radiation W/m²
#     "SOIL_MOISTURE_5": 18,   # volumetric water content m³/m³ at 5cm
#     "SOIL_TEMP_5":     23,   # soil temperature °C at 5cm
# }
USCRN_COLS = {
    "T_CALC":           8,   # air temp °C
    "T_MAX":            10,  # hourly max air temp °C
    "T_MIN":            11,  # hourly min air temp °C
    "RH_HR_AVG":        26,  # relative humidity %
    "SOLARAD":          13,  # solar radiation W/m²
    "SOIL_MOISTURE_5":  28,  # volumetric water content at 5cm
    "SOIL_TEMP_5":      33,  # soil temperature at 5cm
}

MISSING_FLAG = -9999.0

def synthesize_pressure(temp_c, rh_pct, elevation_m=100.0):
    """
    Approximate surface pressure from temperature + humidity + elevation.
    Barometric formula + partial vapor pressure correction.
    Good enough for NN pre-training; real BMP180 values replace this in field.
    """
    T_K = temp_c + 273.15
    P0 = 101325.0  # Pa at sea level
    L = 0.0065     # temperature lapse rate K/m
    R = 8.314
    M = 0.02897
    g = 9.81
    P_dry = P0 * (1 - L * elevation_m / T_K) ** (g * M / (R * L))
    # Vapor pressure correction (Magnus formula)
    e_sat = 611.2 * np.exp(17.67 * temp_c / (temp_c + 243.5))
    e_act = (rh_pct / 100.0) * e_sat
    P_hpa = (P_dry - 0.378 * e_act) / 100.0  # convert to hPa
    return P_hpa * 100.0  # back to Pa

# ─── Step 1: Download USCRN data ──────────────────────────────────────────────
def download_uscrn(station, years):
    dfs = []
    for year in years:
        fname = f"CRNH0203-{year}-{station}.txt"
        fpath = DATA_DIR / fname
        url = f"{USCRN_FTP_BASE}/{year}/{fname}"

        if not fpath.exists():
            print(f"  Downloading {url} ...")
            try:
                urllib.request.urlretrieve(url, fpath)
                print(f"  Saved → {fpath}")
            except Exception as e:
                print(f"  WARNING: Could not download {fname}: {e}")
                continue
        else:
            print(f"  Using cached {fpath}")

        try:
            df = pd.read_csv(fpath, sep=r'\s+', header=None, engine='python')
            dfs.append(df)
        except Exception as e:
            print(f"  WARNING: Could not parse {fname}: {e}")

    if not dfs:
        raise RuntimeError(
            "No USCRN data downloaded. Check station name and years.\n"
            "Browse stations at: https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/\n"
            "Station list example: AK_Bethel_87_WNW, AL_Clanton_2_NE, AZ_Tucson_11_W, ..."
        )
    return pd.concat(dfs, ignore_index=True)

def parse_uscrn(raw_df):
    """Extract and clean the columns we need from raw USCRN space-delimited data."""
    print(f"  Total columns: {raw_df.shape[1]}")
    print(f"  First row: {raw_df.iloc[0].tolist()}")
    print(f"  Col 2 (T_CALC): {raw_df.iloc[:5, 2].tolist()}")
    print(f"  Col 4 (T_MAX):  {raw_df.iloc[:5, 4].tolist()}")
    print(f"  Col 5 (T_MIN):  {raw_df.iloc[:5, 5].tolist()}")
    data = {}
    for name, col in USCRN_COLS.items():
        series = pd.to_numeric(raw_df.iloc[:, col], errors="coerce")
        series = series.where(series > -100, np.nan)
        data[name] = series.values

    # Synthesize pressure from air temp + RH
    temp = data["T_CALC"]
    rh   = data["RH_HR_AVG"]
    pres = np.array([
        synthesize_pressure(t, r) if (not np.isnan(t) and not np.isnan(r)) else np.nan
        for t, r in zip(temp, rh)
    ])
    data["PRESSURE"] = pres
    # Solar → lux approximation
    sol = data["SOLARAD"]
    data["LUX"] = np.where(np.isnan(sol), np.nan, sol * SOLAR_TO_LUX)

    df = pd.DataFrame({
        "air_temp":      data["T_CALC"],
        "humidity":      data["RH_HR_AVG"],
        "pressure":      data["PRESSURE"],
        "lux":           data["LUX"],
        "soil_moisture": data["SOIL_MOISTURE_5"],
        "soil_temp":     data["SOIL_TEMP_5"],
        "t_max":         data["T_MAX"],
        "t_min":         data["T_MIN"],
    })
    df["t_min"]    = df["t_min"].where(df["t_min"].abs() < 60, np.nan)
    df["t_max"]    = df["t_max"].where(df["t_max"].abs() < 60, np.nan)
    df["air_temp"] = df["air_temp"].where(df["air_temp"].abs() < 60, np.nan)
    df["soil_moisture"] = df["soil_moisture"].where((df["soil_moisture"] >= 0) & (df["soil_moisture"] <= 1), np.nan)
    return df

# ─── Step 2: Build sequences ───────────────────────────────────────────────────
FEATURE_COLS = ["air_temp", "humidity", "pressure", "lux", "soil_moisture", "soil_temp"]

def build_sequences(df, lookback=LOOKBACK, horizon=HORIZON):
    """
    Input:  lookback hours of all 6 sensor features (flattened)
    Target: [min_temp, max_temp] over the NEXT horizon hours
    This teaches the NN to predict the overnight low and daytime high coming up.
    """
    arr = df[FEATURE_COLS].values.astype(np.float32)
    t_max = df["t_max"].values.astype(np.float32)
    t_min = df["t_min"].values.astype(np.float32)

    # Forward-fill missing values (max 3 consecutive gaps)
    for col_i in range(arr.shape[1]):
        mask = np.isnan(arr[:, col_i])
        fill_count = 0
        last_valid = np.nan
        for i in range(len(arr)):
            if not mask[i]:
                last_valid = arr[i, col_i]
                fill_count = 0
            elif fill_count < 3 and not np.isnan(last_valid):
                arr[i, col_i] = last_valid
                fill_count += 1

    X, Y = [], []
    n = len(arr)
    for i in range(n - lookback - horizon):
        window = arr[i: i + lookback]
        future_max = t_max[i + lookback: i + lookback + horizon]
        future_min = t_min[i + lookback: i + lookback + horizon]

        # Skip if any NaN in window or targets
        if np.isnan(window).any():
            continue
        if np.isnan(future_max).any() or np.isnan(future_min).any():
            continue
        future_min_clipped = future_min[np.abs(future_min) < 60]
        future_max_clipped = future_max[np.abs(future_max) < 60]
        if len(future_min_clipped) == 0 or len(future_max_clipped) == 0:
            continue

        X.append(window.flatten())
        Y.append([future_min_clipped.min(), future_max_clipped.max()])  # worst-case min and max ahead

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

# ─── Step 3: Normalize ────────────────────────────────────────────────────────
def compute_stats(X, Y):
    x_min  = X.min(axis=0)
    x_max  = X.max(axis=0)
    x_range = x_max - x_min
    x_range[x_range == 0] = 1.0

    y_min  = Y.min(axis=0)
    y_max  = Y.max(axis=0)
    y_range = y_max - y_min
    y_range[y_range == 0] = 1.0

    return x_min, x_range, y_min, y_range

def normalize(X, Y, x_min, x_range, y_min, y_range):
    return (X - x_min) / x_range, (Y - y_min) / y_range

def denormalize_y(Y_norm, y_min, y_range):
    return Y_norm * y_range + y_min

# ─── Step 4: Build & train ────────────────────────────────────────────────────
def build_model(input_dim):
    import tensorflow as tf
    from tensorflow import keras
    model = keras.Sequential([
        keras.layers.Dense(NN_HIDDEN1, activation="relu",
                           input_shape=(input_dim,), name="hidden1"),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(NN_HIDDEN2, activation="relu", name="hidden2"),
        keras.layers.Dense(NN_OUTPUT, activation="linear", name="output"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss="huber",   # robust to outliers (frost spikes, heat waves)
        metrics=["mae"]
    )
    return model

def train_phase(model, X_train, Y_train, X_val, Y_val, epochs=60, lr=0.001, tag="pre-train"):
    import tensorflow as tf
    print(f"\n  [{tag}] Training {epochs} epochs...")
    model.optimizer.learning_rate.assign(lr)
    cb = [
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss"),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-5),
    ]
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=64,
        callbacks=cb,
        verbose=1,
    )
    return history

# ─── Step 5: Export C header ──────────────────────────────────────────────────
def export_header(model, x_min, x_range, y_min, y_range, feat_min_per_feature, feat_range_per_feature, out_path="nn_weights.h"):
    """
    Exports all weights + normalization stats as a PROGMEM C header for ESP32.
    """

    def arr_to_c(name, arr, dtype="float"):
        flat = arr.flatten()
        lines = [f"const {dtype} {name}[{len(flat)}] PROGMEM = {{"]
        for i in range(0, len(flat), 8):
            chunk = flat[i: i + 8]
            row = ", ".join(f"{v:.8f}f" for v in chunk)
            lines.append(f"    {row},")
        lines.append("};")
        return "\n".join(lines)

    total_params = model.count_params()
    input_dim = LOOKBACK * NN_FEATURES

    lines = [
        "// AUTO-GENERATED — FarmSentinel NN Weights",
        f"// Architecture: {input_dim} → {NN_HIDDEN1} (ReLU) → {NN_HIDDEN2} (ReLU) → {NN_OUTPUT} (Linear)",
        f"// Outputs: [pred_min_temp_C, pred_max_temp_C]  (6h ahead)",
        f"// Total parameters: {total_params}",
        f"// PROGMEM usage: ~{total_params * 4} bytes ({total_params * 4 / 1024:.1f} KB)",
        "#pragma once",
        "#include <Arduino.h>",
        "",
        f"#define NN_LOOKBACK     {LOOKBACK}",
        f"#define NN_FEATURES     {NN_FEATURES}",
        f"#define NN_INPUT_DIM    {input_dim}",
        f"#define NN_HIDDEN1      {NN_HIDDEN1}",
        f"#define NN_HIDDEN2      {NN_HIDDEN2}",
        f"#define NN_OUTPUT_DIM   {NN_OUTPUT}",
        "",
        "// Per-feature normalization: [air_temp, humidity, pressure, lux, soil_moisture, soil_temp]",
        f"const float FEAT_MIN[{NN_FEATURES}] PROGMEM = {{{', '.join(f'{v:.6f}f' for v in feat_min_per_feature)}}};",
        f"const float FEAT_RANGE[{NN_FEATURES}] PROGMEM = {{{', '.join(f'{v:.6f}f' for v in feat_range_per_feature)}}};",
        "",
        "// Output denormalization: [min_temp, max_temp]",
        f"const float OUT_MIN[{NN_OUTPUT}] PROGMEM = {{{', '.join(f'{v:.6f}f' for v in y_min)}}};",
        f"const float OUT_RANGE[{NN_OUTPUT}] PROGMEM = {{{', '.join(f'{v:.6f}f' for v in y_range)}}};",
        "",
    ]

    for layer in model.layers:
        weights = layer.get_weights()
        if not weights:
            continue
        W, b = weights[0], weights[1]
        lname = layer.name.upper()
        lines.append(f"// Layer: {layer.name}  shape: {W.shape}")
        lines.append(arr_to_c(f"W_{lname}", W))
        lines.append("")
        lines.append(arr_to_c(f"B_{lname}", b))
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Saved: {out_path}  ({os.path.getsize(out_path)} bytes)")

# ─── Step 6: Accuracy report ──────────────────────────────────────────────────
def accuracy_report(model, X_test, Y_test_norm, y_min, y_range):
    from sklearn.metrics import r2_score, mean_absolute_error
    Y_pred_norm = model.predict(X_test, verbose=0)
    Y_pred = denormalize_y(Y_pred_norm, y_min, y_range)
    Y_true = denormalize_y(Y_test_norm, y_min, y_range)

    labels = ["Min Temp (°C)", "Max Temp (°C)"]
    print("\n  ┌─────────────────────────────────────────────")
    print("  │ Accuracy Report")
    print("  ├─────────────────────────────────────────────")
    for i, lbl in enumerate(labels):
        mae = mean_absolute_error(Y_true[:, i], Y_pred[:, i])
        r2  = r2_score(Y_true[:, i], Y_pred[:, i])
        print(f"  │  {lbl:20s}  MAE: {mae:.2f}°C   R²: {r2:.4f}")
    print("  └─────────────────────────────────────────────")

    # Save for dashboard
    importance_data = {
        "features": FEATURE_COLS,
        "outputs": ["min_temp_ahead", "max_temp_ahead"],
        "mae": {
            "min_temp": float(mean_absolute_error(Y_true[:, 0], Y_pred[:, 0])),
            "max_temp": float(mean_absolute_error(Y_true[:, 1], Y_pred[:, 1])),
        },
        "r2": {
            "min_temp": float(r2_score(Y_true[:, 0], Y_pred[:, 0])),
            "max_temp": float(r2_score(Y_true[:, 1], Y_pred[:, 1])),
        },
        "frost_threshold_c": 2.0,
        "heat_threshold_c": 35.0,
    }
    with open("model_metrics.json", "w") as f:
        json.dump(importance_data, f, indent=2)
    print("  Saved: model_metrics.json")

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    import tensorflow as tf
    from sklearn.model_selection import train_test_split

    print("=" * 62)
    print("  FarmSentinel NN Trainer")
    print("=" * 62)

    if not args.field_only:
        # ── Phase 1: USCRN pre-training ───────────────────────────────────
        print("\n[1/6] Downloading USCRN data...")
        years = [y.strip() for y in args.years.split(",")]
        raw = download_uscrn(args.station, years)

        print("\n[2/6] Parsing USCRN columns...")
        df = parse_uscrn(raw)
        print(f"  Rows parsed: {len(df)}")
        print(f"  NaN rate per column:")
        for col in FEATURE_COLS + ["t_max", "t_min"]:
            pct = df[col].isna().mean() * 100
            print(f"    {col:20s}: {pct:.1f}%")

        print("\n[3/6] Building sequences...")
        X, Y = build_sequences(df)
        print(f"  X shape: {X.shape}   Y shape: {Y.shape}")

        if len(X) < 200:
            raise RuntimeError(
                "Too few clean sequences. Try adding more years or a different station.\n"
                "Stations with good soil coverage: TX_Austin_33_NW, IA_Des_Moines_17_SSE, IL_Shabbona_8_NNE"
            )

        # Per-feature normalization stats (for on-device denorm of raw sensor readings)
        feat_min_per_feature  = df[FEATURE_COLS].min().values.astype(np.float32)
        feat_range_per_feature = (df[FEATURE_COLS].max() - df[FEATURE_COLS].min()).values.astype(np.float32)
        feat_range_per_feature[feat_range_per_feature == 0] = 1.0

        x_min, x_range, y_min, y_range = compute_stats(X, Y)
        X_norm, Y_norm = normalize(X, Y, x_min, x_range, y_min, y_range)

        X_train, X_test, Y_train, Y_test = train_test_split(X_norm, Y_norm, test_size=0.15, random_state=42)
        X_train, X_val, Y_train, Y_val   = train_test_split(X_train, Y_train, test_size=0.15, random_state=42)

        print("\n[4/6] Building & pre-training model...")
        model = build_model(input_dim=LOOKBACK * NN_FEATURES)
        model.summary()
        print(f"\n  Total parameters: {model.count_params()}")
        print(f"  ESP32 PROGMEM budget: ~{model.count_params() * 4} bytes ({model.count_params() * 4 / 1024:.1f} KB)")

        train_phase(model, X_train, Y_train, X_val, Y_val, epochs=80, lr=0.001, tag="USCRN pre-train")

        print("\n[5/6] Evaluation...")
        accuracy_report(model, X_test, Y_test, y_min, y_range)

        # Save normalization stats for possible fine-tune continuation
        np.save("data/x_min.npy", x_min)
        np.save("data/x_range.npy", x_range)
        np.save("data/y_min.npy", y_min)
        np.save("data/y_range.npy", y_range)
        np.save("data/feat_min.npy", feat_min_per_feature)
        np.save("data/feat_range.npy", feat_range_per_feature)
        model.save("pretrained_model.keras")
        print("  Saved pretrained_model.keras")

    else:
        # Load previously saved model + stats
        model = tf.keras.models.load_model("pretrained_model.keras")
        x_min  = np.load("data/x_min.npy")
        x_range = np.load("data/x_range.npy")
        y_min  = np.load("data/y_min.npy")
        y_range = np.load("data/y_range.npy")
        feat_min_per_feature  = np.load("data/feat_min.npy")
        feat_range_per_feature = np.load("data/feat_range.npy")
        print("  Loaded pretrained_model.keras + normalization stats.")

    # ── Phase 2: Field fine-tuning (optional) ─────────────────────────────
    if args.finetune or args.field_only:
        field_path = Path(args.field_data)
        if not field_path.exists():
            print(f"\n  [fine-tune] WARNING: Field data not found at {field_path}")
            print("  Skipping fine-tune. Collect ~2 weeks of field data first.")
            print("  Expected CSV columns: timestamp,air_temp,humidity,pressure,lux,soil_moisture,soil_temp")
        else:
            print(f"\n[fine-tune] Loading field data: {field_path}")
            field_df = pd.read_csv(field_path)
            # Remap if needed
            field_df = field_df.rename(columns=str.lower)

            # Add synthetic t_min / t_max from rolling windows for targets
            field_df["t_min"] = field_df["air_temp"].rolling(HORIZON, min_periods=1).min()
            field_df["t_max"] = field_df["air_temp"].rolling(HORIZON, min_periods=1).max()

            Xf, Yf = build_sequences(field_df)
            if len(Xf) < 50:
                print(f"  Only {len(Xf)} sequences from field data — skipping fine-tune (need ≥50).")
            else:
                Xf_norm = (Xf - x_min) / x_range
                Yf_norm = (Yf - y_min) / y_range
                Xf_tr, Xf_val, Yf_tr, Yf_val = train_test_split(Xf_norm, Yf_norm, test_size=0.2, random_state=7)

                # Freeze hidden layers, only retrain output + optional hidden2
                model.get_layer("hidden1").trainable = False
                model.compile(optimizer=tf.keras.optimizers.Adam(0.0003), loss="huber", metrics=["mae"])
                train_phase(model, Xf_tr, Yf_tr, Xf_val, Yf_val, epochs=40, lr=0.0003, tag="field fine-tune")

                print("\n  Post-fine-tune evaluation on field data:")
                accuracy_report(model, Xf_val, Yf_val, y_min, y_range)

    print("\n[6/6] Exporting C header...")
    export_header(
        model,
        x_min, x_range,
        y_min, y_range,
        feat_min_per_feature,
        feat_range_per_feature,
        out_path="nn_weights.h",
    )

    print("\n✓ Done. Copy nn_weights.h into firmware/farm_sentinel_node/")
    print("  Then flash both hub and sensor modules.")

if __name__ == "__main__":
    main()