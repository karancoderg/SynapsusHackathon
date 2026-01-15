#!/usr/bin/env python3
"""
Complete Training Pipeline for Synapse sEMG Challenge (TensorFlow/Keras Version)
Replicates the functionality of the PyTorch pipeline with reduced code complexity.

Optimized for NVIDIA H200 GPU & High Accuracy:
- Batch Size: 4096 (Utilization of high VRAM)
- Mixed Precision (FP16): Enabled via Keras Policy
- Parallel Feature Extraction: CPU offloading
- Unified Training Loop: Keras model.fit()
- Architecture: Matches the 97% Accuracy PyTorch TCN/CNN
"""

# ⚠️ CRITICAL: Must be set BEFORE TensorFlow import to prevent JAX from locking TPU
import os
os.environ['JAX_PLATFORMS'] = 'cpu'  # Force JAX to use CPU, freeing TPU for TensorFlow

import glob
import re
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score
import lightgbm as lgb
from scipy.stats import mode, entropy
from scipy import signal as scipy_signal
from scipy.signal import butter, filtfilt, iirnotch
from scipy.ndimage import uniform_filter1d
import pywt
import warnings
from typing import List, Tuple, Optional, Dict
from itertools import combinations
import random
import sys
from joblib import Parallel, delayed

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Configuration ---
DATA_DIR = 'data'
ARTIFACTS_DIR = 'artifacts'
FS = 512
EPOCHFS = 512
BATCH_SIZE = 1024 # Optimized for T4 (16GB VRAM)
EPOCHS = 100
RANDOM_SEED = 42
# Window size adjustment to maintain ~200 samples context
# 200 samples / 512Hz ≈ 390ms -> Round to 400ms
WINDOW_MS = 400

# T4 GPU Optimization
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Found {len(gpus)} GPU(s). Applying T4 Optimizations...")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # Optimization Flags
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
        os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'

        # Mixed Precision (FP16) - Significantly faster on T4
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print("✅ Mixed Precision (FP16) Enabled")
    else:
        print("No GPU found. Using specific CPU optimizations.")

except Exception as e:
    print(f"GPU Init Warning: {e}")

strategy = tf.distribute.get_strategy()
print(f"Strategy: {strategy}, Batch Size: {BATCH_SIZE}")

# ==================== PREPROCESSING (Numpy Optimized) ====================

class SignalPreprocessor:
    """Preprocessing pipeline for 8-channel sEMG signals."""
    def __init__(self, fs=1000, bandpass_low=20.0, bandpass_high=450.0, notch_freq=50.0):
        self.fs = fs
        nyq = fs / 2
        low = max(0.001, min(bandpass_low / nyq, 0.99))
        high = max(low + 0.01, min(bandpass_high / nyq, 0.999))
        self.b_bp, self.a_bp = butter(4, [low, high], btype='band')
        self.b_notch, self.a_notch = iirnotch(notch_freq, 30.0, self.fs) if notch_freq > 0 else (None, None)
        self.channel_means, self.channel_stds = None, None
        self.fitted = False

    def fit(self, signals):
        if signals.ndim == 3: signals = signals.reshape(-1, signals.shape[-1])
        self.channel_means = np.mean(signals, axis=0)
        self.channel_stds = np.std(signals, axis=0) + 1e-8
        self.fitted = True
        return self

    def transform(self, signal):
        # Filter
        if len(signal) > 12:
            signal = filtfilt(self.b_bp, self.a_bp, signal, axis=0)
            if self.b_notch is not None:
                signal = filtfilt(self.b_notch, self.a_notch, signal, axis=0)
        # Normalize
        if self.fitted:
             return (signal - self.channel_means) / self.channel_stds
        return (signal - np.mean(signal, axis=0)) / (np.std(signal, axis=0) + 1e-8)

    def segment(self, signal, window_ms=200, overlap=0.5):
        win_sz = int(window_ms * self.fs / 1000)
        step = int(win_sz * (1 - overlap))
        n = len(signal)
        if n < win_sz: return None
        n_win = (n - win_sz) // step + 1
        idx = np.arange(win_sz)[None, :] + np.arange(n_win)[:, None] * step
        return signal[idx]

# ==================== FEATURE EXTRACTION ====================

class FeatureExtractor:
    """Extracts stats, freq, wavelet, hjorth features."""
    def __init__(self, fs=1000):
        self.fs = fs

    def extract(self, x):
        f = []
        # Time & Freq & Wavelet & Hjorth per channel
        for i in range(x.shape[1]):
            s = x[:, i]
            # Simple Time
            f.extend([np.mean(np.abs(s)), np.std(s), np.sum(np.abs(np.diff(s))),
                      np.var(s), np.sum(np.abs(s))])

            # Simple Freq
            freqs, psd = scipy_signal.welch(s, self.fs, nperseg=min(256, len(s)))
            f.extend([np.mean(psd), np.max(psd), np.sum(psd)])

            # Wavelet energy
            coeffs = pywt.wavedec(s, 'db4', level=4)
            f.extend([np.sum(c**2) for c in coeffs[:4]])

            # Hjorth (simplified)
            m0 = np.var(s)
            m2 = np.var(np.diff(s))
            m4 = np.var(np.diff(np.diff(s)))
            f.extend([m0, np.sqrt(m2/m0) if m0>0 else 0, np.sqrt(m4/m2)/np.sqrt(m2/m0) if m2>0 and m0>0 else 0])

        return np.array(f)

    def extract_batch(self, windows):
        return np.array([self.extract(w) for w in windows])

# ==================== CALLBACKS ====================

class F1Callback(keras.callbacks.Callback):
    """Calculates Macro F1 Score at the end of each epoch."""
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):
        val_probs = self.model.predict(self.X_val, verbose=0)
        val_preds = val_probs.argmax(axis=1)
        val_f1 = f1_score(self.y_val, val_preds, average='macro')
        logs['val_f1'] = val_f1
        print(f" - val_f1: {val_f1:.4f}")

# ==================== KERAS MODELS ====================

class SEBlock(layers.Layer):
    """Squeeze-and-Excitation Block for 1D signals."""
    def __init__(self, channels, ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.ratio = ratio
        self.avg_pool = layers.GlobalAveragePooling1D()
        self.dense1 = layers.Dense(channels // ratio, activation='relu')
        self.dense2 = layers.Dense(channels, activation='sigmoid')

    def call(self, inputs):
        x = self.avg_pool(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return inputs * x[:, tf.newaxis, :]

    def get_config(self):
        config = super().get_config()
        config.update({'channels': self.channels, 'ratio': self.ratio})
        return config

def make_tcn(input_shape, n_classes):
    """Standard Keras TCN implementation."""
    inputs = layers.Input(shape=input_shape)
    x = inputs

    # Matching [64, 64, 128, 128] filters, kernel 5, dilation [1,2,4,8]
    for filters, d in zip([64, 64, 128, 128], [1, 2, 4, 8]):
        # Residual
        res = layers.Conv1D(filters, 1, padding='same')(x) if x.shape[-1] != filters else x

        # Block
        x = layers.SeparableConv1D(filters, 5, padding='same', dilation_rate=d)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.2)(x)

        x = layers.SeparableConv1D(filters, 5, padding='same', dilation_rate=d)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Add()([x, res])
        x = layers.Activation('relu')(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(n_classes, activation='softmax', dtype='float32')(x)
    return keras.Model(inputs, outputs, name='TCN')

def make_tcn_attention(input_shape, n_classes):
    """TCN with Temporal Attention (Replacing SE-TCN)."""
    inputs = layers.Input(shape=input_shape)
    x = inputs

    # Filters: [64, 128, 128]
    for filters, d in zip([64, 128, 128], [1, 2, 4]):
        # Residual
        res = layers.Conv1D(filters, 1, padding='same')(x) if x.shape[-1] != filters else x

        # TCN Block
        x = layers.SeparableConv1D(filters, 3, padding='same', dilation_rate=d)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.2)(x)

        x = layers.SeparableConv1D(filters, 3, padding='same', dilation_rate=d)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Add()([x, res])
        x = layers.Activation('relu')(x)

    # --- Attention Mechanism ---
    # Apply self-attention to temporal sequence before pooling
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(n_classes, activation='softmax', dtype='float32')(x)
    return keras.Model(inputs, outputs, name='TCN_Attention')

def make_transformer(input_shape, n_classes):
    """Transformer for capturing global temporal dependencies."""
    inputs = layers.Input(shape=input_shape)
    x = inputs

    # Projection
    x = layers.Dense(128)(x)

    # Transformer Blocks
    for _ in range(4):
        # Attention
        x1 = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
        x1 = layers.Dropout(0.1)(x1)
        x = layers.Add()([x, x1])
        x = layers.LayerNormalization(epsilon=1e-6)(x)

        # Feed Forward
        x2 = layers.Conv1D(filters=128, kernel_size=1, activation='relu')(x)
        x2 = layers.Dropout(0.1)(x2)
        x2 = layers.Conv1D(filters=128, kernel_size=1)(x2)
        x = layers.Add()([x, x2])
        x = layers.LayerNormalization(epsilon=1e-6)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(n_classes, activation='softmax', dtype='float32')(x)
    return keras.Model(inputs, outputs, name='Transformer')

# ==================== MAIN ====================

def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # --- 1. Data Loading ---
    # Smart Check: Only download if data is missing
    existing_csvs = glob.glob(f'{DATA_DIR}/**/*.csv', recursive=True)
    if len(existing_csvs) > 0:
        print(f"Dataset found locally ({len(existing_csvs)} files). Skipping download.")
    else:
        if not os.path.exists(DATA_DIR):
            print("Downloading dataset...")
            import gdown, zipfile
            gdown.download('https://drive.google.com/uc?id=16iNEwhThf2LcX7rOOVM03MTZiwq7G51x', 'dataset.zip', quiet=False)
            with zipfile.ZipFile('dataset.zip', 'r') as z: z.extractall(DATA_DIR)
            os.remove('dataset.zip')

    print("Loading data...")
    csvs = sorted(glob.glob(f'{DATA_DIR}/**/*.csv', recursive=True))
    all_data, all_labels = [], []
    for f in csvs:
        try:
            lbl = int(re.search(r'gesture(\d+)', f).group(1))
            d = pd.read_csv(f).values
            if d.shape[1] >= 8:
                all_data.append(d)
                all_labels.append(np.full(len(d), lbl))
        except: pass

    if not all_data: return

    # --- 2. Preprocessing ---
    print(f"Preprocessing (FS={FS}, Window={WINDOW_MS}ms)...")
    prep = SignalPreprocessor(fs=FS)
    prep.fit(all_data[0])

    X_wins, y_wins = [], []
    for d, l in zip(all_data, all_labels):
        d_filt = prep.transform(d)
        # Use dynamic WINDOW_MS to match effective size (~200 samples)
        w = prep.segment(d_filt, window_ms=WINDOW_MS, overlap=0.5)
        if w is not None:
             X_wins.append(w)
             # Label: mode of window
             # Re-implement stride logic inline for speed (must match SignalPreprocessor.segment)
             # SignalPreprocessor: win_sz = int(window_ms * fs / 1000), step = int(win_sz * (1 - overlap))
             window_ms = WINDOW_MS
             overlap = 0.5
             win_sz = int(window_ms * FS / 1000)
             step = int(win_sz * (1 - overlap))

             n = len(d)
             n_win = (n - win_sz) // step + 1
             idx = np.arange(win_sz)[None, :] + np.arange(n_win)[:, None] * step
             w_lbls = l[idx]
             try:
                w_modes = mode(w_lbls, axis=1, keepdims=True)[0].flatten()
                y_wins.append(w_modes)
             except Exception as e:
                if len(w) != len(w_modes):
                     print(f"Warning: size mismatch in file {f}: X={len(w)}, y={len(w_modes)}")
                     X_wins.pop()

    X = np.concatenate(X_wins)
    y = np.concatenate(y_wins)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)

    # --- 3. Split & Features (70% Train, 15% Val, 15% Test) ---
    print("Splitting data (70% Train, 15% Val, 15% Test)...")
    indices = np.arange(len(X))

    # 1. Split Train (70%) vs Temp (30%)
    X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        X, y_enc, indices, test_size=0.3, stratify=y_enc, random_state=42
    )

    # 2. Split Temp (30%) into Val (15%) and Test (15%)
    # 0.5 of 0.3 = 0.15 of total
    X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
        X_temp, y_temp, idx_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    # Feature Extraction (CPU Parallel)
    print("Extracting features...")
    fe = FeatureExtractor(fs=FS)
    # Using joblib for parallel extraction
    X_feats_train = np.concatenate(Parallel(n_jobs=-1)(delayed(fe.extract_batch)(b) for b in np.array_split(X_train, 20)))
    X_feats_val = np.concatenate(Parallel(n_jobs=-1)(delayed(fe.extract_batch)(b) for b in np.array_split(X_val, 5)))
    X_feats_test = np.concatenate(Parallel(n_jobs=-1)(delayed(fe.extract_batch)(b) for b in np.array_split(X_test, 5)))

    scaler_feat = StandardScaler()
    X_feats_train = scaler_feat.fit_transform(X_feats_train)
    X_feats_val = scaler_feat.transform(X_feats_val)
    X_feats_test = scaler_feat.transform(X_feats_test)

    # --- 4. Train Keras Models ---
    print("\nTraining TCN (Keras)...")
    # Early Stopping & LR Reduction
    es = callbacks.EarlyStopping(monitor='val_accuracy', patience=30, restore_best_weights=True)
    lr_schedule = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5, verbose=1)
    f1_cb = F1Callback(X_val, y_val)

    tcn = make_tcn(X_train.shape[1:], n_classes)
    tcn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    tcn_attn = make_tcn_attention(X_train.shape[1:], n_classes)
    tcn_attn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    tfm = make_transformer(X_train.shape[1:], n_classes)
    tfm.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("Fitting TCN...")
    tcn.fit(X_train, y_train, validation_data=(X_val, y_val),
            epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es, lr_schedule, f1_cb], verbose=1)

    print("\nFitting TCN + Attention (Replacing CNN/SE-TCN)...")
    tcn_attn.fit(X_train, y_train, validation_data=(X_val, y_val),
            epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es, lr_schedule, f1_cb], verbose=1)

    print("\nFitting Transformer...")
    tfm.fit(X_train, y_train, validation_data=(X_val, y_val),
            epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es, lr_schedule, f1_cb], verbose=1)

    # --- 5. Train LightGBM ---
    print("\nTraining LightGBM...")
    lgbm = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, verbose=-1)
    lgbm.fit(X_feats_train, y_train)

    # --- 6. Ensemble ---
    print("\nOptimizing Ensemble (TCN + TCN_Attn + Transformer + LGBM)...")
    tcn_probs = tcn.predict(X_val, batch_size=BATCH_SIZE)
    attn_probs = tcn_attn.predict(X_val, batch_size=BATCH_SIZE)
    tfm_probs = tfm.predict(X_val, batch_size=BATCH_SIZE)
    lgbm_probs = lgbm.predict_proba(X_feats_val)

    best_acc = 0
    best_w = (0.4, 0.2, 0.2, 0.2)
    # Search
    combinations_w = []
    steps = [0.0, 0.2, 0.4, 0.6]
    for w1 in steps: # TCN
        for w2 in steps: # Transformer
            for w3 in [0.0, 0.2, 0.4]: # TCN+Attention
                w4 = 1.0 - w1 - w2 - w3 # LGBM
                if w4 < 0 or abs(w1+w2+w3+w4-1)>1e-5: continue
                combinations_w.append((w1, w2, w3, w4))

    for w in combinations_w:
        ens_p = w[0]*tcn_probs + w[1]*tfm_probs + w[2]*attn_probs + w[3]*lgbm_probs
        acc = accuracy_score(y_val, ens_p.argmax(1))
        if acc > best_acc:
            best_acc = acc
            best_w = w

    print(f"Best Ensemble Acc: {best_acc:.4f} with weights TCN={best_w[0]}, TFM={best_w[1]}, TCN_Attn={best_w[2]}, LGBM={best_w[3]}")

    # Calculate F1 Score
    best_ens_p = best_w[0]*tcn_probs + best_w[1]*tfm_probs + best_w[2]*attn_probs + best_w[3]*lgbm_probs
    f1 = f1_score(y_val, best_ens_p.argmax(1), average='macro')
    print(f"Best Ensemble F1 Score: {f1:.4f}")

    # --- 7. Final Test Evaluation (Hold-out Set) ---
    print("\n--- Final Evaluation on Test Set (15%) ---")
    tcn_test = tcn.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    attn_test = tcn_attn.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    tfm_test = tfm.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    lgbm_test = lgbm.predict_proba(X_feats_test)

    ens_test = best_w[0]*tcn_test + best_w[1]*tfm_test + best_w[2]*attn_test + best_w[3]*lgbm_test
    test_acc = accuracy_score(y_test, ens_test.argmax(1))
    test_f1 = f1_score(y_test, ens_test.argmax(1), average='macro')

    print(f"TEST SET ACCURACY: {test_acc:.4f}")
    print(f"TEST SET F1 SCORE: {test_f1:.4f}")
    print("------------------------------------------")

    # --- 7. Save Artifacts ---
    print("Saving Keras models & artifacts...")
    tcn.save(f'{ARTIFACTS_DIR}/tcn_model.keras')
    tcn_attn.save(f'{ARTIFACTS_DIR}/tcn_attn_model.keras')
    tfm.save(f'{ARTIFACTS_DIR}/tfm_model.keras')
    with open(f'{ARTIFACTS_DIR}/lgbm_model.pkl', 'wb') as f: pickle.dump(lgbm, f)

    meta = {
        'label_encoder': le,
        'feature_scaler': scaler_feat,
        'ensemble_weights': best_w,
        'n_channels': 8,
        'n_classes': n_classes,
        'framework': 'keras_v2'
    }
    with open(f'{ARTIFACTS_DIR}/preprocessing.pkl', 'wb') as f: pickle.dump(meta, f)

    print("Done.")

if __name__ == '__main__':
    main()
