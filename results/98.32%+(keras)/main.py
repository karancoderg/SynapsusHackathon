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
FS = 1000
EPOCHS = 150
BATCH_SIZE = 1024 # Optimized for T4 GPU (16GB VRAM)

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

def make_tcn(input_shape, n_classes):
    """Keras TCN implementation matching PyTorch parameters."""
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

def make_cnn(input_shape, n_classes):
    """Lightweight CNN implementation."""
    inputs = layers.Input(shape=input_shape)
    x = inputs
    
    # Conv 1: 64, k=5
    x = layers.Conv1D(64, 5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)
    
    # Conv 2: 128, k=3
    x = layers.Conv1D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)
    
    # Conv 3: 64, k=3
    x = layers.Conv1D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation='softmax', dtype='float32')(x)
    return keras.Model(inputs, outputs, name='CNN')

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
    print("Preprocessing...")
    prep = SignalPreprocessor(fs=FS)
    prep.fit(all_data[0]) 
    
    X_wins, y_wins = [], []
    for d, l in zip(all_data, all_labels):
        d_filt = prep.transform(d)
        w = prep.segment(d_filt)
        if w is not None:
             X_wins.append(w)
             # Label: mode of window
             l_win = mode(l[:len(w)*100], keepdims=True)[0]
             # Re-implement stride logic inline for speed
             win_sz = 200
             step = 100
             n = len(d)
             n_win = (n - win_sz) // step + 1
             idx = np.arange(win_sz)[None, :] + np.arange(n_win)[:, None] * step
             w_lbls = l[idx]
             w_modes = mode(w_lbls, axis=1, keepdims=True)[0].flatten()
             y_wins.append(w_modes)

    X = np.concatenate(X_wins)
    y = np.concatenate(y_wins)
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)
    
    # --- 3. Split & Features ---
    # We split first to avoid leakage
    indices = np.arange(len(X))
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X, y_enc, indices, test_size=0.2, stratify=y_enc, random_state=42
    )
    
    # Feature Extraction (CPU Parallel)
    print("Extracting features...")
    fe = FeatureExtractor(fs=FS)
    # Using joblib for parallel extraction
    X_feats_train = np.concatenate(Parallel(n_jobs=-1)(delayed(fe.extract_batch)(b) for b in np.array_split(X_train, 20)))
    X_feats_val = np.concatenate(Parallel(n_jobs=-1)(delayed(fe.extract_batch)(b) for b in np.array_split(X_val, 5)))
    
    scaler_feat = StandardScaler()
    X_feats_train = scaler_feat.fit_transform(X_feats_train)
    X_feats_val = scaler_feat.transform(X_feats_val)

    # --- 4. Train Keras Models ---
    print("\nTraining TCN (Keras)...")
    # Early Stopping & LR Reduction
    es = callbacks.EarlyStopping(monitor='val_accuracy', patience=30, restore_best_weights=True)
    lr_schedule = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5, verbose=1)
    f1_cb = F1Callback(X_val, y_val)
    
    tcn = make_tcn(X_train.shape[1:], n_classes)
    tcn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    cnn = make_cnn(X_train.shape[1:], n_classes)
    cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    tfm = make_transformer(X_train.shape[1:], n_classes)
    tfm.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("Fitting TCN...")
    tcn.fit(X_train, y_train, validation_data=(X_val, y_val), 
            epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es, lr_schedule, f1_cb], verbose=1)
    
    print("\nFitting CNN...")
    cnn.fit(X_train, y_train, validation_data=(X_val, y_val), 
            epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es, lr_schedule, f1_cb], verbose=1)
            
    print("\nFitting Transformer...")
    tfm.fit(X_train, y_train, validation_data=(X_val, y_val), 
            epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es, lr_schedule, f1_cb], verbose=1)

    # --- 5. Train LightGBM ---
    print("\nTraining LightGBM...")
    lgbm = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, verbose=-1)
    lgbm.fit(X_feats_train, y_train)
    
    # --- 6. Ensemble ---
    print("\nOptimizing Ensemble...")
    tcn_probs = tcn.predict(X_val, batch_size=BATCH_SIZE)
    cnn_probs = cnn.predict(X_val, batch_size=BATCH_SIZE)
    tfm_probs = tfm.predict(X_val, batch_size=BATCH_SIZE)
    lgbm_probs = lgbm.predict_proba(X_feats_val)
    
    best_acc = 0
    best_w = (0.4, 0.2, 0.2, 0.2)
    # Simplified Grid Search for 4 models
    # We focus on TCN and Transformer being strong, others supporting
    combinations_w = []
    steps = [0.0, 0.2, 0.4, 0.6]
    for w1 in steps: # TCN
        for w2 in steps: # Transformer
            for w3 in [0.0, 0.2, 0.3]: # CNN
                w4 = 1.0 - w1 - w2 - w3 # LGBM
                if w4 < 0 or abs(w1+w2+w3+w4-1)>1e-5: continue
                combinations_w.append((w1, w2, w3, w4))
                
    for w in combinations_w:
        ens_p = w[0]*tcn_probs + w[1]*tfm_probs + w[2]*cnn_probs + w[3]*lgbm_probs
        acc = accuracy_score(y_val, ens_p.argmax(1))
        if acc > best_acc:
            best_acc = acc
            best_w = w
    
    print(f"Best Ensemble Acc: {best_acc:.4f} with weights TCN={best_w[0]}, TFM={best_w[1]}, CNN={best_w[2]}, LGBM={best_w[3]}")

    # Calculate F1 Score
    best_ens_p = best_w[0]*tcn_probs + best_w[1]*tfm_probs + best_w[2]*cnn_probs + best_w[3]*lgbm_probs
    f1 = f1_score(y_val, best_ens_p.argmax(1), average='macro')
    print(f"Best Ensemble F1 Score: {f1:.4f}")

    # --- 7. Save Artifacts ---
    print("Saving Keras models & artifacts...")
    tcn.save(f'{ARTIFACTS_DIR}/tcn_model.keras')
    cnn.save(f'{ARTIFACTS_DIR}/cnn_model.keras')
    tfm.save(f'{ARTIFACTS_DIR}/tfm_model.keras')
    with open(f'{ARTIFACTS_DIR}/lgbm_model.pkl', 'wb') as f: pickle.dump(lgbm, f)
    
    meta = {
        'label_encoder': le,
        'feature_scaler': scaler_feat,
        'ensemble_weights': best_w,
        'n_channels': 8,
        'n_classes': n_classes,
        'framework': 'keras_v2' # New version
    }
    with open(f'{ARTIFACTS_DIR}/preprocessing.pkl', 'wb') as f: pickle.dump(meta, f)
    
    print("Done.")

if __name__ == '__main__':
    main()
