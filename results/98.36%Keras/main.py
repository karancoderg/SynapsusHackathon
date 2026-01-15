#!/usr/bin/env python3
"""
Complete Training Pipeline for Synapse sEMG Challenge (TensorFlow/Keras Version)

IMPROVED PIPELINE - TARGET > 97.99% ACCURACY
--------------------------------------------
1. Ensemble of 6 Models:
   - TCN (Standard)
   - TCN + Attention (Temporal)
   - Transformer (Global Dependencies)
   - Conformer (CNN + Transformer)
   - TCN + BiLSTM (Recurrent)
   - LightGBM (Statistical Features)

2. Data Augmentation:
   - Gaussian Noise
   - Scaling
   - Time-Warping (Optional, currently disabled for speed)

3. Optimization (T4 / Colab):
   - Batch Size: 1024
   - Mixed Precision (FP16)
   - CPU Offloading for Feature Extraction
"""

# ⚠️ CRITICAL: Must be set BEFORE TensorFlow import
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
from sklearn.metrics import accuracy_score, f1_score, classification_report
import lightgbm as lgb
from scipy import signal as scipy_signal
from scipy.signal import butter, filtfilt, iirnotch
from scipy.stats import mode
import pywt
import warnings
from joblib import Parallel, delayed

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_DIR = 'data'
ARTIFACTS_DIR = 'artifacts'
FS = 512
EPOCHFS = 512
BATCH_SIZE = 1024  # Optimized for T4 (16GB VRAM)
EPOCHS = 120       # Increased for convergence with augmentation
RANDOM_SEED = 42
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

        # Mixed Precision (FP16)
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print("✅ Mixed Precision (FP16) Enabled")
    else:
        print("No GPU found. Using CPU.")

except Exception as e:
    print(f"GPU Init Warning: {e}")

# ==================== AUGMENTATION ====================

class DataAugmenter:
    """Applies augmentation to raw signals."""
    def __init__(self, sigma=0.01, scale_range=(0.9, 1.1)):
        self.sigma = sigma
        self.scale_range = scale_range

    def augment(self, X):
        """Applies random augmentation to a batch of windowed signals."""
        X_aug = X.copy()
        for i in range(len(X_aug)):
            # 1. Gaussian Noise
            if np.random.random() < 0.5:
                noise = np.random.normal(0, self.sigma, X_aug[i].shape)
                X_aug[i] += noise

            # 2. Random Scaling
            if np.random.random() < 0.5:
                scale = np.random.uniform(*self.scale_range)
                X_aug[i] *= scale

        return X_aug

# ==================== PREPROCESSING ====================

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

            # Hjorth
            m0 = np.var(s)
            m2 = np.var(np.diff(s))
            m4 = np.var(np.diff(np.diff(s)))
            f.extend([m0, np.sqrt(m2/m0) if m0>0 else 0, np.sqrt(m4/m2)/np.sqrt(m2/m0) if m2>0 and m0>0 else 0])

        return np.array(f)

    def extract_batch(self, windows):
        return np.array([self.extract(w) for w in windows])

# ==================== MODELS ====================

def make_tcn(input_shape, n_classes):
    """(1) Standard TCN"""
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for filters, d in zip([64, 64, 128, 128], [1, 2, 4, 8]):
        res = layers.Conv1D(filters, 1, padding='same')(x) if x.shape[-1] != filters else x
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
    """(2) TCN + Temporal Attention"""
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for filters, d in zip([64, 128, 128], [1, 2, 4]):
        res = layers.Conv1D(filters, 1, padding='same')(x) if x.shape[-1] != filters else x
        x = layers.SeparableConv1D(filters, 3, padding='same', dilation_rate=d)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.SeparableConv1D(filters, 3, padding='same', dilation_rate=d)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, res])
        x = layers.Activation('relu')(x)
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(n_classes, activation='softmax', dtype='float32')(x)
    return keras.Model(inputs, outputs, name='TCN_Attention')

def make_transformer(input_shape, n_classes):
    """(3) Transformer"""
    inputs = layers.Input(shape=input_shape)
    x = inputs
    x = layers.Dense(128)(x)
    for _ in range(4):
        x1 = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
        x1 = layers.Dropout(0.1)(x1)
        x = layers.Add()([x, x1])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
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

def make_conformer(input_shape, n_classes):
    """(4) Conformer (CNN + Transformer)"""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    for _ in range(2):
        res = x
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
        x = layers.Dropout(0.1)(x)
        x = layers.Add()([res, x])
        res = x
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Conv1D(x.shape[-1], 3, padding='same', activation='swish')(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Add()([res, x])
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(n_classes, activation='softmax', dtype='float32')(x)
    return keras.Model(inputs, outputs, name='Conformer')

def make_tcn_bilstm(input_shape, n_classes):
    """(5) TCN + BiLSTM"""
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for filters in [64, 64]:
        x = layers.SeparableConv1D(filters, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation='softmax', dtype='float32')(x)
    return keras.Model(inputs, outputs, name='TCN_BiLSTM')

class F1Callback(keras.callbacks.Callback):
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
    def on_epoch_end(self, epoch, logs=None):
        val_probs = self.model.predict(self.X_val, verbose=0, batch_size=BATCH_SIZE)
        val_preds = val_probs.argmax(axis=1)
        val_f1 = f1_score(self.y_val, val_preds, average='macro')
        logs['val_f1'] = val_f1
        print(f" - val_f1: {val_f1:.4f}")

# ==================== MAIN ====================

def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # --- 1. Data Loading ---
    existing_csvs = glob.glob(f'{DATA_DIR}/**/*.csv', recursive=True)
    if not existing_csvs:
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
        w = prep.segment(d_filt, window_ms=WINDOW_MS, overlap=0.5)
        if w is not None:
             X_wins.append(w)
             # Label logic
             win_sz = int(WINDOW_MS * FS / 1000)
             step = int(win_sz * 0.5)
             n = len(d)
             n_win = (n - win_sz) // step + 1
             idx = np.arange(win_sz)[None, :] + np.arange(n_win)[:, None] * step
             w_lbls = l[idx]
             try:
                w_modes = mode(w_lbls, axis=1, keepdims=True)[0].flatten()
                y_wins.append(w_modes)
             except:
                if len(w) != len(w_modes): X_wins.pop()

    X = np.concatenate(X_wins)
    y = np.concatenate(y_wins)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)

    # --- 3. Split ---
    print("Splitting data (70% Train, 15% Val, 15% Test)...")
    indices = np.arange(len(X))
    X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        X, y_enc, indices, test_size=0.3, stratify=y_enc, random_state=42
    )
    X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
        X_temp, y_temp, idx_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    # --- 4. Feature Extraction (For LightGBM) ---
    print("Extracting features (CPU Parallel)...")
    fe = FeatureExtractor(fs=FS)
    # 20 splits for parallel processing
    X_feats_train = np.concatenate(Parallel(n_jobs=-1)(delayed(fe.extract_batch)(b) for b in np.array_split(X_train, 20)))
    X_feats_val = np.concatenate(Parallel(n_jobs=-1)(delayed(fe.extract_batch)(b) for b in np.array_split(X_val, 5)))
    X_feats_test = np.concatenate(Parallel(n_jobs=-1)(delayed(fe.extract_batch)(b) for b in np.array_split(X_test, 5)))

    scaler_feat = StandardScaler()
    X_feats_train = scaler_feat.fit_transform(X_feats_train)
    X_feats_val = scaler_feat.transform(X_feats_val)
    X_feats_test = scaler_feat.transform(X_feats_test)

    # --- 5. Data Augmentation ---
    print("Augmenting Training Data...")
    augmenter = DataAugmenter()
    # Augment copy of X_train
    X_train_aug = augmenter.augment(X_train.copy())

    # Concatenate Original + Augmented
    X_train_final = np.concatenate([X_train, X_train_aug], axis=0)
    y_train_final = np.concatenate([y_train, y_train], axis=0)

    # Shuffle
    perm = np.random.permutation(len(X_train_final))
    X_train_final = X_train_final[perm]
    y_train_final = y_train_final[perm]

    print(f"Train Size after Augmentation: {X_train_final.shape}")

    # --- 6. Train Models ---
    input_sh = X_train.shape[1:]

    models = {
        'TCN': make_tcn(input_sh, n_classes),
        'TCN_Attn': make_tcn_attention(input_sh, n_classes),
        'Transformer': make_transformer(input_sh, n_classes),
        'Conformer': make_conformer(input_sh, n_classes),
        'BiLSTM': make_tcn_bilstm(input_sh, n_classes)
    }

    es = callbacks.EarlyStopping(monitor='val_accuracy', patience=25, restore_best_weights=True)
    lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, verbose=1)
    f1_cb = F1Callback(X_val, y_val)

    print("\nTraining Neural Networks...")
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train_final, y_train_final,
                  validation_data=(X_val, y_val),
                  epochs=EPOCHS,
                  batch_size=BATCH_SIZE,
                  callbacks=[es, lr, f1_cb],
                  verbose=1)
        model.save(f'{ARTIFACTS_DIR}/model_{name}.keras')

    print("\n--- Training LightGBM ---")
    lgbm = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, verbose=-1)
    lgbm.fit(X_feats_train, y_train) # Train on original feats (augmentation less useful for stats)
    with open(f'{ARTIFACTS_DIR}/model_lgbm.pkl', 'wb') as f: pickle.dump(lgbm, f)

    # --- 7. Ensemble Opt ---
    print("\nOptimizing Ensemble Weights (Greedy Search)...")

    # Predictions on VAL
    preds_val = {}
    for name, model in models.items():
        preds_val[name] = model.predict(X_val, batch_size=BATCH_SIZE)
    preds_val['LGBM'] = lgbm.predict_proba(X_feats_val)

    # Weighted Average Initial
    # High weight to TCN_Attn and Transformer as they are usually strongest
    weights = {
        'TCN': 0.15, 'TCN_Attn': 0.2, 'Transformer': 0.2,
        'Conformer': 0.15, 'BiLSTM': 0.1, 'LGBM': 0.2
    }

    def get_acc(ws):
        ens = np.zeros_like(list(preds_val.values())[0])
        for n, w in ws.items():
            ens += w * preds_val[n]
        return accuracy_score(y_val, ens.argmax(1))

    print(f"Base Ensemble Acc: {get_acc(weights):.4f}")

    # Simple Random Search for Weights
    best_acc = 0
    best_weights = weights.copy()

    for _ in range(200):
        # Generate random weights summing to 1
        w_rand = np.random.dirichlet(np.ones(6))
        curr_w = {k: v for k, v in zip(weights.keys(), w_rand)}
        acc = get_acc(curr_w)
        if acc > best_acc:
            best_acc = acc
            best_weights = curr_w

    print(f"Best Ensemble Acc: {best_acc:.4f}")
    print(f"Best Weights: {best_weights}")

    # --- 8. Final Test ---
    print("\n--- Final Test Evaluation ---")
    preds_test = {}
    for name, model in models.items():
        preds_test[name] = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    preds_test['LGBM'] = lgbm.predict_proba(X_feats_test)

    ens_test = np.zeros_like(list(preds_test.values())[0])
    for n, w in best_weights.items():
        ens_test += w * preds_test[n]

    test_acc = accuracy_score(y_test, ens_test.argmax(1))
    test_f1 = f1_score(y_test, ens_test.argmax(1), average='macro')

    print(f"TEST ACCURACY: {test_acc:.4f}")
    print(f"TEST F1 SCORE: {test_f1:.4f}")
    print(classification_report(y_test, ens_test.argmax(1), digits=4))

    # Save meta
    meta = {
        'label_encoder': le,
        'feature_scaler': scaler_feat,
        'ensemble_weights': best_weights,
        'n_classes': n_classes,
        'model_names': list(models.keys()) + ['LGBM']
    }
    with open(f'{ARTIFACTS_DIR}/pipeline_meta.pkl', 'wb') as f: pickle.dump(meta, f)
    print("Pipeline Finished.")

if __name__ == '__main__':
    main()