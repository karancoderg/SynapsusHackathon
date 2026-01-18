#!/usr/bin/env python3
"""
HIGH-PERFORMANCE TCN PIPELINE
for Synapse sEMG Challenge

🚀 UPGRADES FOR >80% ACCURACY:
1. MODEL: Standard Conv1D (Spatial Correlations) instead of Separable.
2. CAPACITY: 64 Filters to capture subtler patterns.
3. CONTEXT: Added dilation rate 8 to see full gesture duration.
"""

import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import glob
import re
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision, callbacks
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import mode
from scipy.signal import butter, filtfilt, iirnotch
import warnings

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
DATA_DIR = 'data'
ARTIFACTS_DIR = 'artifacts'
FS = 512
EPOCHS = 100
BATCH_SIZE = 128
RANDOM_SEED = 42
VAL_FILE_RATIO = 0.50

# Windowing
WINDOW_MS = 400
STRIDE_MS = 160  # Tip: Lower this to 80 for even more training data

# TCN Hyperparameters
L2_REG = 1e-4
DROPOUT_CONV = 0.2
DROPOUT_HEAD = 0.5

# GPU Setup
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print("✅ Mixed Precision (FP16) Enabled")
except:
    pass

# Set seeds
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ==================== AUGMENTATION ====================

def augment_dataset_advanced(X, y):
    """
    Advanced Augmentation:
    1. Channel Masking (Sensor noise/failure simulation)
    2. MixUp (Data blending)
    """
    print(f"   ⚡ Augmenting Data (Input: {len(X)} windows)...")
    b, t, c = X.shape

    # 1. Channel Masking
    X_mask = X.copy()
    mask_indices = np.random.choice(b, size=int(b * 0.5), replace=False)
    for i in mask_indices:
        ch = np.random.randint(0, c)
        X_mask[i, :, ch] = 0
    X_mask = X_mask + np.random.normal(0, 0.02, size=X_mask.shape)

    # 2. MixUp
    indices = np.random.permutation(b)
    X_shuffled = X[indices]
    alpha = 0.2
    lam = np.random.beta(alpha, alpha, size=(b, 1, 1))
    
    X_mix = lam * X + (1 - lam) * X_shuffled
    y_mix = y.copy() # Dominant label strategy

    # Concatenate
    X_final = np.concatenate([X, X_mask, X_mix], axis=0)
    y_final = np.concatenate([y, y, y_mix], axis=0)

    print(f"   ⚡ Augmentation complete. Size: {len(X_final)} (3x)")
    return X_final, y_final

# ==================== PREPROCESSING ====================

class SignalPreprocessor:
    def __init__(self, fs=1000, bandpass_low=20.0, bandpass_high=450.0, notch_freq=50.0):
        self.fs = fs
        nyq = fs / 2
        low = max(0.001, min(bandpass_low / nyq, 0.99))
        high = max(low + 0.01, min(bandpass_high / nyq, 0.999))
        self.b_bp, self.a_bp = butter(4, [low, high], btype='band')
        self.b_notch, self.a_notch = iirnotch(notch_freq, 30.0, self.fs) if notch_freq > 0 else (None, None)
        self.channel_means, self.channel_stds = None, None
        self.fitted = False

    def fit(self, signals_list):
        all_signals = np.concatenate(signals_list, axis=0)
        self.channel_means = np.mean(all_signals, axis=0)
        self.channel_stds = np.std(all_signals, axis=0) + 1e-8
        self.fitted = True
        return self

    def transform(self, signal):
        if len(signal) > 12:
            signal = filtfilt(self.b_bp, self.a_bp, signal, axis=0)
            if self.b_notch is not None:
                signal = filtfilt(self.b_notch, self.a_notch, signal, axis=0)
        if self.fitted:
            return (signal - self.channel_means) / self.channel_stds
        return (signal - np.mean(signal, axis=0)) / (np.std(signal, axis=0) + 1e-8)

    def segment(self, signal, window_ms=200, stride_ms=100):
        win_sz = int(window_ms * self.fs / 1000)
        step = int(stride_ms * self.fs / 1000)
        n = len(signal)
        if n < win_sz: return None
        n_win = (n - win_sz) // step + 1
        idx = np.arange(win_sz)[None, :] + np.arange(n_win)[:, None] * step
        return signal[idx]

# ==================== STRONGER TCN ARCHITECTURE ====================

def make_strong_tcn(input_shape, n_classes):
    """
    Upgraded TCN for Higher Accuracy:
    - Standard Conv1D (Spatial awareness)
    - 64 Filters (Higher capacity)
    - Dilation up to 8 (Broader context)
    """
    inputs = layers.Input(shape=input_shape)
    
    # 1. Noise for Robustness
    x = layers.GaussianNoise(0.05)(inputs)

    # [UPGRADE] Increased filters & Dilation depth
    filters = 64 
    dilations = [1, 2, 4, 8] # Added 8 to see "longer" patterns

    for dilation_rate in dilations:
        prev_x = x

        # [UPGRADE] Switched SeparableConv1D -> Conv1D
        # This allows the model to learn correlations between different muscles immediately.
        x = layers.Conv1D(filters=filters, kernel_size=3, dilation_rate=dilation_rate,
                          padding='same', kernel_regularizer=l2(L2_REG))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(DROPOUT_CONV)(x)

        x = layers.Conv1D(filters=filters, kernel_size=3, dilation_rate=dilation_rate,
                          padding='same', kernel_regularizer=l2(L2_REG))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(DROPOUT_CONV)(x)

        # Residual Projector
        if prev_x.shape[-1] != filters:
            prev_x = layers.Conv1D(filters=filters, kernel_size=1, padding='same',
                                   kernel_regularizer=l2(L2_REG))(prev_x)
        
        # Add Residual
        x = layers.Add()([x, prev_x])

    # Hybrid Pooling
    pool_max = layers.GlobalMaxPooling1D()(x)
    pool_avg = layers.GlobalAveragePooling1D()(x)
    x = layers.Concatenate()([pool_max, pool_avg])

    # Classification Head
    x = layers.Dense(64, activation='relu', kernel_regularizer=l2(L2_REG))(x)
    x = layers.Dropout(DROPOUT_HEAD)(x)

    outputs = layers.Dense(n_classes, activation='softmax', dtype='float32', kernel_regularizer=l2(L2_REG))(x)

    return keras.Model(inputs, outputs, name='StrongTCN')

# ==================== UTILS ====================

def get_session_files(data_dir, sessions):
    files = []
    for session in sessions:
        pattern = f'{data_dir}/**/{session}/**/*.csv'
        files.extend(sorted(glob.glob(pattern, recursive=True)))
    return files

def split_files_by_ratio(files, val_ratio, seed=RANDOM_SEED):
    gesture_files = {}
    for f in files:
        match = re.search(r'gesture(\d+)', f)
        if match:
            g = int(match.group(1))
            gesture_files.setdefault(g, []).append(f)
    train, val = [], []
    rng = random.Random(seed)
    for g, gfiles in gesture_files.items():
        rng.shuffle(gfiles)
        n_val = max(1, int(len(gfiles) * val_ratio))
        val.extend(gfiles[:n_val])
        train.extend(gfiles[n_val:])
    return train, val

def load_files_data(file_list):
    data_list, labels_list = [], []
    for f in file_list:
        try:
            lbl = int(re.search(r'gesture(\d+)', f).group(1))
            d = pd.read_csv(f).values
            if d.shape[1] >= 8:
                data_list.append(d)
                labels_list.append(np.full(len(d), lbl))
        except: pass
    return data_list, labels_list

def window_data(data_list, labels_list, prep, window_ms, stride_ms):
    X_wins, y_wins = [], []
    win_sz = int(window_ms * FS / 1000)
    step = int(stride_ms * FS / 1000)
    for d, l in zip(data_list, labels_list):
        d_filt = prep.transform(d)
        w = prep.segment(d_filt, window_ms, stride_ms)
        if w is not None:
            X_wins.append(w)
            n_win = (len(d) - win_sz) // step + 1
            idx = np.arange(win_sz)[None, :] + np.arange(n_win)[:, None] * step
            w_modes = mode(l[idx], axis=1, keepdims=True)[0].flatten()
            y_wins.append(w_modes)
    if not X_wins: return None, None
    return np.concatenate(X_wins), np.concatenate(y_wins)

# ==================== MAIN ====================

def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    print("="*70 + "\n🚀 HIGH-PERFORMANCE TCN TRAINING (No Ensemble)\n" + "="*70)

    # 1. Load Data
    existing_csvs = glob.glob(f'{DATA_DIR}/**/*.csv', recursive=True)
    if not existing_csvs:
        import gdown, zipfile
        gdown.download('https://drive.google.com/uc?id=16iNEwhThf2LcX7rOOVM03MTZiwq7G51x', 'dataset.zip', quiet=False)
        with zipfile.ZipFile('dataset.zip', 'r') as z: z.extractall(DATA_DIR)
        os.remove('dataset.zip')

    train_files = get_session_files(DATA_DIR, ['Session1', 'Session2'])
    session3_files = get_session_files(DATA_DIR, ['Session3'])
    val_files, test_files = split_files_by_ratio(session3_files, VAL_FILE_RATIO)

    print(f"\n[Data Loaded] Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    # 2. Raw Data
    train_data, train_labels = load_files_data(train_files)
    val_data, val_labels = load_files_data(val_files)
    test_data, test_labels = load_files_data(test_files)

    # 3. Preprocess
    prep = SignalPreprocessor(fs=FS)
    prep.fit(train_data)

    print("\n[Windowing] Processing...")
    X_train, y_train_raw = window_data(train_data, train_labels, prep, WINDOW_MS, STRIDE_MS)
    X_val, y_val_raw = window_data(val_data, val_labels, prep, WINDOW_MS, STRIDE_MS)
    X_test, y_test_raw = window_data(test_data, test_labels, prep, WINDOW_MS, STRIDE_MS)

    # 4. Augmentation
    X_train, y_train_raw = augment_dataset_advanced(X_train, y_train_raw)

    # 5. Encoding
    le = LabelEncoder()
    le.fit(y_train_raw)
    y_train = le.transform(y_train_raw)

    def safe_transform(enc, labels):
        return np.array([enc.transform([l])[0] if l in enc.classes_ else -1 for l in labels])

    y_val = safe_transform(le, y_val_raw)
    y_test = safe_transform(le, y_test_raw)
    n_classes = len(le.classes_)
    input_shape = X_train.shape[1:]

    # ==================== TRAINING PHASE ====================
    print("\n" + "-"*40 + "\n🏋️ TRAIN MODEL: STRONG TCN\n" + "-"*40)
    
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    
    # Early Stopping to prevent overfitting
    es = callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1)
    lr_schedule = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, verbose=1)

    model = make_strong_tcn(input_shape, n_classes)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    
    model.summary()

    model.fit(X_train, y_train, 
              validation_data=(X_val, y_val),
              epochs=EPOCHS, 
              batch_size=BATCH_SIZE, 
              callbacks=[es, lr_schedule], 
              verbose=1)
    
    # Save
    model.save(f'{ARTIFACTS_DIR}/best_tcn_model.keras')
    print("✅ Model Saved.")

    # ==================== EVALUATION ====================
    print("\n" + "="*70 + "\n🎯 FINAL EVALUATION\n" + "="*70)
    
    test_probs = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    test_preds = test_probs.argmax(axis=1)
    
    acc = accuracy_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds, average='macro')
    
    print(f"🏆 TCN ACCURACY:    {acc:.4f}")
    print(f"🏆 TCN F1 SCORE:    {f1:.4f}")

if __name__ == '__main__':
    main()