#!/usr/bin/env python3
"""
HIGH-PERFORMANCE HYBRID TCN PIPELINE
Optimized for Colab T4 (15GB VRAM)
Target: >85% Accuracy via Hybrid Fusion & Attention
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
BATCH_SIZE = 64  # Lower batch size = better generalization
RANDOM_SEED = 42
VAL_FILE_RATIO = 0.50

# [UPGRADE 1] Stride Reduction: 160 -> 40ms. 
# This generates ~4x more training samples, crucial for >80% acc.
WINDOW_MS = 400
STRIDE_MS = 40 

# Hyperparameters
L2_REG = 5e-4
DROPOUT_CONV = 0.25
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

# ==================== HELPER: FEATURE EXTRACTION ====================
def extract_hybrid_features(window):
    """
    Extracts physical features (RMS, WL, ZC) to fuse with Deep Learning.
    This helps the model distinguish subtle force differences.
    """
    # window shape: (time, channels)
    # 1. Root Mean Square (Energy)
    rms = np.sqrt(np.mean(window**2, axis=0))
    # 2. Waveform Length (Complexity)
    wl = np.sum(np.abs(np.diff(window, axis=0)), axis=0)
    # 3. Mean Absolute Value
    mav = np.mean(np.abs(window), axis=0)
    # 4. Zero Crossings (Frequency approx)
    zc = np.sum(np.diff(np.sign(window), axis=0) != 0, axis=0)
    
    return np.concatenate([rms, wl, mav, zc])

# ==================== AUGMENTATION ====================

def augment_dataset_advanced(X, y):
    print(f"   ⚡ Augmenting Data (Input: {len(X)} windows)...")
    b, t, c = X.shape

    # 1. Channel Masking (Sensor Failure)
    X_mask = X.copy()
    mask_indices = np.random.choice(b, size=int(b * 0.4), replace=False) # Reduced to 40%
    for i in mask_indices:
        ch = np.random.randint(0, c)
        X_mask[i, :, ch] = 0
    X_mask = X_mask + np.random.normal(0, 0.01, size=X_mask.shape)

    # 2. MixUp (Data Blending)
    indices = np.random.permutation(b)
    X_shuffled = X[indices]
    alpha = 0.2
    lam = np.random.beta(alpha, alpha, size=(b, 1, 1))
    X_mix = lam * X + (1 - lam) * X_shuffled
    
    # 3. Magnitude Scaling (Simulate Fatigue)
    X_scale = X * np.random.uniform(0.8, 1.2, size=(b, 1, 1))

    # Combine (Original + Mask + Mix + Scale = 4x Data)
    X_final = np.concatenate([X, X_mask, X_mix, X_scale], axis=0)
    y_final = np.concatenate([y, y, y, y], axis=0)

    print(f"   ⚡ Augmentation complete. Size: {len(X_final)} (4x)")
    return X_final, y_final

# ==================== PREPROCESSING (UNCHANGED) ====================

class SignalPreprocessor:
    def __init__(self, fs=1000, bandpass_low=20.0, bandpass_high=450.0, notch_freq=50.0):
        self.fs = fs
        nyq = fs / 2
        low = max(0.001, min(bandpass_low / nyq, 0.99))
        high = max(low + 0.01, min(bandpass_high / nyq, 0.999))
        self.b_bp, self.a_bp = butter(4, [low, high], btype='band')
        self.b_notch, self.a_notch = iirnotch(notch_freq, 30.0, self.fs) if notch_freq > 0 else (None, None)
        self.fitted = False
        self.channel_means = None
        self.channel_stds = None

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

# ==================== UPGRADED ARCHITECTURE: HYBRID ATTENTION TCN ====================

def se_block(input_tensor, ratio=8):
    """
    [UPGRADE 3] Squeeze-and-Excitation Block
    Allows the model to learn 'which muscle matters' for the current window.
    """
    filters = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling1D()(input_tensor)
    se = layers.Dense(filters // ratio, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1, filters))(se)
    return layers.Multiply()([input_tensor, se])

def make_hybrid_tcn(input_shape, feature_dim, n_classes):
    # --- STREAM 1: RAW SIGNAL (TCN) ---
    raw_inputs = layers.Input(shape=input_shape, name="Raw_Signal")
    x = layers.GaussianNoise(0.02)(raw_inputs) # Slightly reduced noise

    filters = 64 
    dilations = [1, 2, 4, 8, 16] # [UPGRADE] Added dilation 16 for very long context

    for dilation_rate in dilations:
        prev_x = x
        x = layers.Conv1D(filters=filters, kernel_size=3, dilation_rate=dilation_rate,
                          padding='same', kernel_regularizer=l2(L2_REG))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(DROPOUT_CONV)(x)
        
        # Add Attention
        x = se_block(x) 
        
        if prev_x.shape[-1] != filters:
            prev_x = layers.Conv1D(filters=filters, kernel_size=1, padding='same')(prev_x)
        x = layers.Add()([x, prev_x])

    pool_max = layers.GlobalMaxPooling1D()(x)
    pool_avg = layers.GlobalAveragePooling1D()(x)
    x_flat = layers.Concatenate()([pool_max, pool_avg])

    # --- STREAM 2: EXPERT FEATURES (DENSE) ---
    feat_inputs = layers.Input(shape=(feature_dim,), name="Handcrafted_Features")
    y = layers.Dense(64, activation='relu', kernel_regularizer=l2(L2_REG))(feat_inputs)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.3)(y)

    # --- FUSION ---
    combined = layers.Concatenate()([x_flat, y])
    z = layers.Dense(128, activation='relu', kernel_regularizer=l2(L2_REG))(combined)
    z = layers.Dropout(DROPOUT_HEAD)(z)

    outputs = layers.Dense(n_classes, activation='softmax', dtype='float32')(z)
    
    return keras.Model(inputs=[raw_inputs, feat_inputs], outputs=outputs, name='HybridTCN')

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

def window_hybrid_data(data_list, labels_list, prep, window_ms, stride_ms):
    """Modified to return BOTH raw windows AND extracted features"""
    X_wins, X_feats, y_wins = [], [], []
    win_sz = int(window_ms * FS / 1000)
    step = int(stride_ms * FS / 1000)
    
    for d, l in zip(data_list, labels_list):
        d_filt = prep.transform(d)
        w = prep.segment(d_filt, window_ms, stride_ms)
        if w is not None:
            X_wins.append(w)
            # Calculate features for every window
            feats = np.array([extract_hybrid_features(win) for win in w])
            X_feats.append(feats)
            
            n_win = (len(d) - win_sz) // step + 1
            idx = np.arange(win_sz)[None, :] + np.arange(n_win)[:, None] * step
            w_modes = mode(l[idx], axis=1, keepdims=True)[0].flatten()
            y_wins.append(w_modes)
            
    if not X_wins: return None, None, None
    return np.concatenate(X_wins), np.concatenate(X_feats), np.concatenate(y_wins)

# ==================== MAIN ====================

def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    print("="*80 + "\n🚀 HIGH-PERFORMANCE HYBRID TCN (Optimized for Colab T4)\n" + "="*80)

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

    # 2. Raw Data
    train_data, train_labels = load_files_data(train_files)
    val_data, val_labels = load_files_data(val_files)
    test_data, test_labels = load_files_data(test_files)

    # 3. Preprocess
    prep = SignalPreprocessor(fs=FS)
    prep.fit(train_data) # Fit ONLY on training data (No Leakage)

    print("\n[Windowing] Processing Hybrid Inputs...")
    X_train_raw, X_train_feat, y_train_raw = window_hybrid_data(train_data, train_labels, prep, WINDOW_MS, STRIDE_MS)
    X_val_raw, X_val_feat, y_val_raw = window_hybrid_data(val_data, val_labels, prep, WINDOW_MS, STRIDE_MS)
    X_test_raw, X_test_feat, y_test_raw = window_hybrid_data(test_data, test_labels, prep, WINDOW_MS, STRIDE_MS)

    # 4. Augmentation (Train Only)
    # Augment Raw Data
    X_train_raw, y_train_raw = augment_dataset_advanced(X_train_raw, y_train_raw)
    
    # We must calculate features for the NEW augmented raw windows
    # (Since we just created synthetic windows, we need to extract their stats)
    print("   ⚡ Extracting features for augmented data...")
    X_train_feat = np.array([extract_hybrid_features(w) for w in X_train_raw])

    # 5. Scaling
    scaler = StandardScaler()
    X_train_feat = scaler.fit_transform(X_train_feat) # Fit ONLY on Train
    X_val_feat = scaler.transform(X_val_feat)
    X_test_feat = scaler.transform(X_test_feat)

    # 6. Encoding
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_val = le.transform(y_val_raw)
    y_test = le.transform(y_test_raw)
    
    n_classes = len(le.classes_)

    # ==================== TRAINING ====================
    print("\n" + "-"*40 + "\n🏋️ TRAIN MODEL: HYBRID ATTENTION TCN\n" + "-"*40)
    
    # [UPGRADE 4] Cosine Decay Restarts
    # Aggressively lowers LR to find minima, then restarts to escape local traps.
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=0.001,
        first_decay_steps=1000, 
        t_mul=2.0, 
        m_mul=0.9, 
        alpha=0.01
    )
    
    model = make_hybrid_tcn(X_train_raw.shape[1:], X_train_feat.shape[1], n_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule), 
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Early stopping with high patience because Cosine Decay needs time
    es = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)

    model.fit([X_train_raw, X_train_feat], y_train, 
              validation_data=([X_val_raw, X_val_feat], y_val),
              epochs=EPOCHS, 
              batch_size=BATCH_SIZE, 
              callbacks=[es], 
              verbose=1)
    
    model.save(f'{ARTIFACTS_DIR}/best_hybrid_tcn.keras')

    # ==================== TTA EVALUATION ====================
    print("\n" + "="*70 + "\n🎯 FINAL EVALUATION (With TTA)\n" + "="*70)
    
    # [UPGRADE 5] Test-Time Augmentation (TTA)
    # Predict 3 times: Normal, Noisy, Scaled. Average the results.
    print("   Running TTA (3-View Ensemble)...")
    
    # View 1: Normal
    p1 = model.predict([X_test_raw, X_test_feat], batch_size=BATCH_SIZE, verbose=0)
    
    # View 2: Noisy (Simulate sensor fuzz)
    noisy_raw = X_test_raw + np.random.normal(0, 0.01, X_test_raw.shape)
    p2 = model.predict([noisy_raw, X_test_feat], batch_size=BATCH_SIZE, verbose=0)
    
    # View 3: Scaled (Simulate weaker muscle signal)
    scaled_raw = X_test_raw * 0.95
    scaled_feat = X_test_feat * 0.95 # Features must scale too
    p3 = model.predict([scaled_raw, scaled_feat], batch_size=BATCH_SIZE, verbose=0)
    
    # Soft Voting
    final_probs = (p1 + p2 + p3) / 3
    final_preds = final_probs.argmax(axis=1)
    
    acc = accuracy_score(y_test, final_preds)
    f1 = f1_score(y_test, final_preds, average='macro')
    
    print(f"🏆 HYBRID ACCURACY:   {acc:.4f}")
    print(f"🏆 HYBRID F1 SCORE:   {f1:.4f}")

if __name__ == '__main__':
    main()