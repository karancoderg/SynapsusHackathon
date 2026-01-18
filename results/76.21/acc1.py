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

# ==================== NEW: CORRECTED AUGMENTATION ====================
def augment_hybrid_correctly(X_raw, X_feat, y):
    """Augment raw and features INDEPENDENTLY to preserve statistics"""
    print(f"   ⚡ Augmenting Hybrid Data (Input: {len(X_raw)} windows)...")
    b, t, c = X_raw.shape
    
    # 1. Channel Masking (applied to BOTH streams)
    X_raw_mask = X_raw.copy()
    X_feat_mask = X_feat.copy()
    mask_indices = np.random.choice(b, size=int(b * 0.3), replace=False)
    for i in mask_indices:
        ch = np.random.randint(0, c)
        X_raw_mask[i, :, ch] = 0
        # Zero out features for masked channel (4 features per channel)
        X_feat_mask[i, ch*4:(ch+1)*4] = 0
    
    # 2. Magnitude Scaling (preserves RMS/MAV linearity)
    scales = np.random.uniform(0.85, 1.15, size=(b, 1, 1))
    X_raw_scale = X_raw * scales
    # Features scale proportionally
    X_feat_scale = X_feat * scales.reshape(b, 1)
    
    # 3. Temporal Jitter (sEMG-valid: shift activation timing ±10ms)
    X_raw_jitter = X_raw.copy()
    jitter_samples = np.random.randint(-5, 6, size=b)  # ±10ms at 512Hz
    for i, shift in enumerate(jitter_samples):
        if shift > 0:
            X_raw_jitter[i, shift:] = X_raw[i, :-shift]
            X_raw_jitter[i, :shift] = 0
        elif shift < 0:
            X_raw_jitter[i, :shift] = X_raw[i, -shift:]
            X_raw_jitter[i, shift:] = 0
    X_feat_jitter = np.array([extract_hybrid_features(w) for w in X_raw_jitter])
    
    # Combine
    X_raw_final = np.concatenate([X_raw, X_raw_mask, X_raw_scale, X_raw_jitter])
    X_feat_final = np.concatenate([X_feat, X_feat_mask, X_feat_scale, X_feat_jitter])
    y_final = np.concatenate([y, y, y, y])
    
    print(f"   ⚡ Augmentation complete. Size: {len(X_raw_final)} (4x)")
    return X_raw_final, X_feat_final, y_final

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

# ==================== NEW: CAUSAL SE BLOCK ====================
def causal_se_block(x, ratio=8):
    """SE with causal pooling (only uses past context)"""
    filters = x.shape[-1]
    # Causal average: use cumulative mean
    se = layers.Lambda(lambda inp: tf.cumsum(inp, axis=1) / tf.cast(tf.range(1, tf.shape(inp)[1]+1), tf.float32)[None, :, None])(x)
    se = layers.Lambda(lambda inp: inp[:, -1, :])(se)  # Take final cumulative average
    se = layers.Dense(filters // ratio, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1, filters))(se)
    return layers.Multiply()([x, se])

def temporal_self_attention(x, heads=4):
    """Lightweight multi-head self-attention for temporal context"""
    from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
    
    attn_out = MultiHeadAttention(num_heads=heads, key_dim=x.shape[-1]//heads)(x, x)
    x = LayerNormalization()(x + attn_out)  # Residual connection
    return x

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

# ==================== NEW: IMPROVED CAUSAL TCN ====================
def make_hybrid_tcn_causal(input_shape, feature_dim, n_classes):
    """Improved version with causal convolutions and self-attention"""
    # --- STREAM 1: RAW SIGNAL (CAUSAL TCN) ---
    raw_inputs = layers.Input(shape=input_shape, name="Raw_Signal")
    x = layers.GaussianNoise(0.02)(raw_inputs)

    filters = 64 
    dilations = [1, 2, 4, 8, 16]

    for i, dilation_rate in enumerate(dilations):
        prev_x = x
        # CAUSAL padding instead of same
        x = layers.Conv1D(filters=filters, kernel_size=3, dilation_rate=dilation_rate,
                          padding='causal', kernel_regularizer=l2(L2_REG))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(DROPOUT_CONV)(x)
        
        # Causal SE instead of regular SE
        x = causal_se_block(x)
        
        # Add self-attention at mid-depth (after dilation=4)
        if i == 2:
            x = temporal_self_attention(x, heads=4)
        
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
    
    return keras.Model(inputs=[raw_inputs, feat_inputs], outputs=outputs, name='HybridTCN_Causal')

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

# ==================== NEW: WARMUP LR CALLBACK ====================
class WarmupReduceLR(tf.keras.callbacks.Callback):
    def __init__(self, warmup_epochs=5, initial_lr=0.0001, max_lr=0.001):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.max_lr = max_lr
    
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * (epoch / self.warmup_epochs)
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
            print(f"   Warmup LR: {lr:.6f}")

# ==================== NEW: FILE-LEVEL EVALUATION ====================
def evaluate_file_level(model, test_data, test_labels, prep, scaler, window_ms, stride_ms):
    """Evaluate on FILE-level predictions (more realistic)"""
    print("\n" + "="*70)
    print("🎯 FILE-LEVEL EVALUATION (Realistic Metric)")
    print("="*70)
    
    file_preds = []
    file_trues = []
    
    for signal, labels in zip(test_data, test_labels):
        signal_filt = prep.transform(signal)
        windows = prep.segment(signal_filt, window_ms, stride_ms)
        if windows is None: 
            continue
        
        # Extract features for windows
        feats = np.array([extract_hybrid_features(w) for w in windows])
        feats = scaler.transform(feats)
        
        # Predict with TTA
        p1 = model.predict([windows, feats], verbose=0)
        noisy = windows + np.random.normal(0, 0.01, windows.shape)
        p2 = model.predict([noisy, feats], verbose=0)
        scaled = windows * 0.95
        p3 = model.predict([scaled, feats * 0.95], verbose=0)
        
        # Aggregate
        avg_probs = (p1 + p2 + p3) / 3
        window_preds = avg_probs.argmax(axis=1)
        
        # MAJORITY VOTE across all windows in this file
        file_pred = mode(window_preds, keepdims=True)[0][0]
        file_true = mode(labels, keepdims=True)[0][0]
        
        file_preds.append(file_pred)
        file_trues.append(file_true)
    
    acc = accuracy_score(file_trues, file_preds)
    f1 = f1_score(file_trues, file_preds, average='macro')
    
    print(f"\n🏆 FILE-LEVEL ACCURACY: {acc:.4f}")
    print(f"🏆 FILE-LEVEL F1 SCORE: {f1:.4f}")
    print(f"   Total files evaluated: {len(file_preds)}")
    
    return acc, f1

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

    # 4. Augmentation (Train Only) - USING CORRECTED VERSION
    print("\n[Augmentation] Using CORRECTED hybrid augmentation...")
    X_train_raw, X_train_feat, y_train_raw = augment_hybrid_correctly(X_train_raw, X_train_feat, y_train_raw)

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
    print("\n" + "-"*40 + "\n🏋️ TRAIN MODEL: HYBRID ATTENTION TCN (IMPROVED)\n" + "-"*40)
    
    # Using improved causal architecture
    model = make_hybrid_tcn_causal(X_train_raw.shape[1:], X_train_feat.shape[1], n_classes)
    
    # Stable learning rate schedule with warmup
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), 
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Callbacks with warmup and stable reduction
    warmup = WarmupReduceLR(warmup_epochs=5, initial_lr=0.0001, max_lr=0.001)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1)
    es = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)

    model.fit([X_train_raw, X_train_feat], y_train, 
              validation_data=([X_val_raw, X_val_feat], y_val),
              epochs=EPOCHS, 
              batch_size=BATCH_SIZE, 
              callbacks=[warmup, reduce_lr, es], 
              verbose=1)
    
    model.save(f'{ARTIFACTS_DIR}/best_hybrid_tcn_improved.keras')

    # ==================== WINDOW-LEVEL TTA EVALUATION ====================
    print("\n" + "="*70 + "\n🎯 WINDOW-LEVEL EVALUATION (With TTA)\n" + "="*70)
    
    print("   Running TTA (3-View Ensemble)...")
    
    # View 1: Normal
    p1 = model.predict([X_test_raw, X_test_feat], batch_size=BATCH_SIZE, verbose=0)
    
    # View 2: Noisy (Simulate sensor fuzz)
    noisy_raw = X_test_raw + np.random.normal(0, 0.01, X_test_raw.shape)
    p2 = model.predict([noisy_raw, X_test_feat], batch_size=BATCH_SIZE, verbose=0)
    
    # View 3: Scaled (Simulate weaker muscle signal)
    scaled_raw = X_test_raw * 0.95
    scaled_feat = X_test_feat * 0.95
    p3 = model.predict([scaled_raw, scaled_feat], batch_size=BATCH_SIZE, verbose=0)
    
    # Soft Voting
    final_probs = (p1 + p2 + p3) / 3
    final_preds = final_probs.argmax(axis=1)
    
    window_acc = accuracy_score(y_test, final_preds)
    window_f1 = f1_score(y_test, final_preds, average='macro')
    
    print(f"\n🏆 WINDOW-LEVEL ACCURACY: {window_acc:.4f}")
    print(f"🏆 WINDOW-LEVEL F1 SCORE: {window_f1:.4f}")

    # ==================== FILE-LEVEL EVALUATION ====================
    file_acc, file_f1 = evaluate_file_level(model, test_data, test_labels, prep, scaler, WINDOW_MS, STRIDE_MS)
    
    # ==================== SUMMARY ====================
    print("\n" + "="*70)
    print("📊 FINAL PERFORMANCE SUMMARY")
    print("="*70)
    print(f"Window-Level Accuracy: {window_acc:.4f}")
    print(f"Window-Level F1:       {window_f1:.4f}")
    print(f"File-Level Accuracy:   {file_acc:.4f}")
    print(f"File-Level F1:         {file_f1:.4f}")
    print("="*70)

if __name__ == '__main__':
    main()