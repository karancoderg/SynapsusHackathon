#!/usr/bin/env python3
"""
🚀 INFERENCE SCRIPT: 2-MODEL ENSEMBLE ON ALL DATA
--------------------------------------------------
Evaluates the Inception-SE + sEMG-Net ensemble on ALL data files.

CHANGES FROM ORIGINAL:
- Tests on ALL data (Session1 + Session2 + Session3)
- Provides comprehensive evaluation metrics
- Note: This will show overfitting since models were trained on Session1+2
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from scipy.stats import mode
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ==================== CONFIGURATION ====================
DATA_DIR = 'data'
ARTIFACTS_DIR = 'artifacts_final'
FS = 512
WINDOW_MS = 400
STRIDE_MS = 160
BATCH_SIZE = 128
L2_REG = 1e-4

# ==================== ARCHITECTURES (Must Match Training) ====================

def squeeze_excite_block(input_tensor, ratio=8):
    filters = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling1D()(input_tensor)
    se = layers.Dense(filters // ratio, activation='relu', kernel_regularizer=regularizers.l2(L2_REG))(se)
    se = layers.Dense(filters, activation='sigmoid', kernel_regularizer=regularizers.l2(L2_REG))(se)
    se = layers.Reshape((1, filters))(se)
    return layers.Multiply()([input_tensor, se])

def inception_block(x, filters, dilation_rate):
    b1 = layers.Conv1D(filters=filters//2, kernel_size=3, dilation_rate=dilation_rate,
                       padding='same', kernel_regularizer=regularizers.l2(L2_REG))(x)
    b1 = layers.BatchNormalization()(b1)
    b1 = layers.Activation('relu')(b1)
    b2 = layers.Conv1D(filters=filters//2, kernel_size=7, dilation_rate=dilation_rate,
                       padding='same', kernel_regularizer=regularizers.l2(L2_REG))(x)
    b2 = layers.BatchNormalization()(b2)
    b2 = layers.Activation('relu')(b2)
    return layers.Concatenate()([b1, b2])

def make_inception_se_tcn(input_shape, n_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.GaussianNoise(0.05)(inputs)
    filters = 64
    for dilation_rate in [1, 2, 4, 8]:
        prev_x = x
        x = inception_block(x, filters, dilation_rate)
        x = layers.Dropout(0.2)(x)
        x = squeeze_excite_block(x, ratio=8)
        if prev_x.shape[-1] != filters:
            prev_x = layers.Conv1D(filters=filters, kernel_size=1, padding='same')(prev_x)
        x = layers.Add()([x, prev_x])

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.MultiHeadAttention(key_dim=64, num_heads=4, dropout=0.3)(x, x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_classes, activation='softmax', dtype='float32')(x)
    return keras.Model(inputs, outputs, name='Inception_SE_Attn')

def conv_block(x, filters, kernel_size, pool=True):
    x = layers.Conv1D(filters, kernel_size, padding='same', kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.25)(x)
    if pool: x = layers.MaxPooling1D(pool_size=2)(x)
    return x

def make_semg_net(input_shape, n_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.GaussianNoise(0.05)(inputs)
    x = conv_block(x, 64, 9, pool=False)
    x = conv_block(x, 128, 5, pool=True)
    x = conv_block(x, 256, 3, pool=True)
    x = conv_block(x, 512, 3, pool=True)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_classes, activation='softmax', dtype='float32')(x)
    return keras.Model(inputs, outputs, name='sEMG_Net')

# ==================== DATA UTILS ====================

class SignalPreprocessor:
    def __init__(self, fs=1000, bandpass_low=20.0, bandpass_high=450.0, notch_freq=50.0):
        self.fs = fs
        self.nyq = fs / 2
        low = max(0.001, min(bandpass_low / self.nyq, 0.99))
        high = max(low + 0.01, min(bandpass_high / self.nyq, 0.999))
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

def get_session_files(data_dir, sessions):
    files = []
    for session in sessions:
        pattern = f'{data_dir}/**/{session}/**/*.csv'
        files.extend(sorted(glob.glob(pattern, recursive=True)))
    return files

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
    print("="*70)
    print("🚀 TESTING 2-MODEL ENSEMBLE ON ALL DATA")
    print("="*70)

    # 1. SETUP DATA - USE ALL DATA (Session1 + Session2 + Session3)
    print("\n📦 Loading Dataset...")
    all_files = get_session_files(DATA_DIR, ['Session1', 'Session2', 'Session3'])

    print(f"📄 Total files: {len(all_files)} (All Sessions)")

    # 2. LOAD & PREPROCESS
    print("\n⏳ Loading raw data...")
    all_data_raw, all_labels_raw = load_files_data(all_files)

    print("🔧 Preprocessing...")
    prep = SignalPreprocessor(fs=FS)
    prep.fit(all_data_raw)  # Fit on all data

    X_all, y_all_raw = window_data(all_data_raw, all_labels_raw, prep, WINDOW_MS, STRIDE_MS)

    # 3. LABEL ENCODING
    le = LabelEncoder().fit(y_all_raw)
    y_all = le.transform(y_all_raw)
    n_classes = len(le.classes_)
    input_shape = X_all.shape[1:]

    print(f"✅ Data Ready:")
    print(f"   Total windows: {X_all.shape[0]}")
    print(f"   Window shape: {X_all.shape[1:]}")
    print(f"   Classes: {le.classes_}")

    # 4. LOAD MODELS
    print("\n" + "-"*70)
    print("🔮 RUNNING ENSEMBLE INFERENCE ON ALL DATA")
    print("-"*70)

    models_config = [
        ("inception_se", make_inception_se_tcn),
        ("semg_net", make_semg_net)
    ]

    predictions = []
    model_names = []

    for name, builder in models_config:
        path = f'{ARTIFACTS_DIR}/best_{name}.keras'
        if not os.path.exists(path):
            print(f"❌ Error: Could not find {path}")
            continue

        print(f"\n⚡ Loading {name}...")
        model = builder(input_shape, n_classes)
        model.load_weights(path)

        probs = model.predict(X_all, batch_size=BATCH_SIZE, verbose=0)
        predictions.append(probs)
        model_names.append(name)

        # Individual model accuracy
        acc = accuracy_score(y_all, probs.argmax(axis=1))
        f1 = f1_score(y_all, probs.argmax(axis=1), average='macro')
        print(f"   ✓ {name} - Accuracy: {acc:.4f}, F1: {f1:.4f}")

        tf.keras.backend.clear_session()

    # 5. ENSEMBLE PREDICTION (SOFT VOTING)
    if len(predictions) == 2:
        print("\n" + "-"*70)
        print("🎯 ENSEMBLE RESULTS (Simple Averaging)")
        print("-"*70)
        
        ensemble_probs = (predictions[0] + predictions[1]) / 2.0
        final_preds = ensemble_probs.argmax(axis=1)

        # 6. COMPREHENSIVE RESULTS
        acc = accuracy_score(y_all, final_preds)
        f1 = f1_score(y_all, final_preds, average='macro')

        print(f"\n🏆 FINAL ENSEMBLE ACCURACY: {acc*100:.2f}%")
        print(f"🏆 FINAL ENSEMBLE F1 SCORE: {f1:.4f}")
        
        print("\n" + "="*70)
        print("📊 DETAILED CLASSIFICATION REPORT")
        print("="*70)
        print(classification_report(y_all, final_preds, 
                                   target_names=[f"Gesture {c}" for c in le.classes_],
                                   digits=4))

        # Confusion Matrix
        cm = confusion_matrix(y_all, final_preds)
        
        # Calculate per-class accuracy
        print("\n" + "="*70)
        print("📈 PER-CLASS ACCURACY")
        print("="*70)
        for i, gesture in enumerate(le.classes_):
            class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
            print(f"Gesture {gesture}: {class_acc*100:.2f}% ({cm[i, i]}/{cm[i].sum()} correct)")

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                    xticklabels=[f"G{c}" for c in le.classes_], 
                    yticklabels=[f"G{c}" for c in le.classes_])
        plt.title(f'2-Model Ensemble - All Data (Session1+2+3)\nAccuracy: {acc*100:.2f}%', fontsize=14)
        plt.xlabel('Predicted Gesture', fontsize=12)
        plt.ylabel('True Gesture', fontsize=12)
        plt.tight_layout()
        
        output_path = f'{ARTIFACTS_DIR}/ensemble_all_data_matrix.png'
        plt.savefig(output_path, dpi=150)
        print(f"\n📊 Confusion matrix saved to: {output_path}")
        
        # Save results to text file
        results_path = f'{ARTIFACTS_DIR}/ensemble_all_data_results.txt'
        with open(results_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("2-MODEL ENSEMBLE RESULTS - ALL DATA (Session1+2+3)\n")
            f.write("="*70 + "\n\n")
            f.write(f"Total Windows: {len(y_all)}\n")
            f.write(f"Accuracy: {acc*100:.2f}%\n")
            f.write(f"F1 Score: {f1:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(y_all, final_preds, 
                                         target_names=[f"Gesture {c}" for c in le.classes_],
                                         digits=4))
            f.write("\n\nConfusion Matrix:\n")
            f.write(str(cm))
        
        print(f"📄 Results saved to: {results_path}")
        
    else:
        print("❌ Error: Not enough models loaded for ensemble.")

    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
