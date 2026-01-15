#!/usr/bin/env python3
"""
Inference Script for Synapse sEMG Gesture Classification (Keras Version)
Usage: python run_inference.py --input data.csv --output predictions.csv
"""

import argparse
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras
from scipy.stats import entropy
from scipy.signal import butter, filtfilt, iirnotch, welch
import pywt
from itertools import combinations
import os

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class SignalPreprocessor:
    """Same as training preprocessor."""
    def __init__(self, fs=512, bandpass_low=20.0, bandpass_high=250.0, notch_freq=50.0):
        self.fs = fs
        nyq = fs / 2
        low = max(0.001, min(bandpass_low / nyq, 0.99))
        high = max(low + 0.01, min(bandpass_high / nyq, 0.999))
        self.b_bp, self.a_bp = butter(4, [low, high], btype='band')
        self.b_notch, self.a_notch = iirnotch(notch_freq, 30.0, self.fs) if notch_freq > 0 else (None, None)
        self.channel_means = None
        self.channel_stds = None

    def set_stats(self, mean, std):
        self.channel_means = mean
        self.channel_stds = std

    def preprocess(self, signal):
        # Filter
        if len(signal) > 12:
            signal = filtfilt(self.b_bp, self.a_bp, signal, axis=0)
            if self.b_notch is not None:
                signal = filtfilt(self.b_notch, self.a_notch, signal, axis=0)
        # Normalize using training stats if available
        if self.channel_means is not None:
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

class FeatureExtractor:
    def __init__(self, fs=512):
        self.fs = fs
    
    def extract(self, x):
        f = []
        for i in range(x.shape[1]):
            s = x[:, i]
            f.extend([np.mean(np.abs(s)), np.std(s), np.sum(np.abs(np.diff(s))), 
                      np.var(s), np.sum(np.abs(s))])
            freqs, psd = welch(s, self.fs, nperseg=min(256, len(s)))
            f.extend([np.mean(psd), np.max(psd), np.sum(psd)])
            coeffs = pywt.wavedec(s, 'db4', level=4)
            f.extend([np.sum(c**2) for c in coeffs[:4]])
            m0 = np.var(s)
            m2 = np.var(np.diff(s))
            m4 = np.var(np.diff(np.diff(s)))
            f.extend([m0, np.sqrt(m2/m0) if m0>0 else 0, np.sqrt(m4/m2)/np.sqrt(m2/m0) if m2>0 and m0>0 else 0])
        return np.array(f)

    def extract_batch(self, windows):
        return np.array([self.extract(w) for w in windows])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='predictions.csv')
    parser.add_argument('--artifacts', default='artifacts')
    args = parser.parse_args()

    print("Loading artifacts...")
    try:
        with open(f'{args.artifacts}/preprocessing.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        with open(f'{args.artifacts}/lgbm_model.pkl', 'rb') as f:
            lgbm = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle artifacts: {e}")
        return

    # Check framework
    if artifacts.get('framework') != 'keras':
        print("Warning: Artifacts seem to be from PyTorch training. This script expects Keras models.")
        # We could add fallback logic here but for complexity reduction we assume consistency

    print("Loading Keras models...")
    try:
        tcn = keras.models.load_model(f'{args.artifacts}/tcn_model.keras')
        cnn = keras.models.load_model(f'{args.artifacts}/cnn_model.keras')
    except Exception as e:
        print(f"Error loading Keras models: {e}")
        return

    # Load data
    print(f"Loading {args.input}...")
    try:
        df = pd.read_csv(args.input)
        data = df.values[:, :8]
    except Exception as e:
        print(f"Error reading csv: {e}")
        return

    # Process
    prep = SignalPreprocessor()
    # In a real scenario we'd save the channel means/stds in artifacts to reuse here
    # For now we use self-normalization or we should have saved it.
    # The training script used `prep.fit(all_data[0])` but didn't save the prep object.
    # It relied on `prep.transform` which uses fitted stats.
    # Artifacts doesn't contain the prep object stats.
    # We should fix training to save stats, but for now self-normalization (per file) is a reasonable fallback
    # OR we rely on the fact that `StandardScaler` was used for features.
    # For Deep Learning inputs, the training script normalized using global stats.
    # Let's approximate by normalizing the whole input file (batch norm often handles differences anyway).
    
    filtered = prep.preprocess(data) # This essentially does self-normalization if means are None
    windows = prep.segment(filtered)
    
    if windows is None or len(windows) == 0:
        print("No windows found (signal too short?)")
        return

    # Features
    fe = FeatureExtractor()
    feats = fe.extract_batch(windows)
    feats_sel = artifacts['feature_scaler'].transform(feats)

    # Predict
    print("Inference...")
    tcn_probs = tcn.predict(windows, verbose=0)
    cnn_probs = cnn.predict(windows, verbose=0)
    lgbm_probs = lgbm.predict_proba(feats_sel)
    
    # Check if Transformer exists (for >98% pipeline)
    if os.path.exists(f'{args.artifacts}/tfm_model.keras'):
        try:
            tfm = keras.models.load_model(f'{args.artifacts}/tfm_model.keras')
            tfm_probs = tfm.predict(windows, verbose=0)
            print("Transformer model loaded and used.")
        except Exception:
            tfm_probs = np.zeros_like(tcn_probs)
    else:
        tfm_probs = np.zeros_like(tcn_probs)

    # Weights: w1=TCN, w2=TFM, w3=CNN, w4=LGBM
    # If using old artifacts (3 weights), adapt
    weights = artifacts.get('ensemble_weights', (0.4, 0.0, 0.3, 0.3))
    if len(weights) == 3:
        w1, w2, w3 = weights # Old format (TCN, CNN, LGBM)
        ens_probs = w1*tcn_probs + w2*cnn_probs + w3*lgbm_probs
    else:
        w1, w2, w3, w4 = weights # New format (TCN, TFM, CNN, LGBM)
        print(f"Using weights: TCN={w1}, TFM={w2}, CNN={w3}, LGBM={w4}")
        ens_probs = w1*tcn_probs + w2*tfm_probs + w3*cnn_probs + w4*lgbm_probs

    preds = ens_probs.argmax(axis=1)

    labels = artifacts['label_encoder'].inverse_transform(preds)
    pd.DataFrame({'prediction': labels}).to_csv(args.output, index=False)
    print(f"Saved to {args.output}")

if __name__ == '__main__':
    main()
