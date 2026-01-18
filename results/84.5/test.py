import os
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.signal import butter, filtfilt, iirnotch, mode
from tensorflow import keras

# ==================== CONFIGURATION ====================
MODEL_PATH = 'best_inception_se_tcn.keras' # Ensure this matches your uploaded file        # Folder containing new CSVs to test
FS = 512                                   # Sampling Frequency
WINDOW_MS = 400                            # Must match training (204 samples)
STRIDE_MS = 160                            # Overlap for smooth prediction

# ==================== 1. PREPROCESSING (Must match Training) ====================
class SignalPreprocessor:
    def __init__(self, fs=512, bandpass_low=20.0, bandpass_high=450.0, notch_freq=50.0):
        self.fs = fs
        nyq = fs / 2
        low = max(0.001, min(bandpass_low / nyq, 0.99))
        high = max(low + 0.01, min(bandpass_high / nyq, 0.999))
        self.b_bp, self.a_bp = butter(4, [low, high], btype='band')
        self.b_notch, self.a_notch = iirnotch(notch_freq, 30.0, self.fs) if notch_freq > 0 else (None, None)

    def transform(self, signal):
        # Apply filters
        if len(signal) > 12:
            signal = filtfilt(self.b_bp, self.a_bp, signal, axis=0)
            if self.b_notch is not None:
                signal = filtfilt(self.b_notch, self.a_notch, signal, axis=0)
        
        # Standardize (Z-Score)
        # Note: In production, use statistics from training, but per-window is okay for quick tests
        return (signal - np.mean(signal, axis=0)) / (np.std(signal, axis=0) + 1e-8)

    def segment(self, signal):
        win_sz = int(WINDOW_MS * self.fs / 1000) # Should be ~204 samples
        step = int(STRIDE_MS * self.fs / 1000)
        
        n = len(signal)
        if n < win_sz: return None
        
        n_win = (n - win_sz) // step + 1
        idx = np.arange(win_sz)[None, :] + np.arange(n_win)[:, None] * step
        return signal[idx]

# ==================== 2. MAIN PREDICTION PIPELINE ====================
def main():
    # A. Load Model
    print(f"🔄 Loading Model from {MODEL_PATH}...")
    try:
        model = keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully.")
        print(f"   Input Shape: {model.input_shape}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # B. Setup Preprocessor
    prep = SignalPreprocessor(fs=FS)

    # C. Load Data Files
    # Assuming you want to predict on all CSVs in DATA_DIR
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')] if os.path.exists(DATA_DIR) else []
    
    if not files:
        print(f"⚠️ No .csv files found in {DATA_DIR}. Please check the path.")
        # Create a dummy signal to demonstrate if no files exist
        print("   -> Generating dummy signal for demonstration...")
        dummy_signal = np.random.randn(2000, 8) # 2 seconds, 8 channels
        files = [('dummy_data.csv', dummy_signal)]
    else:
        print(f"📂 Found {len(files)} files to process.")

    # D. Process & Predict
    for entry in files:
        if isinstance(entry, tuple):
            filename, raw_data = entry
        else:
            filename = entry
            path = os.path.join(DATA_DIR, filename)
            try:
                df = pd.read_csv(path)
                # Ensure we only take the first 8 columns (sensors)
                raw_data = df.iloc[:, :8].values 
            except Exception as e:
                print(f"   ⚠️ Skipping {filename}: {e}")
                continue

        # 1. Preprocess
        clean_data = prep.transform(raw_data)
        
        # 2. Segment into Windows
        X_windows = prep.segment(clean_data)
        
        if X_windows is None:
            print(f"   ⚠️ Signal too short in {filename}")
            continue

        # 3. Predict
        # Check input shape compatibility
        if X_windows.shape[-1] != model.input_shape[-1]:
             print(f"   ❌ Channel Mismatch! Model expects {model.input_shape[-1]}, file has {X_windows.shape[-1]}")
             continue

        preds_probs = model.predict(X_windows, verbose=0)
        preds_classes = np.argmax(preds_probs, axis=1)

        # 4. Aggregation (Optional)
        # Get the most frequent predicted gesture for the file
        final_prediction = mode(preds_classes)[0]
        confidence = np.max(np.mean(preds_probs, axis=0))

        print(f"🎯 File: {filename}")
        print(f"   Total Windows: {len(preds_classes)}")
        print(f"   Predicted Class: {final_prediction} (Confidence: {confidence:.2f})")
        print(f"   Distribution: {np.bincount(preds_classes)}")
        print("-" * 30)

if __name__ == '__main__':
    # Create a dummy folder if it doesn't exist for testing
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    main()