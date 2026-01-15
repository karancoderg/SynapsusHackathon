#!/usr/bin/env python3
"""
Complete Training Pipeline for Synapse sEMG Challenge (PyTorch Version)
Replicates the functionality of the Keras pipeline with PyTorch.

Optimized for NVIDIA GPUs:
- Batch Size: 1024
- Mixed Precision (AMP): Enabled
- Models: TCN, SE-TCN, TCN+Attention, Transformer
"""

import os
import glob
import re
import pickle
import numpy as np
import pandas as pd
import random
import warnings
from scipy.stats import mode
from scipy import signal as scipy_signal
from scipy.signal import butter, filtfilt, iirnotch
import pywt
from joblib import Parallel, delayed

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import lightgbm as lgb

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_DIR = 'data'
ARTIFACTS_DIR = 'artifacts_pytorch'
FS = 512
EPOCHFS = 512
BATCH_SIZE = 1024
EPOCHS = 100
RANDOM_SEED = 42
WINDOW_MS = 400

# Set Seeds
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(RANDOM_SEED)

# Device Configuration
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"   CUDA Version: {torch.version.cuda}")
else:
    device = torch.device('cpu')
    print("⚠️ No GPU detected. Running on CPU (Expect slow performance).")

print(f"Using device: {device}")

# ==================== PREPROCESSING (Same as Keras) ====================

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

    def fit(self, signals):
        if signals.ndim == 3: signals = signals.reshape(-1, signals.shape[-1])
        self.channel_means = np.mean(signals, axis=0)
        self.channel_stds = np.std(signals, axis=0) + 1e-8
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
    def __init__(self, fs=1000):
        self.fs = fs

    def extract(self, x):
        f = []
        for i in range(x.shape[1]):
            s = x[:, i]
            f.extend([np.mean(np.abs(s)), np.std(s), np.sum(np.abs(np.diff(s))),
                      np.var(s), np.sum(np.abs(s))])
            freqs, psd = scipy_signal.welch(s, self.fs, nperseg=min(256, len(s)))
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

# ==================== PYTORCH MODELS ====================

class SeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv1d, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, channels, ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // ratio, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

# --- 1. TCN Model ---
class TCN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TCN, self).__init__()
        layers_list = []
        in_ch = input_dim
        
        # [64, 64, 128, 128], k=5
        for out_ch, d in zip([64, 64, 128, 128], [1, 2, 4, 8]):
            # Residual connection preparation
            res_conv = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
            
            # Block
            self.add_module(f'res_conv_{d}', res_conv) 
            
            block = nn.Sequential(
                SeparableConv1d(in_ch, out_ch, 5, padding=2*d, dilation=d),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(0.2),
                SeparableConv1d(out_ch, out_ch, 5, padding=2*d, dilation=d),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            layers_list.append((block, res_conv))
            in_ch = out_ch
            
        self.blocks = nn.ModuleList([l[0] for l in layers_list])
        self.res_convs = nn.ModuleList([l[1] for l in layers_list])
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (N, L, C) -> (N, C, L)
        x = x.permute(0, 2, 1)
        
        for block, res_conv in zip(self.blocks, self.res_convs):
            res = res_conv(x)
            x = block(x)
            x = x + res
            x = torch.relu(x)
            
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

# --- 2. SE-TCN Model ---
class SETCN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SETCN, self).__init__()
        layers_list = []
        in_ch = input_dim
        
        # [64, 128, 128], k=3
        for out_ch, d in zip([64, 128, 128], [1, 2, 4]):
            res_conv = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
            
            block = nn.Sequential(
                SeparableConv1d(in_ch, out_ch, 3, padding=1*d, dilation=d),
                nn.BatchNorm1d(out_ch),
                nn.SiLU(), # Swish
                nn.Dropout(0.2),
                SeparableConv1d(out_ch, out_ch, 3, padding=1*d, dilation=d),
                nn.BatchNorm1d(out_ch)
            )
            se = SEBlock(out_ch)
            layers_list.append((block, se, res_conv))
            in_ch = out_ch
            
        self.blocks = nn.ModuleList([l[0] for l in layers_list])
        self.se_blocks = nn.ModuleList([l[1] for l in layers_list])
        self.res_convs = nn.ModuleList([l[2] for l in layers_list])
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        for block, se, res_conv in zip(self.blocks, self.se_blocks, self.res_convs):
            res = res_conv(x)
            x = block(x)
            x = se(x)
            x = x + res
            x = torch.nn.functional.silu(x)
            
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

# --- 3. TCN + Attention Model ---
class TCNAttention(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TCNAttention, self).__init__()
        layers_list = []
        in_ch = input_dim
        
        # [64, 128, 128], k=3
        for out_ch, d in zip([64, 128, 128], [1, 2, 4]):
            res_conv = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
            
            block = nn.Sequential(
                SeparableConv1d(in_ch, out_ch, 3, padding=1*d, dilation=d),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(0.2),
                SeparableConv1d(out_ch, out_ch, 3, padding=1*d, dilation=d),
                nn.BatchNorm1d(out_ch)
            )
            layers_list.append((block, res_conv))
            in_ch = out_ch
            
        self.blocks = nn.ModuleList([l[0] for l in layers_list])
        self.res_convs = nn.ModuleList([l[1] for l in layers_list])
        
        # Attention
        self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.ln = nn.LayerNormalization(128)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1) # (N, C, L)
        for block, res_conv in zip(self.blocks, self.res_convs):
            res = res_conv(x)
            x = block(x)
            x = x + res
            x = torch.relu(x)
            
        # Attention expects (N, L, E)
        x = x.permute(0, 2, 1) 
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = self.ln(x)
        
        # Back to (N, C, L) for pooling
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

# --- 4. Transformer Model ---
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=128, nhead=4, num_layers=4):
        super(TransformerModel, self).__init__()
        self.project = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, 
                                                   dropout=0.1, batch_first=True, activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (N, L, C)
        x = self.project(x)
        x = self.transformer_encoder(x) # (N, L, E)
        x = x.permute(0, 2, 1) # (N, E, L)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

# ==================== TRAINING UTILS ====================

def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    scaler = GradScaler()
    
    best_acc = 0.0
    best_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            with autocast():
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * X_batch.size(0)
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()
            
        train_loss /= total
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item() * X_batch.size(0)
                _, predicted = outputs.max(1)
                total += y_batch.size(0)
                correct += predicted.eq(y_batch).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
                
        val_loss /= total
        val_acc = correct / total
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        
        print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - acc: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f} - val_f1: {val_f1:.4f}")
        
        scheduler.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 30:
                print("Early stopping")
                break
                
    model.load_state_dict(best_state)
    return model

def predict_model(model, loader):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_probs)

# ==================== MAIN ====================

def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    # --- Data Loading ---
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

    # --- Preprocessing ---
    print(f"Preprocessing (FS={FS}, Window={WINDOW_MS}ms)...")
    prep = SignalPreprocessor(fs=FS)
    prep.fit(all_data[0]) 
    
    X_wins, y_wins = [], []
    for d, l in zip(all_data, all_labels):
        d_filt = prep.transform(d)
        w = prep.segment(d_filt, window_ms=WINDOW_MS, overlap=0.5)
        if w is not None:
             X_wins.append(w)
             # Label: mode of window
             # Re-calc stride parameters to match segment()
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

    X = np.concatenate(X_wins).astype(np.float32)
    y = np.concatenate(y_wins)
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y).astype(np.int64)
    n_classes = len(le.classes_)
    
    # --- Split ---
    print("Splitting data...")
    indices = np.arange(len(X))
    X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        X, y_enc, indices, test_size=0.3, stratify=y_enc, random_state=42
    )
    X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
        X_temp, y_temp, idx_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    # --- Feature Extraction (LightGBM) ---
    print("Extracting features (Parallel)...")
    fe = FeatureExtractor(fs=FS)
    
    # Helper for joblib
    def get_feats(arr): return np.concatenate([fe.extract_batch(batch) for batch in np.array_split(arr, max(1, len(arr)//2048))])
    
    # Process in chunks to avoid memory issues with joblib if dataset is huge, 
    # but here array_split inside joblib is better
    X_feats_train = np.concatenate(Parallel(n_jobs=-1)(delayed(fe.extract_batch)(b) for b in np.array_split(X_train, 20)))
    X_feats_val = np.concatenate(Parallel(n_jobs=-1)(delayed(fe.extract_batch)(b) for b in np.array_split(X_val, 5)))
    X_feats_test = np.concatenate(Parallel(n_jobs=-1)(delayed(fe.extract_batch)(b) for b in np.array_split(X_test, 5)))
    
    scaler_feat = StandardScaler()
    X_feats_train = scaler_feat.fit_transform(X_feats_train)
    X_feats_val = scaler_feat.transform(X_feats_val)
    X_feats_test = scaler_feat.transform(X_feats_test)

    # --- PyTorch Datasets ---
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)) # Dummy y for consistency
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    input_dim = X_train.shape[2] # Channels

    # --- Train Models ---
    print("\nTraining TCN...")
    tcn = train_model(TCN(input_dim, n_classes), train_loader, val_loader, epochs=EPOCHS)
    
    print("\nTraining SE-TCN...")
    setcn = train_model(SETCN(input_dim, n_classes), train_loader, val_loader, epochs=EPOCHS)
    
    print("\nTraining TCN+Attention...")
    tcn_attn = train_model(TCNAttention(input_dim, n_classes), train_loader, val_loader, epochs=EPOCHS)
    
    print("\nTraining Transformer...")
    tfm = train_model(TransformerModel(input_dim, n_classes), train_loader, val_loader, epochs=EPOCHS)

    # --- Train LightGBM ---
    print("\nTraining LightGBM...")
    lgbm = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, verbose=-1)
    lgbm.fit(X_feats_train, y_train)

    # --- Ensemble ---
    print("\nOptimizing Ensemble...")
    tcn_probs = predict_model(tcn, val_loader)
    setcn_probs = predict_model(setcn, val_loader)
    attn_probs = predict_model(tcn_attn, val_loader)
    tfm_probs = predict_model(tfm, val_loader)
    lgbm_probs = lgbm.predict_proba(X_feats_val)
    
    best_acc = 0
    best_w = (0.2, 0.2, 0.2, 0.2, 0.2)
    
    # Ensemble Search
    combinations_w = []
    for lgbm_w in [0.0, 0.1, 0.2]:
        remaining = 1.0 - lgbm_w
        for w1 in [0.1, 0.2, 0.3, 0.4]: # TCN
            for w2 in [0.1, 0.2, 0.3, 0.4]: # SE-TCN
                for w3 in [0.1, 0.2, 0.3, 0.4]: # TCN-Attn
                    w4 = remaining - w1 - w2 - w3 # Transformer
                    if w4 < 0 or abs(w1+w2+w3+w4+lgbm_w - 1.0) > 1e-4: continue
                    combinations_w.append((w1, w2, w3, w4, lgbm_w))
                    
    for w in combinations_w:
        ens_p = (w[0]*tcn_probs + w[1]*setcn_probs + w[2]*attn_probs + 
                 w[3]*tfm_probs + w[4]*lgbm_probs)
        acc = accuracy_score(y_val, ens_p.argmax(1))
        if acc > best_acc:
            best_acc = acc
            best_w = w
            
    print(f"Best Ensemble Acc: {best_acc:.4f}")
    print(f"Weights: TCN={best_w[0]}, SE-TCN={best_w[1]}, Attn={best_w[2]}, TFM={best_w[3]}, LGBM={best_w[4]}")
    
    # --- Final Test ---
    print("\n--- Final Evaluation on Test Set ---")
    tcn_test = predict_model(tcn, test_loader)
    setcn_test = predict_model(setcn, test_loader)
    attn_test = predict_model(tcn_attn, test_loader)
    tfm_test = predict_model(tfm, test_loader)
    lgbm_test = lgbm.predict_proba(X_feats_test)
    
    ens_test = (best_w[0]*tcn_test + best_w[1]*setcn_test + best_w[2]*attn_test + 
                best_w[3]*tfm_test + best_w[4]*lgbm_test)
                
    test_acc = accuracy_score(y_test, ens_test.argmax(1))
    test_f1 = f1_score(y_test, ens_test.argmax(1), average='macro')
    
    print(f"TEST SET ACCURACY: {test_acc:.4f}")
    print(f"TEST SET F1 SCORE: {test_f1:.4f}")

    # Save logic (Simplified)
    torch.save(tcn.state_dict(), f'{ARTIFACTS_DIR}/tcn.pth')
    torch.save(setcn.state_dict(), f'{ARTIFACTS_DIR}/setcn.pth')
    torch.save(tcn_attn.state_dict(), f'{ARTIFACTS_DIR}/tcn_attn.pth')
    torch.save(tfm.state_dict(), f'{ARTIFACTS_DIR}/tfm.pth')
    with open(f'{ARTIFACTS_DIR}/lgbm.pkl', 'wb') as f: pickle.dump(lgbm, f)
    print("Done.")

if __name__ == '__main__':
    main()
