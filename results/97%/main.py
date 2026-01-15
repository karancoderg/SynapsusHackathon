#!/usr/bin/env python3
"""
Complete Training Pipeline for Synapse sEMG Challenge
Replicates the functionality of notebooks/synapse_complete.ipynb in a standalone script.
Fully self-contained with no external local dependencies.

Optimized for NVIDIA H200 GPU & High Accuracy:
- Batch Size: 4096 (Utilization of high VRAM)
- Mixed Precision (AMP): Faster training, lower memory
- Parallel Feature Extraction: CPU offloading
- Torch Compile: PyTorch 2.0+ acceleration
- CUDNN Benchmark: Optimized kernels
- Enhanced TCN Architecture: Larger kernels (5) and more filters for better context
- Optimized Ensemble Search: Automatically finds best model weights
"""
    
import os
import glob
import re
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
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

# --- Configuration ---
DATA_DIR = 'data'
ARTIFACTS_DIR = 'artifacts'
FS = 1000
# Optimized for H200 (Has ~141GB VRAM, can handle massive batches)
BATCH_SIZE = 4096
EPOCHS = 150 # Increased epochs for better convergence
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Enable CUDNN Benchmarking for optimized kernels
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# ==================== PREPROCESSING MODULE ====================

class SignalPreprocessor:
    """
    Preprocessing pipeline for 8-channel sEMG signals.
    Optimized for gesture recognition accuracy.
    """
    
    def __init__(
        self,
        fs: int = 1000,
        bandpass_low: float = 20.0,
        bandpass_high: float = 450.0,
        notch_freq: float = 50.0,
        filter_order: int = 4
    ):
        self.fs = fs
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.notch_freq = notch_freq
        self.filter_order = filter_order
        
        # Precompute filter coefficients
        self._init_filters()
        
        # Normalization parameters (fit during training)
        self.channel_means = None
        self.channel_stds = None
        self.fitted = False
    
    def _init_filters(self):
        """Initialize filter coefficients."""
        # Bandpass filter (20-450 Hz for sEMG)
        nyquist = self.fs / 2
        low = self.bandpass_low / nyquist
        high = self.bandpass_high / nyquist
        
        # Clamp to valid range
        low = max(0.001, min(low, 0.99))
        high = max(low + 0.01, min(high, 0.999))
        
        self.b_bp, self.a_bp = butter(
            self.filter_order, [low, high], btype='band'
        )
        
        # Notch filter (50/60 Hz powerline)
        if self.notch_freq > 0:
            self.b_notch, self.a_notch = iirnotch(
                self.notch_freq, 30.0, self.fs
            )
        else:
            self.b_notch, self.a_notch = None, None
    
    def bandpass_filter(self, signal: np.ndarray) -> np.ndarray:
        """Apply bandpass filter to remove noise."""
        if signal.shape[0] < 3 * self.filter_order:
            return signal
        return filtfilt(self.b_bp, self.a_bp, signal, axis=0)
    
    def notch_filter(self, signal: np.ndarray) -> np.ndarray:
        """Apply notch filter to remove powerline interference."""
        if self.b_notch is None or signal.shape[0] < 10:
            return signal
        return filtfilt(self.b_notch, self.a_notch, signal, axis=0)
    
    def rectify(self, signal: np.ndarray) -> np.ndarray:
        """Full-wave rectification."""
        return np.abs(signal)
    
    def smooth(self, signal: np.ndarray, window_ms: float = 50) -> np.ndarray:
        """Apply moving average smoothing."""
        window_samples = int(window_ms * self.fs / 1000)
        if window_samples < 2:
            return signal
        return uniform_filter1d(signal, window_samples, axis=0)
    
    def fit(self, signals: np.ndarray):
        """
        Fit normalization parameters on training data.
        
        Args:
            signals: Array of shape (n_samples, n_channels) or 
                     (n_windows, window_length, n_channels)
        """
        if signals.ndim == 3:
            # Flatten windows
            signals = signals.reshape(-1, signals.shape[-1])
        
        self.channel_means = np.mean(signals, axis=0)
        self.channel_stds = np.std(signals, axis=0) + 1e-8
        self.fitted = True
        return self
    
    def normalize(self, signal: np.ndarray) -> np.ndarray:
        """Z-score normalization per channel."""
        if not self.fitted:
            # Fit on current signal if not fitted
            mean = np.mean(signal, axis=0)
            std = np.std(signal, axis=0) + 1e-8
            return (signal - mean) / std
        return (signal - self.channel_means) / self.channel_stds
    
    def preprocess(
        self, 
        signal: np.ndarray,
        apply_rectification: bool = False,
        apply_smoothing: bool = False
    ) -> np.ndarray:
        """Complete preprocessing pipeline."""
        # 1. Bandpass filter
        filtered = self.bandpass_filter(signal)
        
        # 2. Notch filter
        filtered = self.notch_filter(filtered)
        
        # 3. Optional rectification
        if apply_rectification:
            filtered = self.rectify(filtered)
        
        # 4. Optional smoothing
        if apply_smoothing:
            filtered = self.smooth(filtered)
        
        # 5. Normalization
        normalized = self.normalize(filtered)
        
        return normalized
    
    def segment(
        self,
        signal: np.ndarray,
        window_ms: int = 200,
        overlap: float = 0.5
    ) -> np.ndarray:
        """Segment signal into overlapping windows."""
        window_samples = int(window_ms * self.fs / 1000)
        step_samples = int(window_samples * (1 - overlap))
        
        n_samples, n_channels = signal.shape
        n_windows = (n_samples - window_samples) // step_samples + 1
        
        if n_windows <= 0:
            return signal.reshape(1, -1, n_channels)
        
        windows = np.zeros((n_windows, window_samples, n_channels))
        
        for i in range(n_windows):
            start = i * step_samples
            end = start + window_samples
            windows[i] = signal[start:end]
        
        return windows
    
    def transform(self, x):
        """Compat method for pipeline."""
        # No additional transform needed if just filtering and normalizing, 
        # but matching the interface expected by main()
        # The main() calls transform() expecting filtering + normalization
        return self.preprocess(x)

# ==================== FEATURE EXTRACTION MODULE ====================

class FeatureExtractor:
    """
    Extract comprehensive features from sEMG signals.
    ~180 features total for maximum discriminative power.
    """
    
    def __init__(
        self,
        fs: int = 1000,
        wavelet: str = 'db4',
        wavelet_levels: int = 4,
        ar_order: int = 4,
        use_time_domain: bool = True,
        use_frequency_domain: bool = True,
        use_wavelet: bool = True,
        use_channel_correlation: bool = True,
        use_hjorth: bool = True,
        use_ar: bool = True
    ):
        self.fs = fs
        self.wavelet = wavelet
        self.wavelet_levels = wavelet_levels
        self.ar_order = ar_order
        
        # Feature flags
        self.use_time_domain = use_time_domain
        self.use_frequency_domain = use_frequency_domain
        self.use_wavelet = use_wavelet
        self.use_channel_correlation = use_channel_correlation
        self.use_hjorth = use_hjorth
        self.use_ar = use_ar
    
    # ==================== TIME DOMAIN FEATURES ====================
    
    def _mav(self, x: np.ndarray) -> float:
        return np.mean(np.abs(x))
    
    def _rms(self, x: np.ndarray) -> float:
        return np.sqrt(np.mean(x ** 2))
    
    def _wl(self, x: np.ndarray) -> float:
        return np.sum(np.abs(np.diff(x)))
    
    def _zc(self, x: np.ndarray) -> int:
        x = x - np.mean(x)
        signs = np.sign(x)
        signs[signs == 0] = 1
        return np.sum(np.abs(np.diff(signs)) == 2)
    
    def _ssc(self, x: np.ndarray, threshold: float = 0.01) -> int:
        diff1 = np.diff(x)
        return np.sum((diff1[:-1] * diff1[1:] < 0) & (np.abs(diff1[:-1] - diff1[1:]) > threshold))
    
    def _var(self, x: np.ndarray) -> float:
        return np.var(x)
    
    def _iemg(self, x: np.ndarray) -> float:
        return np.sum(np.abs(x))
    
    def _log(self, x: np.ndarray) -> float:
        return np.exp(np.mean(np.log(np.abs(x) + 1e-10)))
    
    def extract_time_domain(self, window: np.ndarray) -> np.ndarray:
        n_channels = window.shape[1]
        features = []
        for ch in range(n_channels):
            x = window[:, ch]
            features.extend([
                self._mav(x), self._rms(x), self._wl(x), self._zc(x),
                self._ssc(x), self._var(x), self._iemg(x), self._log(x)
            ])
        return np.array(features)
    
    # ==================== FREQUENCY DOMAIN FEATURES ====================
    
    def _compute_psd(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        freqs, psd = scipy_signal.welch(x, self.fs, nperseg=min(256, len(x)))
        return freqs, psd
    
    def _mnf(self, freqs: np.ndarray, psd: np.ndarray) -> float:
        return np.sum(freqs * psd) / (np.sum(psd) + 1e-10)
    
    def _mdf(self, freqs: np.ndarray, psd: np.ndarray) -> float:
        cumsum = np.cumsum(psd)
        return freqs[np.searchsorted(cumsum, cumsum[-1] / 2)]
    
    def _pkf(self, freqs: np.ndarray, psd: np.ndarray) -> float:
        return freqs[np.argmax(psd)]
    
    def _mnp(self, psd: np.ndarray) -> float:
        return np.mean(psd)
    
    def _ttp(self, psd: np.ndarray) -> float:
        return np.sum(psd)
    
    def _sm(self, freqs: np.ndarray, psd: np.ndarray, order: int = 1) -> float:
        return np.sum((freqs ** order) * psd)
    
    def extract_frequency_domain(self, window: np.ndarray) -> np.ndarray:
        n_channels = window.shape[1]
        features = []
        for ch in range(n_channels):
            x = window[:, ch]
            freqs, psd = self._compute_psd(x)
            features.extend([
                self._mnf(freqs, psd), self._mdf(freqs, psd), self._pkf(freqs, psd),
                self._mnp(psd), self._ttp(psd), self._sm(freqs, psd, 1)
            ])
        return np.array(features)
    
    # ==================== WAVELET FEATURES ====================
    
    def extract_wavelet(self, window: np.ndarray) -> np.ndarray:
        n_channels = window.shape[1]
        features = []
        for ch in range(n_channels):
            x = window[:, ch]
            coeffs = pywt.wavedec(x, self.wavelet, level=self.wavelet_levels)
            for level_coeffs in coeffs[:self.wavelet_levels]:
                energy = np.sum(level_coeffs ** 2)
                features.append(energy)
                level_coeffs_norm = level_coeffs ** 2
                level_coeffs_norm = level_coeffs_norm / (np.sum(level_coeffs_norm) + 1e-10)
                ent = entropy(level_coeffs_norm + 1e-10)
                features.append(ent)
        return np.array(features)
    
    # ==================== CHANNEL CORRELATION FEATURES ====================
    
    def extract_channel_correlation(self, window: np.ndarray) -> np.ndarray:
        n_channels = window.shape[1]
        features = []
        for i, j in combinations(range(n_channels), 2):
            corr = np.corrcoef(window[:, i], window[:, j])[0, 1]
            features.append(corr if not np.isnan(corr) else 0)
        return np.array(features)
    
    # ==================== HJORTH PARAMETERS ====================
    
    def extract_hjorth(self, window: np.ndarray) -> np.ndarray:
        n_channels = window.shape[1]
        features = []
        for ch in range(n_channels):
            x = window[:, ch]
            activity = np.var(x)
            dx = np.diff(x)
            mobility = np.sqrt(np.var(dx) / (activity + 1e-10))
            ddx = np.diff(dx)
            mobility_dx = np.sqrt(np.var(ddx) / (np.var(dx) + 1e-10))
            complexity = mobility_dx / (mobility + 1e-10)
            features.extend([activity, mobility, complexity])
        return np.array(features)
    
    # ==================== AR COEFFICIENTS ====================
    
    def extract_ar_coefficients(self, window: np.ndarray) -> np.ndarray:
        n_channels = window.shape[1]
        features = []
        for ch in range(n_channels):
            x = window[:, ch]
            try:
                # Use simple AR extraction or skip if too complex for now
                # Using simple autocorrelation approximation for robustness
                r = np.correlate(x, x, mode='full')[len(x)-1:]
                features.extend(r[1:self.ar_order+1]) # Simple stats instead of solving Yule-Walker
            except:
                features.extend([0] * self.ar_order)
        return np.array(features)
    
    def extract(self, window: np.ndarray) -> np.ndarray:
        features = []
        if self.use_time_domain: features.append(self.extract_time_domain(window))
        if self.use_frequency_domain: features.append(self.extract_frequency_domain(window))
        if self.use_wavelet: features.append(self.extract_wavelet(window))
        if self.use_channel_correlation: features.append(self.extract_channel_correlation(window))
        if self.use_hjorth: features.append(self.extract_hjorth(window))
        if self.use_ar: features.append(self.extract_ar_coefficients(window))
        return np.concatenate(features)
    
    def extract_batch(self, windows: np.ndarray) -> np.ndarray:
        # Optimized for parallelism if called on smaller batches
        n_windows = windows.shape[0]
        first_features = self.extract(windows[0])
        n_features = len(first_features)
        features = np.zeros((n_windows, n_features))
        features[0] = first_features
        
        # Internal loop, but we will parallelize the calls to this function
        for i in range(1, n_windows):
            features[i] = self.extract(windows[i])
        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

# ==================== AUGMENTATION MODULE ====================

class EMGAugmenter:
    """Data augmentation for sEMG signals."""
    
    def __init__(self, gaussian_noise_std=0.01, time_warp_range=0.15, 
                 magnitude_scale_range=(0.8, 1.2), channel_dropout_prob=0.1, cutout_prob=0.2):
        self.gaussian_noise_std = gaussian_noise_std
        self.time_warp_range = time_warp_range
        self.magnitude_scale_range = magnitude_scale_range
        self.channel_dropout_prob = channel_dropout_prob
        self.cutout_prob = cutout_prob
    
    def add_gaussian_noise(self, x):
        return x + np.random.randn(*x.shape) * self.gaussian_noise_std * np.std(x)
    
    def time_warp(self, x):
        warp_factor = 1 + np.random.uniform(-self.time_warp_range, self.time_warp_range)
        n_samples = x.shape[0]
        new_length = int(n_samples * warp_factor)
        if new_length == n_samples: return x
        
        old_indices = np.arange(n_samples)
        new_indices = np.linspace(0, n_samples - 1, new_length)
        warped = np.zeros((new_length, x.shape[1]))
        for ch in range(x.shape[1]):
            warped[:, ch] = np.interp(new_indices, old_indices, x[:, ch])
            
        resampled = np.zeros_like(x)
        resample_indices = np.linspace(0, new_length - 1, n_samples)
        for ch in range(x.shape[1]):
            resampled[:, ch] = np.interp(resample_indices, np.arange(new_length), warped[:, ch])
        return resampled
    
    def augment(self, x, p=0.5):
        aug = x.copy()
        if np.random.rand() < p: aug = self.add_gaussian_noise(aug)
        if np.random.rand() < p: aug = self.time_warp(aug)
        return aug
    
    def augment_batch(self, X, y, n_aug=2):
        X_aug_list, y_aug_list = [X], [y]
        for _ in range(n_aug):
            batch_aug = np.array([self.augment(x) for x in X])
            X_aug_list.append(batch_aug)
            y_aug_list.append(y)
        return np.concatenate(X_aug_list), np.concatenate(y_aug_list)

# ==================== MODELS MODULES ====================

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)
    
    def forward(self, x):
        out = self.conv(x)
        if self.padding > 0: out = out[:, :, :-self.padding]
        return out

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        # Activation fusions handled by torch.compile in main loop
        out = self.dropout(F.relu(self.bn1(self.conv1(x))))
        out = self.dropout(F.relu(self.bn2(self.conv2(out))))
        return F.relu(out + self.residual(x))

class TCN(nn.Module):
    # Tuned hyperparameters: larger kernel, dropout 0.2, deeper filters
    def __init__(self, input_channels=8, num_classes=5, n_filters=[64, 64, 128, 128], kernel_size=5, dropout=0.2):
        super().__init__()
        self.in_ch = input_channels
        layers = []
        in_ch = input_channels
        dilations = [1, 2, 4, 8]
        for i, out_ch in enumerate(n_filters):
            d = dilations[min(i, len(dilations)-1)]
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, d, dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(n_filters[-1], 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, num_classes))
    
    def forward(self, x):
        if x.dim() == 3 and x.shape[-1] == self.in_ch: x = x.transpose(1, 2)
        x = self.tcn(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pool_size=2, dropout=0.3):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(pool_size) if pool_size > 1 else nn.Identity()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.pool(F.relu(self.bn(self.conv(x)))))

class LightweightCNN(nn.Module):
    def __init__(self, input_channels=8, num_classes=5, n_filters=[64, 128, 64], kernel_sizes=[5, 3, 3], dropout=0.3):
        super().__init__()
        self.in_ch = input_channels
        layers = []
        in_ch = input_channels
        for i, (out_ch, ks) in enumerate(zip(n_filters, kernel_sizes)):
            pool = 2 if i < len(n_filters) - 1 else 1
            layers.append(ConvBlock(in_ch, out_ch, ks, pool, dropout))
            in_ch = out_ch
        self.conv_layers = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(n_filters[-1], 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, num_classes))
    
    def forward(self, x):
        if x.dim() == 3 and x.shape[-1] == self.in_ch: x = x.transpose(1, 2)
        out = self.conv_layers(x)
        out = self.gap(out).squeeze(-1)
        return self.fc(out)

# ==================== MAIN PIPELINE ====================

def download_dataset():
    """Download dataset if not present."""
    if os.path.exists(DATA_DIR) and len(glob.glob(f'{DATA_DIR}/**/*.csv', recursive=True)) > 0:
        print('Dataset found locally.')
        return

    print('Dataset not found. Downloading...')
    try:
        import gdown
    except ImportError:
        print('Installing gdown...')
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown

    url = 'https://drive.google.com/uc?id=16iNEwhThf2LcX7rOOVM03MTZiwq7G51x'
    output = 'dataset.zip'
    gdown.download(url, output, quiet=False)
    
    print('Unzipping...')
    import zipfile
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    
    # Clean up
    if os.path.exists(output):
        os.remove(output)
    print('Download complete.')

def load_dataset(data_dir=DATA_DIR):
    """Load all subject CSVs and combine."""
    print(f"Loading data from {data_dir}...")
    all_data = []
    all_labels = []
    all_subjects = []
    
    csv_files = glob.glob(f'{data_dir}/**/*.csv', recursive=True)
    print(f'Found {len(csv_files)} CSV files')
    
    if len(csv_files) == 0:
        print("No CSV files found! Please ensure dataset is in 'data/' folder.")
        return [], [], []

    for i, csv_file in enumerate(sorted(csv_files)):
        try:
            df = pd.read_csv(csv_file)
            
            # 1. Extract label from filename (e.g., "gesture01" -> 1)
            filename = os.path.basename(csv_file)
            match = re.search(r'gesture(\d+)', filename)
            if match:
                label = int(match.group(1))
            else:
                continue
                
            # 2. Extract subject ID
            subj_match = re.search(r'subject_(\d+)', csv_file)
            if subj_match:
                subject_id = int(subj_match.group(1))
            else:
                subject_id = i // 40
            
            # 3. Data is all columns
            data = df.values
            if data.shape[1] < 8: # Basic validation
                continue
                
            # 4. Create labels array
            labels = np.full(len(data), label)
            
            all_data.append(data)
            all_labels.append(labels)
            all_subjects.extend([subject_id] * len(labels))
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} files...")
                
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue
    
    return all_data, all_labels, np.array(all_subjects)

class EMGDataset(Dataset):
    def __init__(self, X, y, augmenter=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.aug = augmenter
    
    def __len__(self): return len(self.X)
    
    def __getitem__(self, idx):
        x, y = self.X[idx].numpy(), self.y[idx]
        if self.aug: x = self.aug(x)
        return torch.FloatTensor(x), y

def train_epoch(model, loader, opt, criterion, scaler):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(set_to_none=True) # Optimized zero_grad
        
        # Mixed Precision Forward
        with autocast():
            out = model(X)
            loss = criterion(out, y)
        
        # Scaled Backward
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        
        total_loss += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        total += len(y)
    return total_loss/total, correct/total

@torch.no_grad()
def eval_model(model, loader):
    model.eval()
    preds, labels = [], []
    for X, y in loader:
        X = X.to(DEVICE)
        # Inference autocast also good practice, though usually FP32/BF16
        with autocast():
            out = model(X)
        preds.extend(out.argmax(1).cpu().numpy())
        labels.extend(y.numpy())
    return np.array(preds), np.array(labels)

def train_network(model, train_loader, val_loader, epochs=EPOCHS, lr=1e-3, patience=20):
    model.to(DEVICE)
    
    # Compile model for optimizations (H200 supports this extensively)
    if hasattr(torch, 'compile'):
        print("Compiling model for acceleration...")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"Could not compile model: {e}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Initialize Scaler for AMP
    scaler = GradScaler()
    
    best_f1, wait, best_state = 0, 0, None
    for ep in range(epochs):
        loss, acc = train_epoch(model, train_loader, opt, criterion, scaler)
        preds, labels = eval_model(model, val_loader)
        f1 = f1_score(labels, preds, average='macro')
        
        if f1 > best_f1:
            best_f1, wait = f1, 0
            # If using compile, best state dict handles it
            state_dict = model.state_dict()
            # Handle compiled model prefix involved with torch.compile
            best_state = {k.replace("_orig_mod.", ""): v.cpu().clone() for k, v in state_dict.items()}
        else:
            wait += 1
        
        if (ep+1) % 5 == 0:
            print(f'Ep {ep+1}: Loss={loss:.4f}, Acc={acc:.4f}, Val F1={f1:.4f}')
        
        if wait >= patience:
            print(f'Early stop at epoch {ep+1}')
            break
        sched.step()
    
    # Re-load clean state
    # Create new instance to load state dict cleanly if torch.compile messed with keys
    if best_state:
        # We need to load into the uncompiled model for saving artifacts cleanly
        # If model is wrapped, we might need to unwrap or just ignore strict keys if prefix mismatch
        try:
            model.load_state_dict(best_state)
        except:
             # Fallback if keys don't match exactly due to compile
            model.load_state_dict(best_state, strict=False)
            
    return model, best_f1

def feature_extraction_wrapper(batch, extractor):
    """Helper for parallel execution."""
    return extractor.extract_batch(batch)

def optimize_ensemble_weights(tcn_probs, cnn_probs, lgbm_probs, y_true):
    """Grid search for best ensemble weights."""
    print("Optimizing ensemble weights...")
    best_acc = 0
    best_f1 = 0
    best_weights = (0.4, 0.3, 0.3)
    
    # Grid search
    steps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    for w1 in steps:
        for w2 in steps:
            w3 = 1.0 - w1 - w2
            if w3 < 0 or w3 > 1.0: continue
            
            # Simple check to avoid crazy floats
            if abs(w1+w2+w3 - 1.0) > 1e-5: continue
            
            ens_probs = w1*tcn_probs + w2*cnn_probs + w3*lgbm_probs
            ens_preds = ens_probs.argmax(1)
            
            f1 = f1_score(y_true, ens_preds, average='macro')
            if f1 > best_f1:
                best_f1 = f1
                best_acc = accuracy_score(y_true, ens_preds)
                best_weights = (w1, w2, w3)
                
    print(f"Best Weights Found: TCN={best_weights[0]:.1f}, CNN={best_weights[1]:.1f}, LGBM={best_weights[2]:.1f}")
    return best_weights

def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    # 0. Download Data
    download_dataset()
    
    # 1. Load Data
    raw_data, raw_labels, subjects = load_dataset()
    if not raw_data:
        return

    # 2. Preprocessing & Segmentation
    print("Preprocessing data...")
    preprocessor = SignalPreprocessor(fs=FS)
    # Fit on first subject to initialize scaler
    if len(raw_data) > 0:
        preprocessor.fit(raw_data[0])
    
    all_windows = []
    all_labels_flat = []
    
    print("Segmenting data...")
    for data, labels in zip(raw_data, raw_labels):
        # Filter
        data_filt = preprocessor.transform(data)
        
        # Segment
        win_len = 200
        overlap = 0.5
        step = int(win_len * (1 - overlap))
        
        n_samples = len(data_filt)
        if n_samples < win_len: continue
        
        n_windows = (n_samples - win_len) // step + 1
        if n_windows <= 0: continue
        
        # Sliding window
        idx = np.arange(win_len)[None, :] + np.arange(n_windows)[:, None] * step
        windows = data_filt[idx]
        
        # Labels
        label_windows = labels[idx]
        win_labels = mode(label_windows, axis=1, keepdims=True)[0].flatten()
        
        all_windows.append(windows)
        all_labels_flat.append(win_labels)

    X_windows = np.concatenate(all_windows)
    y = np.concatenate(all_labels_flat)
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    n_classes = len(le.classes_)
    
    print(f"Total windows: {len(X_windows)}")
    print(f"Classes: {le.classes_}")

    # 3. Feature Extraction
    print("Extracting features (Parallelized for efficiency)...")
    feature_extractor = FeatureExtractor(fs=FS)
    
    # Parallel processing
    n_jobs = -1 # Use all cores
    batch_sz = 2000 # Larger chunks for parallelism
    
    # Create chunks indices
    indices = range(0, len(X_windows), batch_sz)
    
    # Run parallel extraction
    # We use loky backend which is safer for numpy
    X_features_list = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(feature_extraction_wrapper)(X_windows[i:i+batch_sz], feature_extractor) 
        for i in indices
    )
        
    X_features = np.concatenate(X_features_list)
    print(f"Extracted features shape: {X_features.shape}")
    
    # Feature Selection
    print("Selecting top features...")
    # Subsample for speed
    idx_sub = np.random.choice(len(X_features), min(10000, len(X_features)), replace=False)
    mi = mutual_info_classif(X_features[idx_sub], y[idx_sub], random_state=42)
    top_k = 80
    top_idx = np.argsort(mi)[-top_k:]
    X_features_sel = X_features[:, top_idx]
    
    feat_scaler = StandardScaler()
    X_features_sel = feat_scaler.fit_transform(X_features_sel)

    # 4. Split and Augment
    X_train, X_val, y_train, y_val, feat_train, feat_val = train_test_split(
        X_windows, y, X_features_sel, test_size=0.2, stratify=y, random_state=42
    )
    
    augmenter = EMGAugmenter()
    # Augment training set
    print("Augmenting training data...")
    X_train_aug, y_train_aug = augmenter.augment_batch(X_train, y_train, n_aug=2)
    
    # DataLoaders optimization
    # num_workers=4 is usually safe, can go higher on big CPUs
    # pin_memory=True speeds up host-to-device transfer
    train_loader = DataLoader(
        EMGDataset(X_train_aug, y_train_aug), 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=os.cpu_count() or 4,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        EMGDataset(X_val, y_val), 
        batch_size=BATCH_SIZE,
        num_workers=os.cpu_count() or 4,
        pin_memory=True,
        persistent_workers=True
    )

    # 5. Train Models
    # TCN
    print("\nTraining TCN...")
    # High capacity TCN init
    tcn = TCN(input_channels=8, num_classes=n_classes, 
              n_filters=[64, 64, 128, 128], kernel_size=5, dropout=0.2)
    tcn, tcn_f1 = train_network(tcn, train_loader, val_loader)
    print(f"TCN Best F1: {tcn_f1:.4f}")
    
    # CNN
    print("\nTraining CNN...")
    cnn = LightweightCNN(input_channels=8, num_classes=n_classes)
    cnn, cnn_f1 = train_network(cnn, train_loader, val_loader)
    print(f"CNN Best F1: {cnn_f1:.4f}")
    
    # LightGBM
    print("\nTraining LightGBM...")
    # Attempt GPU training for LightGBM if possible, else CPU
    lgbm_params = {
        'n_estimators': 500,
        'num_leaves': 31,
        'learning_rate': 0.05,
        'verbose': -1,
        'device': 'gpu' # Try GPU
    }
    
    lgbm = lgb.LGBMClassifier(**lgbm_params)
    try:
        lgbm.fit(feat_train, y_train)
    except Exception as e:
        print(f"LightGBM GPU failed ({e}), falling back to CPU...")
        lgbm_params['device'] = 'cpu'
        lgbm = lgb.LGBMClassifier(**lgbm_params)
        lgbm.fit(feat_train, y_train)
        
    lgbm_preds = lgbm.predict(feat_val)
    lgbm_f1 = f1_score(y_val, lgbm_preds, average='macro')
    print(f"LightGBM F1: {lgbm_f1:.4f}")

    # 6. Ensemble
    print("\nEvaluating Ensemble...")
    tcn.eval()
    cnn.eval()
    with torch.no_grad():
        X_val_t = torch.FloatTensor(X_val).to(DEVICE)
        # Process in batches for simple inference validation to avoid OOM on massive valid sets
        # though H200 unlikely to OOM here
        tcn_probs = []
        cnn_probs = []
        
        inf_batch = 4096
        for i in range(0, len(X_val_t), inf_batch):
            batch = X_val_t[i:i+inf_batch]
            with autocast():
                tcn_probs.append(F.softmax(tcn(batch), dim=1).cpu().numpy())
                cnn_probs.append(F.softmax(cnn(batch), dim=1).cpu().numpy())
                
        tcn_probs = np.concatenate(tcn_probs)
        cnn_probs = np.concatenate(cnn_probs)
    
    lgbm_probs = lgbm.predict_proba(feat_val)
    
    # Optimize weights
    w1, w2, w3 = optimize_ensemble_weights(tcn_probs, cnn_probs, lgbm_probs, y_val)
    
    ensemble_probs = w1 * tcn_probs + w2 * cnn_probs + w3 * lgbm_probs
    ensemble_preds = ensemble_probs.argmax(1)
    
    final_acc = accuracy_score(y_val, ensemble_preds)
    final_f1 = f1_score(y_val, ensemble_preds, average='macro')
    
    print(f"Ensemble Accuracy: {final_acc:.4f}")
    print(f"Ensemble F1: {final_f1:.4f}")
    
    # 7. Save Artifacts
    print("\nSaving artifacts...")
    # Strip compile wrapper if present
    tcn_sd = tcn.state_dict()
    cnn_sd = cnn.state_dict()
    
    # Clean keys if compiled
    tcn_sd = {k.replace("_orig_mod.", ""): v for k, v in tcn_sd.items()}
    cnn_sd = {k.replace("_orig_mod.", ""): v for k, v in cnn_sd.items()}
    
    torch.save(tcn_sd, f'{ARTIFACTS_DIR}/tcn_model.pth')
    torch.save(cnn_sd, f'{ARTIFACTS_DIR}/cnn_model.pth')
    with open(f'{ARTIFACTS_DIR}/lgbm_model.pkl', 'wb') as f:
        pickle.dump(lgbm, f)
        
    meta_artifacts = {
        'label_encoder': le,
        'feature_scaler': feat_scaler,
        'top_features_idx': top_idx,
        'n_channels': 8,
        'n_classes': n_classes,
        'ensemble_weights': (w1, w2, w3) # Save optimized weights
    }
    with open(f'{ARTIFACTS_DIR}/preprocessing.pkl', 'wb') as f:
        pickle.dump(meta_artifacts, f)
        
    print("Done! Artifacts saved to 'artifacts/'.")

if __name__ == '__main__':
    main()
