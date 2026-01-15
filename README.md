# Synapse Solution - sEMG Gesture Classification

## 🎯 Overview
Accuracy-optimized lightweight solution for the Synapse Neuro-Tech Challenge (PARSEC 6.0).

**Key Features:**
- 180+ handcrafted features (zero parameter cost)
- TCN + CNN + LightGBM ensemble (~60K neural params)
- Test-Time Augmentation (TTA) for +1-3% accuracy boost
- Subject-wise cross-validation

## 🚀 Quick Start (Google Colab)

1. Open `notebooks/synapse_complete.ipynb` in Google Colab
2. Run all cells - the notebook handles:
   - Dataset download from Google Drive
   - Preprocessing & feature extraction
   - Model training (TCN, CNN, LightGBM)
   - Ensemble prediction with TTA
   - Artifact export

## 📁 Project Structure

```
synapse-solution/
├── notebooks/
│   └── synapse_complete.ipynb    # Complete Colab notebook
├── src/
│   ├── preprocessing.py          # Signal filtering & normalization
│   ├── features.py               # 180+ feature extraction
│   ├── augmentation.py           # Data augmentation & TTA
│   └── models/
│       ├── tcn.py                # TCN model (~35K params)
│       ├── cnn.py                # CNN model (~25K params)
│       └── ensemble.py           # Ensemble wrapper
├── config/
│   └── config.yaml               # Hyperparameters
├── artifacts/                    # Saved models & scalers
├── scripts/
│   └── run_inference.py          # CLI inference
└── report/                       # LaTeX technical report
```

## 🔬 Technical Approach

### Signal Processing
- Bandpass filter (20-450 Hz) for noise removal
- Notch filter (50 Hz) for powerline interference
- Z-score normalization per channel

### Feature Engineering (180+ features)
| Category | Features | Per Channel |
|----------|----------|-------------|
| Time Domain | MAV, RMS, WL, ZC, SSC, VAR, IEMG, LOG | 8 |
| Frequency Domain | MNF, MDF, PKF, MNP, TTP, SM1 | 6 |
| Wavelet | Energy + Entropy (4 levels) | 8 |
| Hjorth | Activity, Mobility, Complexity | 3 |
| Cross-Channel | Correlation (28 pairs) | - |

### Models
| Model | Parameters | Expected F1 |
|-------|------------|-------------|
| TCN | ~35K | 0.91-0.93 |
| CNN | ~25K | 0.89-0.91 |
| LightGBM | 0 (trees) | 0.87-0.89 |
| **Ensemble** | **~60K** | **0.93-0.95** |

## 📊 Inference

```python
# Load trained models
import torch
import pickle

tcn = TCN()
tcn.load_state_dict(torch.load('artifacts/tcn_model.pth'))

with open('artifacts/lgbm_model.pkl', 'rb') as f:
    lgbm = pickle.load(f)

# Predict
predictions = predict_with_tta(X_windows, X_features, tcn, cnn, lgbm)
```

## 📝 Requirements

```
torch>=2.0
numpy
pandas
scipy
scikit-learn
lightgbm
pywavelets
```

## 👥 Team
Synapse Solution for PARSEC 6.0 @ IIT Dharwad
