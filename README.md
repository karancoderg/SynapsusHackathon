# Synapse Solution - sEMG Gesture Classification

## 🎯 Overview
High-accuracy deep learning solution for the Synapse Neuro-Tech Challenge (PARSEC 6.0).

**Achieved Performance: 85.11% Accuracy**

**Key Features:**
- 2-Model Deep Ensemble (Inception-SE-TCN + sEMG-Net)
- Class weighting strategy for imbalanced gestures
- Advanced data augmentation (Channel Masking + MixUp)
- Session-based split (no data leakage)
- Mixed precision training (FP16)

## 🚀 Quick Start

### Training
```bash
# Train the 2-model ensemble
python results/85/training_pipeline.ipynb

# Models will be saved to artifacts_final/
# - best_inception_se.keras
# - best_semg_net.keras
```

### Inference
```bash
# Evaluate on test set
python results/85/run_inference.ipynb

# Outputs:
# - Accuracy & F1 Score
# - Classification Report
# - Confusion Matrix (saved as PNG)
```

## 📁 Project Structure

```
synapse-solution/
├── results/
│   ├── 85/
│   │   ├── training_pipeline.ipynb    # Main training script
│   │   ├── run_inference.ipynb        # Evaluation script
│   │   └── CODE_ANALYSIS.md           # Detailed analysis
│   ├── 84.5/                          # Previous experiments
│   └── 76.21/                         # Earlier baselines
├── data/                              # Dataset (auto-downloaded)
├── artifacts_final/                   # Trained models
├── requirements.txt                   # Dependencies
└── README.md                          # This file
```

## 🔬 Technical Approach

### Signal Processing
- **Bandpass filter**: 20-450 Hz (4th order Butterworth)
- **Notch filter**: 50 Hz (removes power line interference)
- **Normalization**: Channel-wise z-score (fitted on training only)
- **Windowing**: 400ms windows, 160ms stride (60% overlap)

### Data Augmentation (3x)
1. **Channel Masking**: Random channel dropout + Gaussian noise
2. **MixUp**: Beta distribution blending (α=0.2)
3. **Original**: Unmodified data

### Model Architectures

#### Model 1: Inception-SE-TCN
- **Inception blocks**: Multi-scale feature extraction (kernels 3 & 7)
- **Squeeze-Excitation**: Channel attention mechanism
- **TCN**: Temporal convolutions with dilations [1, 2, 4, 8]
- **Multi-Head Attention**: 4 heads, key_dim=64
- **Parameters**: ~150K

#### Model 2: sEMG-Net
- **Deep CNN**: 4 conv blocks (64→128→256→512 filters)
- **Progressive pooling**: MaxPooling after blocks 2, 3, 4
- **Kernel sizes**: 9→5→3→3 (coarse to fine)
- **Parameters**: ~200K

#### Ensemble Strategy
- **Method**: Soft voting (simple averaging)
- **Formula**: `(model1_probs + model2_probs) / 2`
- **Class Weights**: 1.5x for classes 1 & 2 (addresses confusion)

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam with Cosine Decay Restarts |
| Initial LR | 0.001 |
| Loss | Categorical Crossentropy + Label Smoothing (0.1) |
| Regularization | L2 (1e-4), Dropout (0.2-0.5) |
| Batch Size | 128 |
| Epochs | 60 |
| Mixed Precision | FP16 (if GPU available) |

### Data Split
- **Train**: Session1 + Session2 (100%)
- **Val**: Session3 (50% of files)
- **Test**: Session3 (50% of files)
- **Split Level**: File-level (prevents data leakage)

## 📊 Results

### Performance Metrics
```
Accuracy: 85.11%
F1 Score: 0.84+
```

### Individual Model Performance
| Model | Accuracy |
|-------|----------|
| Inception-SE-TCN | ~84% |
| sEMG-Net | ~83% |
| **Ensemble** | **85.11%** |

## 💻 Inference Example

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load models
model1 = keras.models.load_model('artifacts_final/best_inception_se.keras')
model2 = keras.models.load_model('artifacts_final/best_semg_net.keras')

# Predict (X_test shape: [n_samples, 205, 8])
pred1 = model1.predict(X_test, batch_size=128)
pred2 = model2.predict(X_test, batch_size=128)

# Ensemble
ensemble_probs = (pred1 + pred2) / 2.0
predictions = ensemble_probs.argmax(axis=1)
```

## 📝 Requirements

```
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0
scikit-learn>=1.3.0
tensorflow>=2.13.0
gdown>=4.7.0
seaborn>=0.12.0
matplotlib>=3.7.0
```

Install with:
```bash
pip install -r requirements.txt
```

## 🔧 Hardware Requirements

- **Minimum**: CPU (training will be slow)
- **Recommended**: NVIDIA GPU with 4GB+ VRAM
- **Optimal**: NVIDIA T4/V100 (mixed precision enabled)


See `results/85/CODE_ANALYSIS.md` for detailed analysis.

