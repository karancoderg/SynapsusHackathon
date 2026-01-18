# Code Analysis: training_pipeline.ipynb

## Summary
This is a **2-Model Ensemble Pipeline** using TensorFlow/Keras for sEMG gesture classification, targeting >85% accuracy through class weighting.

## Architecture Overview

### Model 1: Inception-SE-TCN
- **Inception blocks**: Multi-scale feature extraction (kernel sizes 3 and 7)
- **Squeeze-Excitation**: Channel attention mechanism
- **TCN**: Temporal Convolutional Network with dilations [1, 2, 4, 8]
- **Multi-Head Attention**: 4 heads with key_dim=64
- **Regularization**: L2, Dropout (0.2, 0.3, 0.5), GaussianNoise (0.05)

### Model 2: sEMG-Net
- **Deep CNN**: 4 conv blocks (64→128→256→512 filters)
- **Progressive pooling**: MaxPooling after blocks 2, 3, 4
- **Kernel sizes**: 9→5→3→3 (coarse to fine)
- **Regularization**: L2, Dropout (0.25, 0.5), GaussianNoise (0.05)

### Ensemble Strategy
- **Simple averaging**: `(model1_probs + model2_probs) / 2`
- **No weighted ensemble**: Both models treated equally

## Key Features

### 1. Data Preprocessing
- **Bandpass filter**: 20-450 Hz (4th order Butterworth)
- **Notch filter**: 50 Hz (removes power line interference)
- **Normalization**: Channel-wise z-score (fitted on training data only)
- **Windowing**: 400ms windows, 160ms stride (60% overlap)

### 2. Data Augmentation (3x)
- **Channel masking**: 50% of samples, random channel zeroed
- **MixUp**: Beta distribution (α=0.2) blending
- **Gaussian noise**: σ=0.02

### 3. Class Weighting Strategy ⭐
```python
class_weights = {
    0: 1.0,
    1: 1.5,  # 50% more focus on Class 1
    2: 1.5,  # 50% more focus on Class 2
    3: 1.0,
    4: 1.0
}
```
**Purpose**: Address confusion between Class 1 and Class 2 (identified from confusion matrix)

### 4. Training Configuration
- **Loss**: Categorical Crossentropy with label smoothing (0.1)
- **Optimizer**: Adam with Cosine Decay Restarts
  - Initial LR: 0.001
  - First decay: 30% of total steps
  - t_mul: 2.0, m_mul: 0.9, alpha: 1e-5
- **Epochs**: 60
- **Batch size**: 128
- **Mixed precision**: FP16 (if GPU available)

### 5. Data Split
- **Train**: Session1 + Session2 (100%)
- **Val**: Session3 (50% of files)
- **Test**: Session3 (50% of files)
- **Split level**: File-level (prevents data leakage)

## Dependencies Analysis

### Required (Actually Used)
```
numpy>=1.24.0          # Array operations
pandas>=2.0.0          # CSV loading
scipy>=1.11.0          # Signal processing (butter, filtfilt, iirnotch, mode)
scikit-learn>=1.3.0    # LabelEncoder, metrics
tensorflow>=2.13.0     # Deep learning framework
gdown>=4.7.0           # Dataset download
```
## Expected Performance

### Current Setup
- **Target**: >85% accuracy
- **Strategy**: Class weighting to fix Class 1/2 confusion
- **Baseline**: 85.11% (mentioned in docstring)

## Conclusion

This is a **solid 85%+ pipeline** with good practices:
- Proper data split (no leakage)
- Strong regularization
- Class weighting for imbalanced classes
- Ensemble of complementary architectures

