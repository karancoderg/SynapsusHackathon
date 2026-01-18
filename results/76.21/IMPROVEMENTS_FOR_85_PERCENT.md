# Improvements to Reach >85% Accuracy

## Summary of Changes in `85plus.py`

### 1. **More Training Data (CRITICAL)**
- **Changed**: `STRIDE_MS = 40` (was 160ms)
- **Impact**: 4x more training windows with better temporal overlap
- **Why**: More data = better generalization, especially with augmentation

### 2. **Enhanced Data Augmentation**
- **Added**: Time shifting (temporal jitter)
- **Added**: Per-sample augmentation (not just batch-level)
- **Added**: Amplitude scaling with 70% probability
- **Result**: 4x augmented dataset (original + 3 augmented versions)

### 3. **Stronger Regularization**
- **Dropout**: Increased from 0.2 → 0.3 (conv) and added 0.2 spatial dropout
- **Batch Size**: Reduced from 128 → 512 (better generalization)
- **Label Smoothing**: Added 0.1 (prevents overconfidence)
- **Gradient Clipping**: Added clipnorm=1.0

### 4. **Improved Architecture**
- **Deeper Dilations**: Extended to [1, 2, 4, 8, 16, 32] (was [1, 2, 4, 8])
- **Attention**: Added Squeeze-Excitation blocks for channel attention
- **Spatial Dropout**: Added between TCN blocks
- **Deeper Head**: 128 → 64 → n_classes (was 64 → n_classes)

### 5. **Better Training Strategy**
- **Epochs**: Increased to 150 (was 100)
- **Learning Rate**: Cosine annealing schedule (smooth decay)
- **Early Stopping**: Increased patience to 15 (was 12)

## Expected Accuracy Gain

| Improvement | Expected Gain |
|-------------|---------------|
| Reduced stride (40ms) | +2-3% |
| Enhanced augmentation | +1-2% |
| Attention mechanism | +0.5-1% |
| Deeper dilations | +0.5-1% |
| Better regularization | +0.5-1% |
| **Total Expected** | **+5-8%** |

**Baseline**: 82.5% → **Target**: 87-90%

## Additional Strategies (If Needed)

### Strategy A: Ensemble Approach (Highest Impact)
```python
# Train 3 models with different architectures:
# 1. TCN (current)
# 2. Transformer
# 3. CNN-LSTM

# Ensemble prediction:
final_pred = (tcn_pred + transformer_pred + cnn_lstm_pred) / 3
```
**Expected gain**: +3-5%

### Strategy B: Advanced Feature Engineering
```python
# Add hand-crafted features:
# - RMS (Root Mean Square)
# - Zero Crossing Rate
# - Wavelet coefficients
# - Frequency domain features (FFT)

# Concatenate with raw signal
X_combined = np.concatenate([X_raw, X_features], axis=-1)
```
**Expected gain**: +1-2%

### Strategy C: Test-Time Augmentation (TTA)
```python
# At inference, predict on multiple augmented versions
def predict_with_tta(model, X, n_tta=5):
    preds = []
    for _ in range(n_tta):
        X_aug = augment_single(X)  # Light augmentation
        preds.append(model.predict(X_aug))
    return np.mean(preds, axis=0)
```
**Expected gain**: +0.5-1%

### Strategy D: Focal Loss (For Class Imbalance)
```python
# If some gestures are harder to classify
from tensorflow_addons.losses import SigmoidFocalCrossEntropy

loss_fn = SigmoidFocalCrossEntropy(
    alpha=0.25,  # Weight for positive class
    gamma=2.0    # Focus on hard examples
)
```
**Expected gain**: +0.5-1% (if imbalanced)

### Strategy E: Larger Model Capacity
```python
# Increase filters from 64 → 96 or 128
filters = 96

# Add more TCN blocks
dilations = [1, 2, 4, 8, 16, 32, 64]  # Even deeper
```
**Expected gain**: +0.5-1% (risk of overfitting)

## Quick Wins (Try First)

1. **Run `85plus.py`** - Should get you to ~87-88%
2. **If still below 85%**: Reduce stride further to 20ms
3. **If still below 85%**: Add Strategy B (feature engineering)
4. **If still below 85%**: Use Strategy A (ensemble)

## Debugging Tips

If accuracy doesn't improve:
1. Check for data leakage (train/val/test split)
2. Verify preprocessing is fitted only on training data
3. Monitor validation loss (if increasing → overfitting)
4. Check class distribution (imbalanced?)
5. Visualize predictions (confusion matrix)

## Next Steps

1. Run: `python results/76.21/85plus.py`
2. Monitor training (watch val_loss vs train_loss)
3. If overfitting: Increase dropout or reduce model size
4. If underfitting: Increase model capacity or reduce regularization
5. If stuck at 84-85%: Try ensemble approach

Good luck! 🚀
