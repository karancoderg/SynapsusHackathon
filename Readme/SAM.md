# 🧱 System Architecture Map (SAM)

## High-Level Flow

```
Raw CSV Files
   ↓
Signal Preprocessing
   ├─ Bandpass Filter (20–450 Hz)
   ├─ Notch Filter (50 Hz)
   └─ Channel-wise Z-score
   ↓
Windowing
   ├─ 400 ms windows
   └─ 160 ms stride
   ↓
Deep Models (Parallel)
   ├─ Inception-SE-TCN
   └─ sEMG-Net
   ↓
Soft Voting Ensemble
   ↓
Final Prediction
```

---

## Component Responsibilities

### SignalPreprocessor
- Ensures train-only normalization
- Prevents data leakage
- Matches training preprocessing exactly

### Inception-SE-TCN
- Multi-scale temporal modeling
- Channel attention via SE blocks
- Long-range context via TCN + Attention

### sEMG-Net
- Hierarchical CNN feature extraction
- Robust to amplitude variations

### Ensemble Layer
- Equal probability averaging
- Reduces variance
- Improves generalization

---

## Design Constraints

- Models **must not be retrained**
- Preprocessing **must match training**
- File naming **must encode class labels**
- Session split **must remain unchanged**

---

## Failure Modes

| Issue | Cause |
|-----|------|
| Wrong accuracy | Mismatched preprocessing |
| Label errors | Incorrect filename format |
| Model load failure | Missing `.keras` files |
| Shape mismatch | Different window config |

---

## Summary

This architecture prioritizes **reproducibility, fairness, and evaluation integrity**.
