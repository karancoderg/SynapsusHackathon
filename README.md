# 🏆 Weighted 2-Model Ensemble Pipeline for sEMG Gesture Classification

This repository contains a **high-performance TensorFlow/Keras pipeline** for **sEMG gesture classification**, built around a **weighted 2-model ensemble**.  

---

## 📌 Key Highlights

- Two complementary deep models:
  - Inception-SE-TCN with Attention
  - Deep CNN (sEMG-Net)
- File-level session split (no data leakage)
- Advanced signal preprocessing
- 3× data augmentation
- Class-weighted loss to fix known confusion
- Ensemble averaging for stable generalization
- Mixed precision (FP16) support

---

## 🧠 Architecture Overview

### Inception-SE-TCN
- Inception blocks (kernel sizes 3 & 7)
- Temporal Convolutional Network with dilations [1, 2, 4, 8]
- Squeeze-Excitation blocks
- Multi-Head Self-Attention
- Heavy regularization

### sEMG-Net
- Deep CNN with 4 convolution blocks
- Kernel sizes: 9 → 5 → 3 → 3
- Progressive pooling
- Dense classification head

### Ensemble
Final prediction:
P = (P_model1 + P_model2) / 2

---

## 🔬 Data Processing

- Bandpass filter: 20–450 Hz
- Notch filter: 50 Hz
- Channel-wise z-score normalization
- Window size: 400 ms
- Stride: 160 ms

---

## 🔁 Data Augmentation (3×)

- Channel masking
- Gaussian noise (σ = 0.02)
- MixUp (α = 0.2)

---

## 🎯 Class Weighting

```python
class_weights = {
    0: 1.0,
    1: 1.5,
    2: 1.5,
    3: 1.0,
    4: 1.0
}
```

---

## ⚙️ Training Setup

- Optimizer: Adam
- LR Schedule: Cosine Decay Restarts
- Loss: Categorical Crossentropy (label smoothing = 0.1)
- Epochs: 60
- Batch size: 128

---

## 📂 Dataset Split

- Train: Session 1 + Session 2
- Validation: 50% of Session 3
- Test: 50% of Session 3

---

## 🚀 How to Run

```bash
pip install numpy pandas scipy scikit-learn tensorflow gdown
python3 training_pipeline.py
```

---

## 📁 Outputs

```text
artifacts_final/
├── best_inception_se.keras
├── best_semg_net.keras
```

---

## 📈 Metrics

- Accuracy
- Macro F1-score

---

## 📌 Summary

A robust, research-grade pipeline for sEMG gesture classification using deep learning, ensemble modeling, and class-aware optimization.
