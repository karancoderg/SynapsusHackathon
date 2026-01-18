# 🧪 Ensemble Test & Evaluation Script (sEMG)

This script evaluates the **final 2-model ensemble** consisting of:
- Inception-SE-TCN
- sEMG-Net

It **reproduces the exact test split**, loads trained weights, and performs **soft voting (50/50)** to report final metrics.

---

## 📂 Expected Directory Structure

```
project_root/
│
├── data/
│   ├── Subject_01/
│   │   ├── Session1/
│   │   │   └── gesture0_*.csv
│   │   ├── Session2/
│   │   │   └── gesture1_*.csv
│   │   └── Session3/
│   │       └── gesture2_*.csv
│   │
├── artifacts_final/
│   ├── best_inception_se.keras
│   ├── best_semg_net.keras
│
├── test_ensemble.py
```

⚠️ **Important**
- CSV filenames **must contain** `gestureX` where `X` is the class label.
- Each CSV must have **≥8 EMG channels (columns)**.

---

## 🔬 What This Script Does

1. Recreates **Session-3 test split** using the same random seed.
2. Fits the **SignalPreprocessor only on training data**.
3. Applies **identical windowing and filtering** as training.
4. Loads trained models from `artifacts_final/`.
5. Performs **soft voting ensemble inference**.
6. Reports:
   - Accuracy
   - Macro F1-score
   - Classification report
   - Confusion matrix (saved as image)

---

## ⚙️ Configuration Parameters

| Parameter | Value |
|---------|------|
| Sampling Rate | 512 Hz |
| Window Size | 400 ms |
| Stride | 160 ms |
| Batch Size | 128 |
| Ensemble | Equal-weight (50/50) |

---

## 🚀 How to Run

```bash
pip install numpy pandas scipy scikit-learn tensorflow seaborn matplotlib
python3 run_inference.py
```

---

## 📈 Output Files

```
artifacts_final/
├── ensemble_2model_matrix.png
```

---

## 🧠 Notes

- GaussianNoise layers are **inactive during inference**
- Class labels are inferred from training data
- This script must match training architecture exactly

---

## ✅ Expected Outcome

Typical performance:
- Accuracy: ~84–85%
- Macro F1: Stable across classes

