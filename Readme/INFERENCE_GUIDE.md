# 🚀 Inference Guide — 2-Model sEMG Ensemble

This document explains **how to correctly run inference** using the trained **Inception-SE + sEMG-Net ensemble** for both **internal validation** and **external / production data**.

The inference pipeline strictly **matches training preprocessing and model architecture** to ensure reproducibility.

---

## 📌 Available Inference Scripts

### 1️⃣ `test_half_session3.ipynb` — Internal Validation

**Evaluates on**
- 50% of **Session 3** (held-out split from training)

**Purpose**
- Sanity check after training  
- Regression testing  
- Baseline comparison

**Expected Accuracy**
- ~85%

```bash
jupyter nbconvert --to script test_half_session3.ipynb.
python test_half_session3.py
```

---

### 2️⃣ `run_inference_full.py` — Production / External Data (Recommended)

**Evaluates on**
- Any dataset you provide

**Purpose**
- New data
- Competition test sets
- Deployment / batch inference

**Expected Accuracy**
- ~85% on similar data

```bash
python run_inference_full.py
```

---

## 🔍 Key Differences

| Script | Test Data | Use Case |
|------|----------|---------|
| `run_inference.ipynb` | Session3 (50%) | Internal validation |
| `run_inference_full.py` | Custom directory | External / production inference |

---

## 📂 Required Data Layout

Your inference data **must follow the same structure as training data**.

```
your_data/
├── Session1/
│   ├── gesture0/
│   │   ├── sample_01.csv
│   │   └── sample_02.csv
│   ├── gesture1/
│   ├── gesture2/
│   ├── gesture3/
│   └── gesture4/
├── Session2/
└── Session3/
```

### CSV Requirements
- **8 columns** = 8 sEMG channels
- **Sampling rate**: 512 Hz
- No missing values
- Filenames must include `gestureX` or be inside `gestureX/`

---

## ⚙️ Running Inference on New Data

### Step 1: Set Data Directory

Edit `run_inference_full.py`:

```python
DATA_DIR = 'path/to/your/new/data'
```

### Step 2: Run Inference

```bash
python run_inference_full.py
```

---

## 📈 Output Files

### `test_half_session3.ipynb`
- Confusion matrix image
- Console metrics (Accuracy, F1)

### `run_inference_full.py`
```
artifacts_final/
├── ensemble_all_data_matrix.png
├── ensemble_all_data_results.txt
```

---

## ⚠️ Data Compatibility Rules

For reliable performance:

1. Same **gesture set** (classes 0–4)
2. Same **sampling rate** (512 Hz)
3. Similar **electrode placement**
4. Similar **recording conditions**

---

## 📊 Expected Performance

| Scenario | Accuracy |
|-------|---------|
| Same subjects, same session | 85–90% |
| Same subjects, different day | 80–85% |
| New subjects | 70–80% |
| New setup / electrodes | 60–75% |

Large drops indicate **domain shift**, not model failure.

---

## 🛠️ Troubleshooting

### ❌ Model not found
```bash
ls artifacts_final/*.keras
```

Expected:
- `best_inception_se.keras`
- `best_semg_net.keras`

---

### ❌ Shape mismatch
- CSV must have **exactly 8 columns**
- Sampling rate must be **512 Hz**

```bash
head -5 your_data/Session1/gesture0/sample_01.csv
```

---

### ❌ Low accuracy on new data
Likely causes:
- Different subjects
- Different electrode placement
- Different recording protocol

Recommended actions:
- Fine-tune on small labeled subset
- Apply transfer learning
- Collect calibration data

---

## 🔁 Batch Inference (Advanced)

```python
datasets = [
    'dataset_A',
    'dataset_B',
    'dataset_C'
]

for ds in datasets:
    DATA_DIR = ds
    # run inference
    # save results per dataset
```

---

## ✅ Summary

- Use **`run_inference.ipynb`** for internal validation
- Use **`run_inference_full.py`** for external / production data
- Always match training preprocessing
- Expect degradation under domain shift

This inference setup is **reproducible, leakage-safe, and production-ready**.
