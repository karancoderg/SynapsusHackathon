# Inference Guide

## Available Inference Scripts

### 1. `run_inference.ipynb` (Original - Jupyter Notebook)
**Tests on**: 50% of Session3 (test split used during training)
**Purpose**: Evaluate on held-out test set (proper evaluation)
**Expected Accuracy**: ~85%

```bash
# Run in Jupyter or convert to Python
jupyter nbconvert --to script run_inference.ipynb
python run_inference.py
```

### 2. `run_inference_full.py` (New - Python Script)
**Tests on**: ALL data (Session1 + Session2 + Session3)
**Purpose**: See model performance on entire dataset
**Expected Accuracy**: ~95%+ (will show overfitting on training data)

```bash
python run_inference_full.py
```

## Key Differences

| Script | Test Data | Purpose | Expected Acc |
|--------|-----------|---------|--------------|
| `run_inference.ipynb` | Session3 (50%) | Proper evaluation | ~85% |
| `run_inference_full.py` | All Sessions | Full dataset check | ~95%+ |

## Important Notes

### ⚠️ About Testing on All Data

When you test on **all data** (Session1+2+3), the accuracy will be **artificially high** because:

1. **Session1 + Session2** were used for **training**
   - The model has already seen this data
   - It will perform very well on it (overfitting indicator)

2. **Session3** is the true **test set**
   - The model has never seen this data
   - This is the real performance metric

### Example Results:

```
Testing on Session3 only (proper):
  Accuracy: 85.11% ✓ (Real performance)

Testing on All Data (Session1+2+3):
  Accuracy: 95%+ ✗ (Inflated due to training data)
```

## Recommended Usage

### For Evaluation (Proper)
Use the **original split** to evaluate on unseen data:
```bash
# Option 1: Use the notebook
jupyter notebook run_inference.ipynb

# Option 2: Convert and run
jupyter nbconvert --to script run_inference.ipynb
python run_inference.py
```

### For Debugging/Analysis
Use **all data** to check if model learned the training data:
```bash
python run_inference_full.py
```

If accuracy on all data is high (95%+) but Session3 is low (85%), it means:
- ✓ Model learned training data well
- ✗ Model doesn't generalize well to new data
- → Need better regularization or more diverse training data

## Output Files

### run_inference.ipynb
- `artifacts_final/ensemble_2model_matrix.png`

### run_inference_full.py
- `artifacts_final/ensemble_all_data_matrix.png`
- `artifacts_final/ensemble_all_data_results.txt`

## Quick Commands

```bash
# Proper evaluation (Session3 only)
jupyter nbconvert --to script run_inference.ipynb && python run_inference.py

# Full dataset check (All sessions)
python run_inference_full.py

# Compare results
cat artifacts_final/ensemble_all_data_results.txt
```

## Understanding the Results

### Good Model (Generalizes Well)
```
Session3 (test): 85%
All Data: 90%
Difference: 5% → Good generalization
```

### Overfitting Model
```
Session3 (test): 70%
All Data: 95%
Difference: 25% → Severe overfitting
```

### Your Current Model
```
Session3 (test): ~85%
All Data: ~95%+ (expected)
Difference: ~10% → Moderate overfitting (acceptable)
```

## Summary

- **Use `run_inference.ipynb`** for proper evaluation (85% on Session3)
- **Use `run_inference_full.py`** to check if model learned training data
- High accuracy on all data is expected (it includes training data)
- The real metric is **Session3 accuracy** (~85%)
