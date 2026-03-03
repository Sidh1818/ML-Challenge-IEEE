# ML Challenge — Binary Fault Detection
### IEEE SB, GEHU | Online Qualifiers

## Problem Statement
Predict whether a device is operating **normally (Class 0)** or experiencing a **fault condition (Class 1)** based on 47 numerical sensor readings (F01–F47).

## Approach

### 1. Data Analysis (`datacheck.ipynb`)
- 43,776 samples, 47 features, binary target (60/40 class balance)
- No missing values; 738 duplicate rows identified
- 44/47 features are highly skewed; 10 feature pairs with |r| > 0.9 (mirror pattern)
- Top predictors: F01, F09, F29, F19, F21

### 2. Preprocessing (`preprocessing.ipynb`)
- Removed 738 duplicate rows
- Outlier capping at 1st/99th percentile (Winsorization)
- Yeo-Johnson power transform (reduced skewed features from 40 → 0)
- Removed 11 redundant features (47 → 36)
- StandardScaler normalization
- Same pipeline applied to test data

### 3. Feature Engineering & Modeling (`improved_model.py`)
Engineered **47 new features** (36 → 83 total):

| Category | Count | Rationale |
|----------|-------|-----------|
| Interaction features | 8 | Capture non-linear relationships between top predictors |
| Ratio features | 3 | Capture relative sensor magnitudes |
| Row-wise aggregates | 11 | Holistic device health metrics (mean, std, skew, kurtosis, etc.) |
| Sensor group aggregates | 11 | Exploit known sensor structure (early/mid/high/late groups) |
| Polynomial (squared) | 6 | Quadratic relationships for top predictors |
| Absolute values | 6 | Magnitude-based fault indicators |
| Cross-group interactions | 2 | Inter-system relationships |

**Models evaluated** (5-fold stratified CV):

| Model | CV F1-Score | CV Accuracy | CV ROC-AUC |
|-------|-----------|------------|-----------|
| **LightGBM** | **0.9848** | **0.9878** | **0.9992** |
| XGBoost | 0.9832 | 0.9866 | 0.9990 |
| ExtraTrees | 0.9778 | 0.9824 | 0.9990 |

**Final model: LightGBM** — best F1 with fast training (14.5s on full data).

## Files

| File | Description |
|------|-------------|
| `improved_model.py` | **Main pipeline** — feature engineering + model training + prediction |
| `preprocessing.ipynb` | Data preprocessing pipeline |
| `datacheck.ipynb` | Exploratory data analysis |
| `TRAIN.csv` | Raw training data |
| `TEST.csv` | Raw test data |
| `TRAIN_PREPROCESSED.csv` | Preprocessed training data |
| `TEST_PREPROCESSED.csv` | Preprocessed test data |
| `FINAL.csv` | **Submission file** — predictions (ID → CLASS) |

## How to Run

```bash
pip install numpy pandas scikit-learn xgboost lightgbm
python improved_model.py
```

This will:
1. Load preprocessed data
2. Engineer 47 new features
3. Cross-validate XGBoost, LightGBM, ExtraTrees
4. Train the best model on full data
5. Generate `FINAL.csv`

## Requirements
- Python 3.8+
- numpy, pandas, scikit-learn, xgboost, lightgbm
