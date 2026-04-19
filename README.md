# Binary Fault Detection - ML Challenge
**Team Tech Ninjas | IEEE SB GEHU**

## Overview

Binary classification to predict device status (Normal/Fault) from 47 sensor readings. We used preprocessing + feature engineering + ensemble models, with **LightGBM** giving the best results.

**Results:**
- Accuracy: 98.78%
- F1-Score: 98.48%
- ROC-AUC: 99.92%

## Dataset

- Training: 43,776 samples × 47 features (738 duplicates removed → 43,038)
- Test: 10,944 samples
- Classes: Normal (60%) / Fault (40%)
- No missing values

## Pipeline

```
Raw Data (43,776)
  → Remove duplicates (43,038)
  → Outlier capping (1st-99th percentile)
  → Yeo-Johnson transform (fixed 44 skewed features)
  → Drop correlated features (47 → 36, threshold |r| > 0.9)
  → StandardScaler
  → Feature Engineering (36 → 83 features)
  → Model Training (5-fold Stratified CV)
  → Predictions
```

## Feature Engineering

We created 47 new features from the preprocessed 36:

| Type | Count | Details |
|------|-------|---------|
| Interactions | 8 | Pairs like F05×F09, F06×F07, F19×F21 |
| Ratios | 3 | F19/F09, F21/F09, F05/F06 |
| Row-wise stats | 11 | mean, std, min, max, range, median, skew, kurtosis, etc. |
| Group stats | 11 | Stats for early/middle/high/late sensor groups |
| Polynomial | 6 | Squared values of top features (F05², F09², etc.) |
| Absolute | 6 | |F05|, |F06|, |F07|, |F09|, |F19|, |F21| |
| Cross-group | 2 | early×middle, early×high interactions |

Top predictors from our analysis: **F05, F06, F07, F09, F19, F21**

## Models Compared

| Model | F1-Score | Accuracy | ROC-AUC |
|-------|----------|----------|---------|
| **LightGBM** | **0.9848** | **0.9878** | **0.9992** |
| XGBoost | 0.9832 | 0.9866 | 0.9990 |
| ExtraTrees | 0.9778 | 0.9824 | 0.9990 |

LightGBM config: n_estimators=800, learning_rate=0.05, num_leaves=100, colsample_bytree=0.7

## Files

```
datacheck.ipynb          - EDA and data exploration
preprocessing.ipynb      - Data cleaning pipeline
improved_model.py        - Feature engineering + model training
TRAIN.csv / TEST.csv     - Raw data
TRAIN_PREPROCESSED.csv   - Cleaned training data
TEST_PREPROCESSED.csv    - Cleaned test data
FINAL.csv                - Final predictions
requirements.txt         - Python dependencies
```

## How to Run

```bash
pip install -r requirements.txt

# run notebooks first for preprocessing, then:
python improved_model.py
```

Output: `FINAL.csv` with columns [ID, CLASS]

## Team
**Team Tech Ninjas**
Graphic Era Hill University, Dehradun
B.Tech CSE (AI & ML), 3rd Year

- **Rajat Pundir** - Team Lead
- **Sidh Khurana**
