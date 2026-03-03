# 🤖 Binary Fault Detection System
### IEEE SB GEHU | Machine Learning Challenge

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Competition%20Ready-success.svg)]()

---

## 📊 Overview

Binary classification system for industrial sensor fault detection achieving **98.78% accuracy** and **98.48% F1-score** through ensemble learning and advanced feature engineering.

**🏆 Key Results:**
- ✅ 98.78% Accuracy | 98.48% F1-Score | 99.92% ROC-AUC
- 🔧 83 features (36 preprocessed + 47 engineered)
- ⚡ 14.5s training time with LightGBM
- 🎯 Production-ready modular architecture

---

## 🎯 Problem Statement

**Objective:** Predict device operational status (Normal/Fault) from 47 sensor readings

**Dataset:**
- Training: 43,776 samples × 47 features
- Class distribution: 60% Normal / 40% Fault
- No missing values, 738 duplicates removed

---

## 🔄 Pipeline Architecture

```
📥 Raw Data (43,776 samples)
    ↓
🧹 Data Cleaning → Remove duplicates (43,038 samples)
    ↓
📊 Preprocessing Pipeline
    ├─ Outlier Treatment (Winsorization 1-99%)
    ├─ Skewness Correction (Yeo-Johnson: 44→0 skewed)
    ├─ Multicollinearity Removal (|r|>0.9: 47→36 features)
    └─ Normalization (StandardScaler)
    ↓
🔧 Feature Engineering (+47 features)
    ├─ Interactions (8): F01×F09, F29×F19
    ├─ Ratios (3): F01/F09, F29/F21
    ├─ Aggregates (22): row/group statistics
    ├─ Polynomials (6): F01², F09², F29²
    └─ Others (8): absolute values, cross-groups
    ↓
🤖 Model Training (5-Fold CV)
    ├─ 🥇 LightGBM (F1: 0.9848) ✓
    ├─ 🥈 XGBoost (F1: 0.9832)
    └─ 🥉 ExtraTrees (F1: 0.9778)
    ↓
🎯 Final Predictions
```

---

## 📈 Performance Metrics

| Model | F1-Score | Accuracy | ROC-AUC | Training Time |
|-------|----------|----------|---------|---------------|
| **LightGBM** ✓ | **0.9848** | **0.9878** | **0.9992** | **14.5s** |
| XGBoost | 0.9832 | 0.9866 | 0.9990 | 22.3s |
| ExtraTrees | 0.9778 | 0.9824 | 0.9990 | 18.7s |

**Confusion Matrix (Estimated):**
- False Positive Rate: ~1.2%
- False Negative Rate: ~1.2%

---

## 📁 Project Structure

```
MLChallenge/
├── README.md                          # Documentation
├── requirements.txt                   # Dependencies
│
├── datacheck.ipynb                    # EDA & visualization
├── preprocessing.ipynb                # Data preprocessing
├── improved_model.py                  # Feature engineering + training
│
├── TRAIN.csv                          # Raw training data
├── TEST.csv                           # Raw test data
├── TRAIN_PREPROCESSED.csv             # Cleaned training data
├── TEST_PREPROCESSED.csv              # Cleaned test data
└── FINAL.csv                          # Submission file
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 4GB RAM minimum

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

**Required packages:**
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
lightgbm>=3.3.0
xgboost>=1.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
```

### Usage

**Option 1: Full Pipeline**
```bash
python improved_model.py
```

**Option 2: Step-by-Step**
```bash
# 1. Exploratory analysis
jupyter notebook datacheck.ipynb

# 2. Preprocessing
jupyter notebook preprocessing.ipynb

# 3. Model training
python improved_model.py
```

**Output:** `FINAL.csv` with predictions (ID, CLASS)

---

## 🔧 Technical Details

### Preprocessing Pipeline

| Step | Method | Impact |
|------|--------|--------|
| **Cleaning** | Duplicate removal | 43,776 → 43,038 samples |
| **Outliers** | Winsorization (1-99%) | ~2% values capped |
| **Skewness** | Yeo-Johnson transform | 44 → 0 skewed features |
| **Multicollinearity** | Correlation threshold (0.9) | 47 → 36 features |
| **Normalization** | StandardScaler | z-score standardization |

### Feature Engineering

**47 engineered features** capturing sensor interactions:

| Category | Count | Examples |
|----------|-------|----------|
| Interactions | 8 | F01×F09, F29×F19 |
| Ratios | 3 | F01/F09, F29/F21 |
| Row Aggregates | 11 | mean, std, min, max, skew, kurtosis |
| Group Aggregates | 11 | early/mid/late sensor statistics |
| Polynomials | 6 | F01², F09², F29² |
| Absolute Values | 6 | \|F01\|, \|F09\|, \|F29\| |
| Cross-groups | 2 | early_mean × late_mean |

### Model Configuration

**LightGBM Hyperparameters:**
```python
{
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5
}
```

**Validation:** Stratified 5-Fold Cross-Validation

---

## 📊 Key Insights

### Data Analysis
- 738 duplicate rows (1.7%) removed
- 44/47 features highly skewed (|skew| > 1)
- 10 feature pairs highly correlated (|r| > 0.9)
- Top predictors: F01, F09, F29, F19, F21

### Model Selection
- LightGBM selected for best F1-score and efficiency
- Ensemble methods outperform single models
- Feature engineering improved performance by ~2%

---

## 🔄 Reproducibility

All results are reproducible with:
- ✅ Fixed random seed (`random_state=42`)
- ✅ Pinned package versions (`requirements.txt`)
- ✅ Stratified cross-validation
- ✅ Saved preprocessed data

---

## 👥 Team

**Team Name:** Tech Ninjas  
**Institution:** Graphic Era Hill University, Dehradun  
**Program:** B.Tech CSE (AI & ML), 3rd Year

**Members:**
- Rajat Pundir (Team Lead)
- Sidh Khurana

---

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details

---

## 🙏 Acknowledgments

- **Competition:** IEEE Student Branch, GEHU
- **Libraries:** scikit-learn, LightGBM, XGBoost, pandas, NumPy

---

**⭐ Star this repo if you found it helpful!**
