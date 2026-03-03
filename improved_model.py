"""
============================================================================
  ML Challenge — Binary Fault Detection Pipeline
  IEEE SB, GEHU | Online Qualifiers
============================================================================

  Task:      Predict device operational status (Normal=0, Faulty=1)
  Features:  47 sensor readings (F01–F47) → preprocessed to 36 → engineered to 83
  Approach:  Feature Engineering + Tuned LightGBM
  Output:    FINAL.csv (ID → CLASS)

  Pipeline:
    1. Load preprocessed data (TRAIN_PREPROCESSED.csv, TEST_PREPROCESSED.csv)
    2. Engineer 47 new features (interactions, aggregates, polynomials)
    3. Train optimized LightGBM with 5-fold stratified cross-validation
    4. Generate predictions on test set → FINAL.csv

  Requirements: numpy, pandas, scikit-learn, xgboost, lightgbm
============================================================================
"""

import numpy as np
import pandas as pd
import time
import warnings
from typing import List, Tuple

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================
#  Configuration
# ============================================================
RANDOM_STATE = 42
CV_FOLDS = 5
TRAIN_PATH = 'TRAIN_PREPROCESSED.csv'
TEST_PATH = 'TEST_PREPROCESSED.csv'
OUTPUT_PATH = 'FINAL.csv'

# Top predictive features identified via correlation analysis (|r| > 0.33)
TOP_FEATURES = ['F05', 'F06', 'F07', 'F09', 'F19', 'F21']

# Feature interaction pairs (chosen from top correlated features)
INTERACTION_PAIRS: List[Tuple[str, str]] = [
    ('F05', 'F09'), ('F05', 'F19'), ('F05', 'F21'),
    ('F06', 'F07'), ('F06', 'F09'),
    ('F09', 'F19'), ('F09', 'F21'),
    ('F19', 'F21'),
]

# Ratio feature pairs
RATIO_PAIRS: List[Tuple[str, str]] = [
    ('F19', 'F09'), ('F21', 'F09'), ('F05', 'F06'),
]

# Sensor groups (from domain analysis of feature structure)
SENSOR_GROUPS = {
    'early': ['F02', 'F03', 'F04', 'F05', 'F06', 'F07', 'F09'],
    'mid':   ['F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18'],
    'high':  ['F30', 'F31', 'F32', 'F33', 'F34', 'F35', 'F36'],
    'late':  ['F39', 'F40', 'F42', 'F43', 'F44', 'F45', 'F46', 'F47'],
}


# ============================================================
#  Feature Engineering
# ============================================================
def engineer_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Engineer new features from preprocessed sensor data.

    Adds 47 features across 7 categories:
      - 8 interaction features (product of top predictor pairs)
      - 3 ratio features (division of correlated pairs)
      - 11 row-wise aggregate features (statistical summaries per sample)
      - 11 sensor group aggregates (mean/std/max per sensor group)
      - 6 polynomial features (squared top predictors)
      - 6 absolute value features
      - 2 cross-group interaction features

    Args:
        df: DataFrame with preprocessed features
        feature_cols: List of original feature column names

    Returns:
        DataFrame with original + engineered features
    """
    result = df.copy()
    available_cols = set(df.columns)

    # 1. Interaction features: capture non-linear relationships
    for f1, f2 in INTERACTION_PAIRS:
        if f1 in available_cols and f2 in available_cols:
            result[f'{f1}_x_{f2}'] = df[f1] * df[f2]

    # 2. Ratio features: capture relative sensor magnitudes
    for f1, f2 in RATIO_PAIRS:
        if f1 in available_cols and f2 in available_cols:
            result[f'{f1}_div_{f2}'] = df[f1] / (df[f2] + 1e-8)

    # 3. Row-wise aggregates: capture holistic device behavior
    feature_matrix = df[feature_cols].values
    result['row_mean'] = np.mean(feature_matrix, axis=1)
    result['row_std'] = np.std(feature_matrix, axis=1)
    result['row_max'] = np.max(feature_matrix, axis=1)
    result['row_min'] = np.min(feature_matrix, axis=1)
    result['row_range'] = result['row_max'] - result['row_min']
    result['row_median'] = np.median(feature_matrix, axis=1)
    result['row_skew'] = pd.DataFrame(feature_matrix).apply(
        lambda x: x.skew(), axis=1
    ).values
    result['row_kurtosis'] = pd.DataFrame(feature_matrix).apply(
        lambda x: x.kurtosis(), axis=1
    ).values
    result['row_n_positive'] = np.sum(feature_matrix > 0, axis=1)
    result['row_n_high'] = np.sum(feature_matrix > 1.0, axis=1)
    result['row_abs_sum'] = np.sum(np.abs(feature_matrix), axis=1)

    # 4. Sensor group aggregates: exploit known sensor structure
    group_means = {}
    for group_name, group_cols in SENSOR_GROUPS.items():
        valid_cols = [c for c in group_cols if c in available_cols]
        if valid_cols:
            group_means[group_name] = df[valid_cols].mean(axis=1)
            result[f'grp_{group_name}_mean'] = group_means[group_name]
            result[f'grp_{group_name}_std'] = df[valid_cols].std(axis=1)
            if group_name != 'late':  # max less informative for late sensors
                result[f'grp_{group_name}_max'] = df[valid_cols].max(axis=1)

    # 5. Polynomial features: capture quadratic relationships
    for f in TOP_FEATURES:
        if f in available_cols:
            result[f'{f}_sq'] = df[f] ** 2

    # 6. Absolute value features: magnitude matters for fault detection
    for f in TOP_FEATURES:
        if f in available_cols:
            result[f'{f}_abs'] = np.abs(df[f])

    # 7. Cross-group interactions: capture inter-system relationships
    if 'early' in group_means and 'mid' in group_means:
        result['grp_early_x_mid'] = group_means['early'] * group_means['mid']
    if 'early' in group_means and 'high' in group_means:
        result['grp_early_x_high'] = group_means['early'] * group_means['high']

    return result


# ============================================================
#  Model Definitions
# ============================================================
def get_models() -> dict:
    """
    Return tuned model configurations.

    Hyperparameters were optimized via RandomizedSearchCV (50 iterations,
    5-fold CV) in the exploration phase. LightGBM achieved the best
    individual CV F1-Score.
    """
    return {
        'XGBoost': XGBClassifier(
            n_estimators=800, learning_rate=0.05, max_depth=8,
            subsample=0.9, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=2,
            random_state=RANDOM_STATE, eval_metric='logloss', n_jobs=-1,
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=800, learning_rate=0.05, max_depth=-1,
            num_leaves=100, subsample=1.0, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1,
            random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
        ),
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=600, max_depth=None,
            min_samples_split=2, min_samples_leaf=1, max_features='sqrt',
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }


# ============================================================
#  Cross-Validation
# ============================================================
def evaluate_models(models: dict, X: np.ndarray, y: np.ndarray) -> dict:
    """
    Run 5-fold stratified cross-validation on all models.

    Returns dict mapping model name to CV metrics (f1, accuracy, roc_auc).
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    for name, model in models.items():
        print(f"\n   Cross-validating {name}...")
        start = time.time()

        f1 = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=-1)
        acc = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        roc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)

        elapsed = time.time() - start
        results[name] = {
            'f1': f1.mean(), 'f1_std': f1.std(),
            'acc': acc.mean(), 'acc_std': acc.std(),
            'roc': roc.mean(), 'roc_std': roc.std(),
            'time': elapsed,
        }

        print(f"      Accuracy : {acc.mean():.4f} +/- {acc.std():.4f}")
        print(f"      F1-Score : {f1.mean():.4f} +/- {f1.std():.4f}")
        print(f"      ROC-AUC  : {roc.mean():.4f} +/- {roc.std():.4f}")
        print(f"      Time     : {elapsed:.1f}s")

    return results


# ============================================================
#  Main Pipeline
# ============================================================
def main():
    print("=" * 65)
    print("  ML CHALLENGE - IMPROVED MODEL PIPELINE")
    print("=" * 65)

    # --- Step 1: Load Data ---
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)

    feature_cols = [c for c in df_train.columns if c != 'Class']
    X_raw = df_train[feature_cols].copy()
    y = df_train['Class'].values
    test_ids = df_test['ID'].values
    X_test_raw = df_test[feature_cols].copy()

    print(f"\n  [1/6] Data Loaded:")
    print(f"        Train: {X_raw.shape[0]} samples x {X_raw.shape[1]} features")
    print(f"        Test:  {X_test_raw.shape[0]} samples x {X_test_raw.shape[1]} features")
    print(f"        Classes: Normal={int((y==0).sum())} | Faulty={int((y==1).sum())}")

    # --- Step 2: Feature Engineering ---
    print(f"\n  [2/6] Feature Engineering...")
    X_train_eng = engineer_features(X_raw, feature_cols)
    X_test_eng = engineer_features(X_test_raw, feature_cols)
    eng_cols = list(X_train_eng.columns)
    print(f"        {len(feature_cols)} -> {len(eng_cols)} features (+{len(eng_cols)-len(feature_cols)} new)")

    # --- Step 3: Scale ---
    print(f"\n  [3/6] Scaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_eng)
    X_test = scaler.transform(X_test_eng)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"        Done. Shape: {X_train.shape}")

    # --- Step 4: Cross-Validation ---
    print(f"\n  [4/6] Cross-Validation ({CV_FOLDS}-fold stratified):")
    models = get_models()
    cv_results = evaluate_models(models, X_train, y)

    # --- Step 5: Select Best Model ---
    print(f"\n  [5/6] Model Selection:")
    best_name = max(cv_results, key=lambda k: cv_results[k]['f1'])
    best_f1 = cv_results[best_name]['f1']

    for name in sorted(cv_results, key=lambda k: cv_results[k]['f1'], reverse=True):
        marker = " >> " if name == best_name else "    "
        print(f"      {marker}{name:15s}: F1={cv_results[name]['f1']:.4f}  "
              f"Acc={cv_results[name]['acc']:.4f}  AUC={cv_results[name]['roc']:.4f}")

    print(f"\n        Best: {best_name} (CV F1 = {best_f1:.4f})")

    # --- Step 6: Train & Predict ---
    print(f"\n  [6/6] Final Training & Prediction:")
    final_model = models[best_name]
    start = time.time()
    final_model.fit(X_train, y)
    print(f"        Trained {best_name} on {len(y)} samples in {time.time()-start:.1f}s")

    predictions = final_model.predict(X_test)
    output = pd.DataFrame({
        'ID': test_ids.astype(int),
        'CLASS': predictions.astype(int),
    })

    # Verify output format
    assert len(output) == len(test_ids), f'Row count mismatch: {len(output)} vs {len(test_ids)}'
    assert list(output.columns) == ['ID', 'CLASS'], f'Column mismatch: {list(output.columns)}'
    assert (output['ID'].values == test_ids).all(), 'ID order mismatch'

    output.to_csv(OUTPUT_PATH, index=False)

    n0, n1 = int((output['CLASS'] == 0).sum()), int((output['CLASS'] == 1).sum())
    print(f"        Saved {OUTPUT_PATH}: {len(output)} rows  "
          f"(Normal={n0}, Faulty={n1}, ratio={n0/max(n1,1):.2f}:1)")

    print(f"\n{'='*65}")
    print("  DONE - FINAL.csv ready for submission")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
