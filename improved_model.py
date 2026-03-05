# Binary Fault Detection - ML Challenge
# Team Tech Ninjas | IEEE SB GEHU
# Rajat Pundir, Sidh Khurana

import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

SEED = 42
np.random.seed(SEED)

TRAIN_FILE = 'TRAIN_PREPROCESSED.csv'
TEST_FILE = 'TEST_PREPROCESSED.csv'
SUBMISSION_FILE = 'FINAL.csv'
NUM_FOLDS = 5

# ============================================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================================

def create_interaction_features(data, cols):
    """
    Multiply important sensor pairs together.
    Creates 8 interaction features from top predictors (F05, F06, F07, F09, F19, F21).
    """
    df = data.copy()
    
    important_pairs = [
        ('F05', 'F09'), ('F05', 'F19'), ('F05', 'F21'),
        ('F06', 'F07'), ('F06', 'F09'),
        ('F09', 'F19'), ('F09', 'F21'),
        ('F19', 'F21')
    ]
    
    for sensor1, sensor2 in important_pairs:
        if sensor1 in df.columns and sensor2 in df.columns:
            df[f'{sensor1}_{sensor2}_interaction'] = df[sensor1] * df[sensor2]
    
    return df


def create_ratio_features(data):
    """
    Ratio features for key sensor pairs.
    Creates 3 ratio features - helps capture relative relationships between sensors.
    """
    df = data.copy()
    
    ratio_pairs = [('F19', 'F09'), ('F21', 'F09'), ('F05', 'F06')]
    
    for numerator, denominator in ratio_pairs:
        if numerator in df.columns and denominator in df.columns:
            df[f'{numerator}_to_{denominator}_ratio'] = df[numerator] / (df[denominator] + 1e-8)
    
    return df


def create_statistical_features(data, feature_columns):
    """
    Row-wise stats across all sensors.
    Creates 11 statistical features - mean, std, min, max, range, median, skew, kurtosis, etc.
    Captures overall sensor behavior patterns for each sample.
    """
    df = data.copy()
    sensor_values = df[feature_columns].values
    df['sensors_mean'] = np.mean(sensor_values, axis=1)
    df['sensors_std'] = np.std(sensor_values, axis=1)
    df['sensors_max'] = np.max(sensor_values, axis=1)
    df['sensors_min'] = np.min(sensor_values, axis=1)
    df['sensors_range'] = df['sensors_max'] - df['sensors_min']
    df['sensors_median'] = np.median(sensor_values, axis=1)
    df['sensors_skewness'] = pd.DataFrame(sensor_values).apply(lambda x: x.skew(), axis=1).values
    df['sensors_kurtosis'] = pd.DataFrame(sensor_values).apply(lambda x: x.kurtosis(), axis=1).values
    df['positive_sensor_count'] = np.sum(sensor_values > 0, axis=1)
    df['high_value_count'] = np.sum(sensor_values > 1.0, axis=1)
    df['total_abs_magnitude'] = np.sum(np.abs(sensor_values), axis=1)
    
    return df


def create_group_features(data):
    """
    Group sensors and get stats per group.
    Creates 11 group-based features + 2 cross-group interactions.
    Groups: early (F02-F09), middle (F10-F18), high_range (F30-F36), late (F39-F47).
    """
    df = data.copy()
    
    groups = {
        'early_sensors': ['F02', 'F03', 'F04', 'F05', 'F06', 'F07', 'F09'],
        'middle_sensors': ['F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18'],
        'high_range_sensors': ['F30', 'F31', 'F32', 'F33', 'F34', 'F35', 'F36'],
        'late_sensors': ['F39', 'F40', 'F42', 'F43', 'F44', 'F45', 'F46', 'F47']
    }
    
    group_stats = {}
    
    for group_name, sensor_list in groups.items():
        available_sensors = [s for s in sensor_list if s in df.columns]
        if available_sensors:
            group_stats[group_name] = df[available_sensors].mean(axis=1)
            df[f'{group_name}_avg'] = group_stats[group_name]
            df[f'{group_name}_std'] = df[available_sensors].std(axis=1)
            if group_name != 'late_sensors':
                df[f'{group_name}_max'] = df[available_sensors].max(axis=1)
    
    if 'early_sensors' in group_stats and 'middle_sensors' in group_stats:
        df['early_middle_interaction'] = group_stats['early_sensors'] * group_stats['middle_sensors']
    
    if 'early_sensors' in group_stats and 'high_range_sensors' in group_stats:
        df['early_high_interaction'] = group_stats['early_sensors'] * group_stats['high_range_sensors']
    
    return df


def create_polynomial_features(data):
    """
    Squared and absolute features for top predictors.
    Creates 12 features (6 squared + 6 absolute) for sensors: F05, F06, F07, F09, F19, F21.
    Helps capture non-linear relationships.
    """
    df = data.copy()
    top_sensors = ['F05', 'F06', 'F07', 'F09', 'F19', 'F21']
    
    for sensor in top_sensors:
        if sensor in df.columns:
            df[f'{sensor}_squared'] = df[sensor] ** 2
            df[f'{sensor}_absolute'] = np.abs(df[sensor])
    
    return df


def engineer_all_features(data, original_features):
    """
    Run all feature engineering steps.
    Takes 36 preprocessed features -> creates 47 new features -> returns 83 total features.
    """
    print("    Creating interaction features...")
    data = create_interaction_features(data, original_features)
    
    print("    Creating ratio features...")
    data = create_ratio_features(data)
    
    print("    Creating statistical features...")
    data = create_statistical_features(data, original_features)
    
    print("    Creating group-based features...")
    data = create_group_features(data)
    
    print("    Creating polynomial features...")
    data = create_polynomial_features(data)
    
    return data

# ============================================================================
# MODEL BUILDING & EVALUATION
# ============================================================================

def build_models():
    """
    Set up the models we're comparing.
    XGBoost, LightGBM, and ExtraTrees with tuned hyperparameters.
    """
    models = {}
    
    models['XGBoost'] = XGBClassifier(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=2,
        random_state=SEED,
        eval_metric='logloss',
        n_jobs=-1
    )
    
    models['LightGBM'] = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=100,
        subsample=1.0,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=SEED,
        verbose=-1,
        n_jobs=-1
    )
    
    models['ExtraTrees'] = ExtraTreesClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=SEED,
        n_jobs=-1
    )
    
    return models


def evaluate_model_performance(models, X_train, y_train):
    """
    Cross-validate all models and compare.
    Uses 5-fold Stratified CV to evaluate F1, Accuracy, and ROC-AUC scores.
    """
    cv_splitter = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    performance_results = {}
    
    for model_name, model in models.items():
        print(f"\n  Evaluating {model_name}...")
        start_time = time.time()
        
        f1_scores = cross_val_score(model, X_train, y_train, cv=cv_splitter, scoring='f1', n_jobs=-1)
        accuracy_scores = cross_val_score(model, X_train, y_train, cv=cv_splitter, scoring='accuracy', n_jobs=-1)
        auc_scores = cross_val_score(model, X_train, y_train, cv=cv_splitter, scoring='roc_auc', n_jobs=-1)
        
        elapsed_time = time.time() - start_time
        
        performance_results[model_name] = {
            'f1_mean': f1_scores.mean(),
            'f1_std': f1_scores.std(),
            'accuracy_mean': accuracy_scores.mean(),
            'accuracy_std': accuracy_scores.std(),
            'auc_mean': auc_scores.mean(),
            'auc_std': auc_scores.std(),
            'training_time': elapsed_time
        }
        
        print(f"    Accuracy: {accuracy_scores.mean():.4f} (+/- {accuracy_scores.std():.4f})")
        print(f"    F1 Score: {f1_scores.mean():.4f} (+/- {f1_scores.std():.4f})")
        print(f"    ROC AUC:  {auc_scores.mean():.4f} (+/- {auc_scores.std():.4f})")
        print(f"    Time:     {elapsed_time:.1f} seconds")
    
    return performance_results

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("=" * 70)
    print("  BINARY FAULT DETECTION SYSTEM - ML CHALLENGE")
    print("  Team Tech Ninjas | IEEE SB GEHU")
    print("=" * 70)
    
    # ========================================================================
    # Step 1: Load preprocessed data
    # ========================================================================
    print("\n[Step 1/6] Loading preprocessed data...")
    train_data = pd.read_csv(TRAIN_FILE)
    test_data = pd.read_csv(TEST_FILE)
    
    original_features = [col for col in train_data.columns if col != 'Class']
    X_train_raw = train_data[original_features].copy()
    y_train = train_data['Class'].values
    
    test_ids = test_data['ID'].values
    X_test_raw = test_data[original_features].copy()
    
    print(f"  Training samples: {len(X_train_raw)}")
    print(f"  Test samples: {len(X_test_raw)}")
    print(f"  Original features: {len(original_features)}")
    print(f"  Normal cases: {(y_train == 0).sum()}")
    print(f"  Fault cases: {(y_train == 1).sum()}")
    
    # ========================================================================
    # Step 2: Feature engineering (36 -> 83 features)
    # ========================================================================
    print("\n[Step 2/6] Engineering new features...")
    X_train_engineered = engineer_all_features(X_train_raw, original_features)
    X_test_engineered = engineer_all_features(X_test_raw, original_features)
    
    total_features = len(X_train_engineered.columns)
    new_features = total_features - len(original_features)
    print(f"  Total features after engineering: {total_features}")
    print(f"  New features created: {new_features}")
    
    # ========================================================================
    # Step 3: Scale features (StandardScaler)
    # ========================================================================
    print("\n[Step 3/6] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_engineered)
    X_test_scaled = scaler.transform(X_test_engineered)
    
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"  Scaled data shape: {X_train_scaled.shape}")
    
    # ========================================================================
    # Step 4: Cross-validation (5-fold Stratified CV)
    # ========================================================================
    print(f"\n[Step 4/6] Cross-validation ({NUM_FOLDS} folds)...")
    all_models = build_models()
    results = evaluate_model_performance(all_models, X_train_scaled, y_train)
    
    # ========================================================================
    # Step 5: Select best model based on F1 score
    # ========================================================================
    print("\n[Step 5/6] Selecting best model...")
    best_model_name = max(results, key=lambda name: results[name]['f1_mean'])
    
    print("\n  Model Rankings (by F1 Score):")
    for name in sorted(results, key=lambda n: results[n]['f1_mean'], reverse=True):
        is_best = " *** SELECTED ***" if name == best_model_name else ""
        print(f"    {name:12s}: F1={results[name]['f1_mean']:.4f}, "
              f"Acc={results[name]['accuracy_mean']:.4f}, "
              f"AUC={results[name]['auc_mean']:.4f}{is_best}")
    
    print(f"\n  Best model: {best_model_name}")
    print(f"  Cross-validated F1 Score: {results[best_model_name]['f1_mean']:.4f}")
    
    # ========================================================================
    # Step 6: Train final model and generate predictions
    # ========================================================================
    print("\n[Step 6/6] Training final model and generating predictions...")
    final_model = all_models[best_model_name]
    
    train_start = time.time()
    final_model.fit(X_train_scaled, y_train)
    train_time = time.time() - train_start
    
    print(f"  Model trained in {train_time:.1f} seconds")
    
    predictions = final_model.predict(X_test_scaled)
    
    submission = pd.DataFrame({
        'ID': test_ids.astype(int),
        'CLASS': predictions.astype(int)
    })
    assert len(submission) == len(test_ids)
    assert list(submission.columns) == ['ID', 'CLASS']
    
    submission.to_csv(SUBMISSION_FILE, index=False)
    
    normal_count = (predictions == 0).sum()
    fault_count = (predictions == 1).sum()
    print(f"\n  Predictions saved to {SUBMISSION_FILE}")
    print(f"  Total predictions: {len(predictions)}")
    print(f"  Predicted Normal: {normal_count}")
    print(f"  Predicted Fault: {fault_count}")
    print(f"  Normal:Fault ratio: {normal_count/max(fault_count, 1):.2f}:1")
    
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETED SUCCESSFULLY")
    print("  Submission file ready: FINAL.csv")
    print("=" * 70)


if __name__ == '__main__':
    main()
