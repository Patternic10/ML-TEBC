#!/usr/bin/env python3
"""
train_kappa.py

Train and save a production Random Forest model for kappa
from an already-featurized CSV.

This version is designed to match the original RF setup:
    RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

Usage:
python train_kappa.py --train-csv kappa_n_F.csv --model-dir saved_kappa_rf_models
"""

import os
import json
import joblib
import argparse
import warnings
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")


def evaluate_model(y_true, y_pred, label="Model"):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"\n=== {label} ===")
    print(f"R2   : {r2:.6f}")
    print(f"MAE  : {mae:.6f}")
    print(f"RMSE : {rmse:.6f}")

    return {
        "R2": float(r2),
        "MAE": float(mae),
        "RMSE": float(rmse),
    }


def train_and_save(
    train_csv: str,
    model_dir: str,
    composition_col: str = "Composition",
    temp_col: str = "T",
    target_col: str = "kappa",
    random_state: int = 42,
    test_size: float = 0.2,
    n_estimators: int = 100,
    n_jobs: int = -1,
):
    os.makedirs(model_dir, exist_ok=True)

    print(f"[INFO] Reading data from: {train_csv}")
    df = pd.read_csv(train_csv)

    required_cols = {composition_col, temp_col, target_col}
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    # Match your original kappa code exactly:
    # features = ['T'] + all columns after target_col
    target_idx = df.columns.get_loc(target_col)
    descriptor_features = df.columns[target_idx + 1:].tolist()
    feature_cols = [temp_col] + descriptor_features

    if not descriptor_features:
        raise ValueError("No descriptor feature columns found after the target column.")

    print(f"[INFO] Number of descriptor features: {len(descriptor_features)}")
    print(f"[INFO] Total model inputs (including T): {len(feature_cols)}")

    # Keep rows as in the original workflow unless required columns are missing.
    model_df = df[[composition_col, target_col] + feature_cols].copy()

    # Optional safety check: fail if NaNs exist instead of silently changing the dataset.
    if model_df.isna().any().any():
        nan_counts = model_df.isna().sum()
        nan_counts = nan_counts[nan_counts > 0]
        raise ValueError(
            "Input CSV contains missing values. This script does not drop rows automatically "
            "because that would change your split and metrics.\n"
            f"Columns with NaNs:\n{nan_counts.to_string()}"
        )

    X = model_df[feature_cols]
    y = model_df[target_col]

    # Force monazite markers into training set
    monazite_markers = ['LaPO4', 'CePO4', 'PrPO4', 'NdPO4', 'SmPO4', 'EuPO4', 'GdPO4']
    forced_train_mask = model_df[composition_col].isin(monazite_markers)
    forced_train_idx = model_df.index[forced_train_mask].tolist()

    print(f"[INFO] Forced monazite marker compositions: {monazite_markers}")
    print(f"[INFO] Marker rows forced into training: {len(forced_train_idx)}")

    # Split remaining compositions exactly like your original code
    remaining_df = model_df[~forced_train_mask].copy()
    if remaining_df.empty:
        raise ValueError("No non-marker rows found. Cannot perform train/test split.")

    X_rem = remaining_df[feature_cols]
    y_rem = remaining_df[target_col]
    groups_rem = remaining_df[composition_col]

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    rem_train_sub_idx, test_idx_raw = next(gss.split(X_rem, y_rem, groups_rem))

    # Reconstruct final indices exactly like your original code
    train_idx = forced_train_idx + remaining_df.iloc[rem_train_sub_idx].index.tolist()
    test_idx = remaining_df.iloc[test_idx_raw].index.tolist()

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]

    train_df = model_df.loc[train_idx].copy()
    test_df = model_df.loc[test_idx].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Training or test set is empty after splitting.")

    print(f"[INFO] Train rows: {len(train_df)}")
    print(f"[INFO] Test rows: {len(test_df)}")
    print(f"[INFO] Unique train compositions: {train_df[composition_col].nunique()}")
    print(f"[INFO] Unique test compositions: {test_df[composition_col].nunique()}")

    train_comps = sorted(train_df[composition_col].unique().tolist())
    test_comps = sorted(test_df[composition_col].unique().tolist())

    # Scale features exactly like your original code
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train RF exactly like your original code
    print("\n[INFO] Training Random Forest Regressor...")
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    metrics = evaluate_model(y_test, y_pred, label="Random Forest Test Performance")

    # Save test predictions for checking
    test_results = model_df.loc[test_idx].copy()
    test_results["RF_Pred"] = y_pred
    test_results.to_csv(os.path.join(model_dir, "test_predictions.csv"), index=False)

    # Save artifacts
    print("\n[INFO] Saving artifacts...")
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
    joblib.dump(model, os.path.join(model_dir, "rf_model.joblib"))
    joblib.dump(feature_cols, os.path.join(model_dir, "features.joblib"))
    joblib.dump(descriptor_features, os.path.join(model_dir, "descriptor_features.joblib"))

    metadata = {
        "train_csv": train_csv,
        "composition_col": composition_col,
        "temp_col": temp_col,
        "target_col": target_col,
        "features": feature_cols,
        "descriptor_features": descriptor_features,
        "n_descriptor_features": int(len(descriptor_features)),
        "n_total_features": int(len(feature_cols)),
        "random_state": int(random_state),
        "test_size": float(test_size),
        "monazite_markers": monazite_markers,
        "n_marker_rows_forced_into_training": int(len(forced_train_idx)),
        "n_rows_input": int(len(model_df)),
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
        "n_unique_train_compositions": int(train_df[composition_col].nunique()),
        "n_unique_test_compositions": int(test_df[composition_col].nunique()),
        "train_compositions": train_comps,
        "test_compositions": test_comps,
        "model_type": "RandomForestRegressor",
        "model_params": {
            "n_estimators": int(n_estimators),
            "random_state": int(random_state),
            "n_jobs": int(n_jobs),
        },
        "test_metrics": metrics,
        "featurization_done_in_script": False,
        "input_expected_to_be_pre_featurized": True,
    }

    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Saved artifacts to: {model_dir}")
    print("[OK] Files:")
    print(f" - {os.path.join(model_dir, 'scaler.joblib')}")
    print(f" - {os.path.join(model_dir, 'rf_model.joblib')}")
    print(f" - {os.path.join(model_dir, 'features.joblib')}")
    print(f" - {os.path.join(model_dir, 'descriptor_features.joblib')}")
    print(f" - {os.path.join(model_dir, 'metadata.json')}")
    print(f" - {os.path.join(model_dir, 'test_predictions.csv')}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and save a Random Forest kappa model from an already-featurized CSV."
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        required=True,
        help="Path to already-featurized CSV"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="saved_kappa_rf_models",
        help="Directory to save trained artifacts"
    )
    parser.add_argument(
        "--composition-col",
        type=str,
        default="Composition",
        help="Composition column name"
    )
    parser.add_argument(
        "--temp-col",
        type=str,
        default="T",
        help="Temperature column name"
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="kappa",
        help="Target column name"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of non-marker compositions used for test"
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in the forest"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train_and_save(
        train_csv=args.train_csv,
        model_dir=args.model_dir,
        composition_col=args.composition_col,
        temp_col=args.temp_col,
        target_col=args.target_col,
        random_state=args.random_state,
        test_size=args.test_size,
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
    )
