import os
import json
import joblib
import argparse
import warnings
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import GradientBoostingRegressor
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
    target_col: str = "CTE",
    random_state: int = 42,
    test_size: float = 0.2,
    n_estimators: int = 200,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    subsample: float = 1.0,
):
    os.makedirs(model_dir, exist_ok=True)

    print(f"[INFO] Reading data from: {train_csv}")
    df = pd.read_csv(train_csv)

    required_cols = {composition_col, temp_col, target_col}
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    magpie_features = [col for col in df.columns if "MagpieData" in col]
    if not magpie_features:
        raise ValueError("No MagpieData feature columns found in the input CSV.")

    features = [temp_col] + magpie_features

    print(f"[INFO] Number of Magpie features: {len(magpie_features)}")
    print(f"[INFO] Total model inputs (including T): {len(features)}")

    # Drop rows with missing required values
    model_df = df[[composition_col, target_col] + features].copy()
    before_drop = len(model_df)
    model_df = model_df.dropna()
    after_drop = len(model_df)

    if after_drop == 0:
        raise ValueError("No rows remain after dropping NaNs.")

    print(f"[INFO] Dropped {before_drop - after_drop} rows with missing values.")
    print(f"[INFO] Remaining rows: {after_drop}")

    # Monazite markers forced into training
    monazite_markers = ["LaPO4", "CePO4", "PrPO4", "NdPO4", "SmPO4", "EuPO4", "GdPO4"]
    marker_indices = model_df[model_df[composition_col].isin(monazite_markers)].index

    print(f"[INFO] Forced monazite marker compositions: {monazite_markers}")
    print(f"[INFO] Marker rows forced into training: {len(marker_indices)}")

    # Split only non-marker compositions
    non_marker_df = model_df[~model_df[composition_col].isin(monazite_markers)].copy()
    if non_marker_df.empty:
        raise ValueError("No non-marker rows found. Cannot perform train/test split.")

    non_marker_groups = non_marker_df[composition_col]

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )

    nm_train_idx_rel, nm_test_idx_rel = next(
        gss.split(non_marker_df[features], non_marker_df[target_col], non_marker_groups)
    )

    nm_train_idx = non_marker_df.iloc[nm_train_idx_rel].index
    nm_test_idx = non_marker_df.iloc[nm_test_idx_rel].index

    # Final split
    train_idx = marker_indices.union(nm_train_idx)
    test_idx = nm_test_idx

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

    # Prepare matrices
    X_train = train_df[features]
    y_train = train_df[target_col]

    X_test = test_df[features]
    y_test = test_df[target_col]

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Gradient Boosting
    print("\n[INFO] Training Gradient Boosting Regressor...")
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        random_state=random_state,
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    metrics = evaluate_model(y_test, y_pred, label="Gradient Boosting Test Performance")

    # Save artifacts
    print("\n[INFO] Saving artifacts...")
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
    joblib.dump(model, os.path.join(model_dir, "gbr_model.joblib"))
    joblib.dump(features, os.path.join(model_dir, "features.joblib"))
    joblib.dump(magpie_features, os.path.join(model_dir, "magpie_features.joblib"))

    metadata = {
        "train_csv": train_csv,
        "composition_col": composition_col,
        "temp_col": temp_col,
        "target_col": target_col,
        "features": features,
        "magpie_features": magpie_features,
        "n_magpie_features": int(len(magpie_features)),
        "n_total_features": int(len(features)),
        "random_state": int(random_state),
        "test_size": float(test_size),
        "monazite_markers": monazite_markers,
        "n_marker_rows_forced_into_training": int(len(marker_indices)),
        "n_rows_input_after_dropna": int(len(model_df)),
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
        "n_unique_train_compositions": int(train_df[composition_col].nunique()),
        "n_unique_test_compositions": int(test_df[composition_col].nunique()),
        "train_compositions": train_comps,
        "test_compositions": test_comps,
        "model_type": "GradientBoostingRegressor",
        "model_params": {
            "n_estimators": int(n_estimators),
            "learning_rate": float(learning_rate),
            "max_depth": int(max_depth),
            "subsample": float(subsample),
            "random_state": int(random_state),
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
    print(f" - {os.path.join(model_dir, 'gbr_model.joblib')}")
    print(f" - {os.path.join(model_dir, 'features.joblib')}")
    print(f" - {os.path.join(model_dir, 'magpie_features.joblib')}")
    print(f" - {os.path.join(model_dir, 'metadata.json')}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and save a Gradient Boosting CTE model from an already-featurized CSV."
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
        default="saved_cte_gbr_models",
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
        default="CTE",
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
        default=200,
        help="Number of boosting stages"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate for Gradient Boosting"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Max depth of individual regression trees"
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=1.0,
        help="Subsample fraction for stochastic boosting"
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
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
    )
