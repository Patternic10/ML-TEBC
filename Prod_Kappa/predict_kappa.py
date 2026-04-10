#!/usr/bin/env python3
"""
predict_kappa.py

Batch predict kappa vs temperature from composition using a saved RF model.
Designed to mirror the optimized CTE prediction workflow.

Usage examples:

Single composition:
python predict_kappa.py \
    --model-dir saved_kappa_rf_models \
    --composition "Lu0.25Y0.25Er0.25Yb0.25PO4" \
    --output-csv pred_kappa.csv \
    --plot

Multiple compositions from CSV:
python predict_kappa.py \
    --model-dir saved_kappa_rf_models \
    --input-csv new_compositions.csv \
    --composition-col Composition \
    --output-csv pred_kappa.csv \
    --plot
"""

import os
import joblib
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pymatgen.core import Composition
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.composition import ElementProperty, Stoichiometry


def load_artifacts(model_dir: str):
    required_files = [
        "scaler.joblib",
        "rf_model.joblib",
        "features.joblib",
        "descriptor_features.joblib",
    ]

    for fname in required_files:
        fpath = os.path.join(model_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing required artifact: {fpath}")

    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    model = joblib.load(os.path.join(model_dir, "rf_model.joblib"))
    features = joblib.load(os.path.join(model_dir, "features.joblib"))
    descriptor_features = joblib.load(os.path.join(model_dir, "descriptor_features.joblib"))

    return scaler, model, features, descriptor_features


def build_featurizer():
    return MultipleFeaturizer([
        ElementProperty.from_preset("magpie"),
        Stoichiometry(),
    ])


def normalize_formula(formula: str) -> str:
    return str(formula).strip().replace(" ", "")


def load_compositions(args):
    if args.composition:
        return [normalize_formula(args.composition)]

    df = pd.read_csv(args.input_csv)
    if args.composition_col not in df.columns:
        raise ValueError(f"Column '{args.composition_col}' not found in {args.input_csv}")

    compositions = (
        df[args.composition_col]
        .dropna()
        .astype(str)
        .map(normalize_formula)
        .unique()
        .tolist()
    )

    if not compositions:
        raise ValueError("No compositions found in the input CSV.")

    return compositions


def featurize_compositions_batch(compositions, descriptor_features):
    """
    Featurize all compositions in one batch using matminer.
    Keeps only the descriptor columns expected by the trained kappa model.
    """
    featurizer = build_featurizer()

    comp_objs = []
    valid_formulas = []
    failed = []

    for formula in compositions:
        try:
            comp_objs.append(Composition(formula))
            valid_formulas.append(formula)
        except Exception as e:
            failed.append({
                "Composition": formula,
                "error": f"Composition parsing failed: {e}"
            })

    if not valid_formulas:
        raise RuntimeError("No valid compositions available for featurization.")

    df = pd.DataFrame({
        "Composition": valid_formulas,
        "composition_obj": comp_objs,
    })

    feat_df = featurizer.featurize_dataframe(
        df,
        col_id="composition_obj",
        ignore_errors=True,
        return_errors=True,
        inplace=False,
    )

    if "MultipleFeaturizer Exceptions" in feat_df.columns:
        error_mask = feat_df["MultipleFeaturizer Exceptions"].notna()
        if error_mask.any():
            for _, row in feat_df.loc[error_mask, ["Composition", "MultipleFeaturizer Exceptions"]].iterrows():
                failed.append({
                    "Composition": row["Composition"],
                    "error": f"Featurization failed: {row['MultipleFeaturizer Exceptions']}"
                })
        feat_df = feat_df.loc[~error_mask].copy()

    missing = [c for c in descriptor_features if c not in feat_df.columns]
    if missing:
        raise ValueError(
            "Generated features do not match the saved training descriptor columns.\n"
            f"First few missing columns: {missing[:10]}"
        )

    feat_df = feat_df[["Composition"] + descriptor_features].copy()
    failed_df = pd.DataFrame(failed) if failed else None

    return feat_df, failed_df


def expand_features_over_temperature(
    feat_df: pd.DataFrame,
    descriptor_features: list[str],
    temp_col: str = "T",
    tmin: int = 100,
    tmax: int = 2000,
    step: int = 100,
):
    """
    Expand each composition row over the requested temperature grid efficiently.
    """
    temps = np.arange(tmin, tmax + step, step)
    n_temps = len(temps)
    n_comps = len(feat_df)

    repeated_feat = feat_df.loc[feat_df.index.repeat(n_temps)].reset_index(drop=True)
    repeated_feat[temp_col] = np.tile(temps, n_comps)

    repeated_feat = repeated_feat[["Composition", temp_col] + descriptor_features]

    return repeated_feat


def predict_batch(
    compositions,
    scaler,
    model,
    features,
    descriptor_features,
    temp_col="T",
    tmin=100,
    tmax=2000,
    step=100,
):
    feat_df, failed_df = featurize_compositions_batch(compositions, descriptor_features)

    pred_df = expand_features_over_temperature(
        feat_df=feat_df,
        descriptor_features=descriptor_features,
        temp_col=temp_col,
        tmin=tmin,
        tmax=tmax,
        step=step,
    )

    missing = [c for c in features if c not in pred_df.columns]
    if missing:
        raise ValueError(
            "Prediction dataframe is missing required model inputs.\n"
            f"First few missing columns: {missing[:10]}"
        )

    X_pred = pred_df[features]
    X_scaled = scaler.transform(X_pred)
    pred_df["RF_Pred"] = model.predict(X_scaled)

    pred_df = pred_df[["Composition", temp_col, "RF_Pred"]]
    return pred_df, failed_df


def make_plot(pred_df, temp_col="T", pred_col="RF_Pred", output_plot=None):
    plt.figure(figsize=(9, 6))

    for comp, grp in pred_df.groupby("Composition"):
        grp = grp.sort_values(temp_col)
        plt.plot(grp[temp_col], grp[pred_col], marker="o", label=comp)

    plt.xlabel("Temperature (K)")
    plt.ylabel("kappa")
    plt.title("Predicted kappa vs T (Random Forest)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    if output_plot:
        plt.savefig(output_plot, dpi=300, bbox_inches="tight")
        print(f"[OK] Plot saved to: {output_plot}")

    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch predict kappa vs temperature from composition using a saved RF model."
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing saved model artifacts"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--composition",
        type=str,
        help="Single composition formula"
    )
    group.add_argument(
        "--input-csv",
        type=str,
        help="CSV containing a Composition column"
    )

    parser.add_argument(
        "--composition-col",
        type=str,
        default="Composition",
        help="Composition column name if using --input-csv"
    )
    parser.add_argument(
        "--temp-col",
        type=str,
        default="T",
        help="Temperature column name"
    )
    parser.add_argument(
        "--tmin",
        type=int,
        default=100,
        help="Minimum temperature"
    )
    parser.add_argument(
        "--tmax",
        type=int,
        default=2000,
        help="Maximum temperature"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=100,
        help="Temperature step"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional output CSV path"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Make a plot"
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default=None,
        help="Optional output plot path"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    scaler, model, features, descriptor_features = load_artifacts(args.model_dir)
    compositions = load_compositions(args)

    pred_df, failed_df = predict_batch(
        compositions=compositions,
        scaler=scaler,
        model=model,
        features=features,
        descriptor_features=descriptor_features,
        temp_col=args.temp_col,
        tmin=args.tmin,
        tmax=args.tmax,
        step=args.step,
    )

    print(pred_df)

    if args.output_csv:
        pred_df.to_csv(args.output_csv, index=False)
        print(f"[OK] Predictions saved to: {args.output_csv}")

        if failed_df is not None and not failed_df.empty:
            fail_path = os.path.splitext(args.output_csv)[0] + "_failed.csv"
            failed_df.to_csv(fail_path, index=False)
            print(f"[WARN] Failed compositions saved to: {fail_path}")

    if args.plot:
        make_plot(
            pred_df=pred_df,
            temp_col=args.temp_col,
            pred_col="RF_Pred",
            output_plot=args.output_plot,
        )
