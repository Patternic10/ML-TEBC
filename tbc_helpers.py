from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.composition import ElementProperty, Stoichiometry
from pymatgen.core import Composition

REQUIRED_KAPPA_ARTIFACTS = [
    "scaler.joblib",
    "rf_model.joblib",
    "features.joblib",
    "descriptor_features.joblib",
]

REQUIRED_CTE_ARTIFACTS = [
    "scaler.joblib",
    "gbr_model.joblib",
    "features.joblib",
    "magpie_features.joblib",
]


class ArtifactError(RuntimeError):
    """Raised when model artifacts are missing or invalid."""


def normalize_formula(formula: str) -> str:
    return str(formula).strip().replace(" ", "")


def parse_compositions_from_sources(uploaded_df: pd.DataFrame | None, typed_formula: str | None) -> list[str]:
    compositions: list[str] = []

    if uploaded_df is not None:
        if "Composition" not in uploaded_df.columns:
            raise ValueError("Uploaded CSV must include a 'Composition' column.")
        csv_comps = (
            uploaded_df["Composition"]
            .dropna()
            .astype(str)
            .map(normalize_formula)
            .tolist()
        )
        compositions.extend(csv_comps)

    if typed_formula and typed_formula.strip():
        compositions.append(normalize_formula(typed_formula))

    compositions = [c for c in compositions if c]
    unique = list(dict.fromkeys(compositions))
    if not unique:
        raise ValueError("No valid compositions were provided.")
    return unique


def _load_artifacts(model_dir: str | Path, required_files: Iterable[str]):
    model_dir = Path(model_dir)
    missing = [name for name in required_files if not (model_dir / name).exists()]
    if missing:
        missing_str = ", ".join(missing)
        raise ArtifactError(f"Missing artifact(s) in '{model_dir}': {missing_str}")
    return {name: joblib.load(model_dir / name) for name in required_files}


def load_kappa_artifacts(model_dir: str | Path):
    return _load_artifacts(model_dir, REQUIRED_KAPPA_ARTIFACTS)


def load_cte_artifacts(model_dir: str | Path):
    return _load_artifacts(model_dir, REQUIRED_CTE_ARTIFACTS)


def _build_featurizer() -> MultipleFeaturizer:
    return MultipleFeaturizer([
        ElementProperty.from_preset("magpie"),
        Stoichiometry(),
    ])


def _featurize(compositions: list[str], expected_feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    featurizer = _build_featurizer()
    valid_rows = []
    failed_rows = []

    for formula in compositions:
        try:
            valid_rows.append({"Composition": formula, "composition_obj": Composition(formula)})
        except Exception as exc:  # noqa: BLE001
            failed_rows.append({"Composition": formula, "error": f"Composition parsing failed: {exc}"})

    if not valid_rows:
        raise ValueError("All provided compositions failed to parse.")

    df = pd.DataFrame(valid_rows)
    feat_df = featurizer.featurize_dataframe(
        df,
        col_id="composition_obj",
        ignore_errors=True,
        return_errors=True,
        inplace=False,
    )

    if "MultipleFeaturizer Exceptions" in feat_df.columns:
        err_mask = feat_df["MultipleFeaturizer Exceptions"].notna()
        if err_mask.any():
            failed_rows.extend(
                {
                    "Composition": row["Composition"],
                    "error": f"Featurization failed: {row['MultipleFeaturizer Exceptions']}",
                }
                for _, row in feat_df.loc[err_mask, ["Composition", "MultipleFeaturizer Exceptions"]].iterrows()
            )
        feat_df = feat_df.loc[~err_mask].copy()

    missing = [col for col in expected_feature_cols if col not in feat_df.columns]
    if missing:
        raise ValueError(
            "Generated features are missing model-required columns. "
            f"Examples: {missing[:10]}"
        )

    kept = feat_df[["Composition"] + expected_feature_cols].copy()
    failed_df = pd.DataFrame(failed_rows)
    return kept, failed_df


def _expand_temperature_grid(
    feat_df: pd.DataFrame,
    descriptor_cols: list[str],
    tmin: int = 100,
    tmax: int = 2000,
    step: int = 100,
) -> pd.DataFrame:
    temps = np.arange(tmin, tmax + step, step)
    expanded = feat_df.loc[feat_df.index.repeat(len(temps))].reset_index(drop=True)
    expanded["T"] = np.tile(temps, len(feat_df))
    return expanded[["Composition", "T"] + descriptor_cols]


def predict_kappa(
    compositions: list[str],
    artifacts: dict,
    tmin: int = 100,
    tmax: int = 2000,
    step: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    descriptor_features = artifacts["descriptor_features.joblib"]
    model_features = artifacts["features.joblib"]
    feat_df, failed_df = _featurize(compositions, descriptor_features)
    pred_df = _expand_temperature_grid(feat_df, descriptor_features, tmin=tmin, tmax=tmax, step=step)

    missing = [col for col in model_features if col not in pred_df.columns]
    if missing:
        raise ValueError(f"Prediction dataframe is missing columns required by model: {missing[:10]}")

    X = pred_df[model_features]
    Xs = artifacts["scaler.joblib"].transform(X)
    pred_df["kappa_pred"] = artifacts["rf_model.joblib"].predict(Xs)
    return pred_df[["Composition", "T", "kappa_pred"]], failed_df


def predict_cte(
    compositions: list[str],
    artifacts: dict,
    tmin: int = 100,
    tmax: int = 2000,
    step: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    magpie_features = artifacts["magpie_features.joblib"]
    model_features = artifacts["features.joblib"]
    feat_df, failed_df = _featurize(compositions, magpie_features)
    pred_df = _expand_temperature_grid(feat_df, magpie_features, tmin=tmin, tmax=tmax, step=step)

    missing = [col for col in model_features if col not in pred_df.columns]
    if missing:
        raise ValueError(f"Prediction dataframe is missing columns required by model: {missing[:10]}")

    X = pred_df[model_features]
    Xs = artifacts["scaler.joblib"].transform(X)
    pred_df["cte_pred"] = artifacts["gbr_model.joblib"].predict(Xs)
    return pred_df[["Composition", "T", "cte_pred"]], failed_df


def rank_lowest_kappa_at_temperature(kappa_df: pd.DataFrame, target_t: int = 1500, top_n: int = 10) -> pd.DataFrame:
    required = {"Composition", "T", "kappa_pred"}
    if not required.issubset(kappa_df.columns):
        raise ValueError(f"kappa dataframe missing required columns: {sorted(required - set(kappa_df.columns))}")

    selected = kappa_df[kappa_df["T"] == target_t].copy()
    if selected.empty:
        raise ValueError(f"No kappa predictions found at T={target_t} K")

    ranked = (
        selected.sort_values("kappa_pred", ascending=True)
        .head(top_n)
        .reset_index(drop=True)
    )
    ranked.insert(0, "Rank", np.arange(1, len(ranked) + 1))
    return ranked


def coerce_cte_input(uploaded_cte_df: pd.DataFrame) -> pd.DataFrame:
    required = {"Composition", "T"}
    if not required.issubset(uploaded_cte_df.columns):
        raise ValueError("CTE CSV must include 'Composition' and 'T' columns.")

    if "cte_pred" in uploaded_cte_df.columns:
        cte_col = "cte_pred"
    elif "GBR_Pred" in uploaded_cte_df.columns:
        cte_col = "GBR_Pred"
    else:
        raise ValueError("CTE CSV must include either 'cte_pred' or 'GBR_Pred'.")

    out = uploaded_cte_df[["Composition", "T", cte_col]].copy()
    out = out.rename(columns={cte_col: "cte_pred"})
    return out


def build_final_shortlist(
    kappa_ranked: pd.DataFrame,
    cte_df: pd.DataFrame,
    cte_threshold: float,
    cte_mode: str = "EBC",
) -> pd.DataFrame:
    required_kappa = {"Composition", "T", "kappa_pred"}
    required_cte = {"Composition", "T", "cte_pred"}

    if not required_kappa.issubset(kappa_ranked.columns):
        raise ValueError("Ranked kappa dataframe missing required columns.")
    if not required_cte.issubset(cte_df.columns):
        raise ValueError("CTE dataframe missing required columns.")

    merged = pd.merge(
        kappa_ranked[["Composition", "T", "kappa_pred"]],
        cte_df[["Composition", "T", "cte_pred"]],
        on=["Composition", "T"],
        how="inner",
    )

    normalized_mode = cte_mode.strip().upper()
    if normalized_mode == "EBC":
        merged = merged[merged["cte_pred"] <= cte_threshold].copy()
    elif normalized_mode == "TBC":
        merged = merged[merged["cte_pred"] >= cte_threshold].copy()
    else:
        raise ValueError("CTE mode must be either 'EBC' (max threshold) or 'TBC' (min threshold).")

    # Simple and transparent scoring:
    # lower kappa is always better, so Screening_Score is set to kappa_pred.
    merged["Screening_Score"] = merged["kappa_pred"]
    merged = merged.sort_values(["Screening_Score", "cte_pred"], ascending=[True, True]).reset_index(drop=True)
    merged.insert(0, "Final_Rank", np.arange(1, len(merged) + 1))
    return merged
