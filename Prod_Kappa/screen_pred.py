#!/usr/bin/env python3
"""
screen_kappa.py

Rank predicted kappa results from predict_kappa.py.

Examples:

Lowest kappa at 1500 K:
python screen_kappa.py \
    --input-csv pred_kappa.csv \
    --output-csv lowest_kappa_1500K.csv \
    --mode lowest_at_T \
    --target-T 1500 \
    --top-n 10 \
    --plot

Lowest mean kappa across temperature range:
python screen_kappa.py \
    --input-csv pred_kappa.csv \
    --output-csv lowest_mean_kappa.csv \
    --mode lowest_mean \
    --top-n 10 \
    --plot
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt


def load_predictions(input_csv):
    df = pd.read_csv(input_csv)

    required = {"Composition", "T", "RF_Pred"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def rank_at_T(df, target_T, top_n, ascending):
    df_T = df[df["T"] == target_T].copy()

    if df_T.empty:
        raise ValueError(f"No rows found at T = {target_T}")

    ranked = (
        df_T.sort_values("RF_Pred", ascending=ascending)
        .head(top_n)
        [["Composition", "T", "RF_Pred"]]
        .reset_index(drop=True)
    )
    return ranked


def rank_aggregate(df, mode, top_n, ascending):
    grouped = df.groupby("Composition")["RF_Pred"]

    if mode == "max":
        ranked = grouped.max().reset_index(name="Score")
    elif mode == "mean":
        ranked = grouped.mean().reset_index(name="Score")
    elif mode == "min":
        ranked = grouped.min().reset_index(name="Score")
    else:
        raise ValueError("Invalid aggregate mode")

    ranked = ranked.sort_values("Score", ascending=ascending).head(top_n).reset_index(drop=True)
    return ranked


def plot_results(full_df, selected):
    plt.figure(figsize=(9, 6))

    comps = selected["Composition"].unique()

    for comp in comps:
        grp = full_df[full_df["Composition"] == comp].sort_values("T")
        plt.plot(grp["T"], grp["RF_Pred"], marker="o", label=comp)

    plt.xlabel("Temperature (K)")
    plt.ylabel("kappa")
    plt.title("Selected compositions")
    plt.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "highest_at_T", "lowest_at_T",
            "highest_max", "lowest_max",
            "highest_mean", "lowest_mean",
            "highest_min", "lowest_min",
        ]
    )
    parser.add_argument("--target-T", type=float, default=None)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--plot", action="store_true")

    args = parser.parse_args()

    df = load_predictions(args.input_csv)

    if args.mode == "highest_at_T":
        if args.target_T is None:
            raise ValueError("Need --target-T")
        ranked = rank_at_T(df, args.target_T, args.top_n, ascending=False)

    elif args.mode == "lowest_at_T":
        if args.target_T is None:
            raise ValueError("Need --target-T")
        ranked = rank_at_T(df, args.target_T, args.top_n, ascending=True)

    elif args.mode == "highest_max":
        ranked = rank_aggregate(df, "max", args.top_n, ascending=False)

    elif args.mode == "lowest_max":
        ranked = rank_aggregate(df, "max", args.top_n, ascending=True)

    elif args.mode == "highest_mean":
        ranked = rank_aggregate(df, "mean", args.top_n, ascending=False)

    elif args.mode == "lowest_mean":
        ranked = rank_aggregate(df, "mean", args.top_n, ascending=True)

    elif args.mode == "highest_min":
        ranked = rank_aggregate(df, "min", args.top_n, ascending=False)

    elif args.mode == "lowest_min":
        ranked = rank_aggregate(df, "min", args.top_n, ascending=True)

    else:
        raise ValueError("Invalid mode")

    ranked.to_csv(args.output_csv, index=False)
    print(ranked)

    if args.plot:
        plot_results(df, ranked)


if __name__ == "__main__":
    main()
