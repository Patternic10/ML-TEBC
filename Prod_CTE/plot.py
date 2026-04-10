import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_compositions(
    input_csv: str,
    compositions: list[str] | None = None,
    model: str = "MLP_Pred",
    temp_col: str = "T",
    output: str | None = None,
):
    df = pd.read_csv(input_csv)

    if "Composition" not in df.columns:
        raise ValueError("CSV must contain 'Composition' column")

    if model not in df.columns:
        raise ValueError(f"Column '{model}' not found in CSV")

    # filter compositions if provided
    if compositions:
        df = df[df["Composition"].isin(compositions)]
        if df.empty:
            raise ValueError("No matching compositions found.")

    plt.figure(figsize=(9, 6))

    for comp, grp in df.groupby("Composition"):
        plt.plot(grp[temp_col], grp[model], marker="o", label=comp)

    plt.xlabel("Temperature (K)")
    plt.ylabel("CTE")
    plt.title(f"CTE vs T ({model})")
    plt.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=300, bbox_inches="tight")
        print(f"[OK] Plot saved to: {output}")

    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Plot CTE predictions from CSV.")
    parser.add_argument("--input-csv", type=str, required=True, help="Prediction CSV")
    parser.add_argument("--model", type=str, default="MLP_Pred", help="Model column (MLP_Pred or RF_Pred)")
    parser.add_argument("--compositions", nargs="*", help="Optional list of compositions to plot")
    parser.add_argument("--output", type=str, help="Optional output image path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    plot_compositions(
        input_csv=args.input_csv,
        compositions=args.compositions,
        model=args.model,
        output=args.output,
    )
