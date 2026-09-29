from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "output/positive_negative_feature_sample.csv"
)

FEATURE_COLUMNS = [
    "name_exact",
    "name_compact_exact",
    "name_ratio",
    "name_token_set_ratio",
    "name_token_overlap",
    "name_containment",
    "address_exact",
    "address_ratio",
    "address_token_set_ratio",
    "address_token_overlap",
    "address_containment",
]


def main():

    print("=" * 72)
    print("POSITIVE VS CONTROLLED NEGATIVE FEATURE REPORT")
    print("=" * 72)

    df = pd.read_csv(
        INPUT_PATH,
        dtype=str,
        keep_default_na=False,
    )

    print(
        f"Rows: {len(df):,}"
    )

    # -------------------------------------------------------------
    # Explicitly convert feature columns to numeric.
    # -------------------------------------------------------------

    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # -------------------------------------------------------------
    # Check labels
    # -------------------------------------------------------------

    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce",
    )

    print(
        f"Positive rows: "
        f"{(df['label'] == 1).sum():,}"
    )

    print(
        f"Negative rows: "
        f"{(df['label'] == 0).sum():,}"
    )

    # -------------------------------------------------------------
    # Report
    # -------------------------------------------------------------

    print()
    print("=" * 72)
    print("FEATURE COMPARISON")
    print("=" * 72)

    for feature in FEATURE_COLUMNS:

        positive = df.loc[
            df["label"] == 1,
            feature,
        ].dropna()

        negative = df.loc[
            df["label"] == 0,
            feature,
        ].dropna()

        print()
        print(f"{feature}")
        print("-" * 60)

        print(
            f"positive "
            f"mean={positive.mean():.4f} "
            f"median={positive.median():.4f} "
            f"p05={positive.quantile(.05):.4f} "
            f"p95={positive.quantile(.95):.4f}"
        )

        print(
            f"negative "
            f"mean={negative.mean():.4f} "
            f"median={negative.median():.4f} "
            f"p05={negative.quantile(.05):.4f} "
            f"p95={negative.quantile(.95):.4f}"
        )

    # -------------------------------------------------------------
    # Compact comparison table
    # -------------------------------------------------------------

    rows = []

    for feature in FEATURE_COLUMNS:

        pos = df.loc[
            df["label"] == 1,
            feature,
        ].dropna()

        neg = df.loc[
            df["label"] == 0,
            feature,
        ].dropna()

        rows.append(
            {
                "feature": feature,

                "positive_mean":
                    pos.mean(),

                "positive_median":
                    pos.median(),

                "positive_p05":
                    pos.quantile(.05),

                "positive_p95":
                    pos.quantile(.95),

                "negative_mean":
                    neg.mean(),

                "negative_median":
                    neg.median(),

                "negative_p05":
                    neg.quantile(.05),

                "negative_p95":
                    neg.quantile(.95),

                "mean_gap":
                    pos.mean() - neg.mean(),
            }
        )

    summary = pd.DataFrame(rows)

    print()
    print("=" * 72)
    print("COMPACT SUMMARY")
    print("=" * 72)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


if __name__ == "__main__":
    main()