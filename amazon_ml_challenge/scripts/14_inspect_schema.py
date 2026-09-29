from pathlib import Path
import pandas as pd

from src.config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TRAIN_GROUND_TRUTH,
    TSV_SEPARATOR,
)


FILES = {
    "source1": TRAIN_SOURCE1,
    "source2": TRAIN_SOURCE2,
    "source3": TRAIN_SOURCE3,
    "ground_truth": TRAIN_GROUND_TRUTH,
}


def inspect_file(label, path):
    print("\n" + "=" * 80)
    print(label)
    print(path)
    print("=" * 80)

    # Header + 5 rows only.
    df = pd.read_csv(
        path,
        sep=TSV_SEPARATOR,
        nrows=5,
        dtype=str,
        keep_default_na=False,
    )

    print("\nColumns:")
    for i, col in enumerate(df.columns):
        print(f"  {i}: {col}")

    print("\nSample:")
    print(df.to_string(index=False))

    print("\nDtypes:")
    print(df.dtypes.to_string())


def main():
    for label, path in FILES.items():
        inspect_file(label, path)


if __name__ == "__main__":
    main()