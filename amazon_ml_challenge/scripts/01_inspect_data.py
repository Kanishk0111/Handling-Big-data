from pathlib import Path
import sys

import pandas as pd


# Allow imports from src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TRAIN_GROUND_TRUTH,
    TEST_SOURCE1,
    TEST_SOURCE2,
    TEST_SOURCE3,
)


def inspect_file(path: Path, name: str) -> None:
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    if not path.exists():
        print(f"NOT FOUND: {path}")
        return

    print(f"Path: {path}")
    print(f"Size: {path.stat().st_size / (1024 ** 2):.2f} MB")

    # Only read a small sample first.
    df = pd.read_csv(
        path,
        sep="\t",
        nrows=5,
        dtype=str,
        keep_default_na=False,
    )

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nSample:")
    print(df.to_string(index=False))


def main():
    files = [
        (TRAIN_SOURCE1, "TRAIN SOURCE 1"),
        (TRAIN_SOURCE2, "TRAIN SOURCE 2"),
        (TRAIN_SOURCE3, "TRAIN SOURCE 3"),
        (TRAIN_GROUND_TRUTH, "TRAIN GROUND TRUTH"),
        (TEST_SOURCE1, "TEST SOURCE 1"),
        (TEST_SOURCE2, "TEST SOURCE 2"),
        (TEST_SOURCE3, "TEST SOURCE 3"),
    ]

    for path, name in files:
        inspect_file(path, name)


if __name__ == "__main__":
    main()