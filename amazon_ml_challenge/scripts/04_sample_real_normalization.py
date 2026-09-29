from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TEST_SOURCE1,
    TEST_SOURCE2,
    TEST_SOURCE3,
)

from src.normalization import (
    normalize_name,
    normalize_address,
)


def inspect_file(path: Path, name: str) -> None:

    print("\n" + "=" * 100)
    print(name)
    print("=" * 100)

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        nrows=20,
    )

    for _, row in df.iterrows():

        print("\nID:", row["entity_id"])
        print("Country:", row["country"])

        print("\nOriginal name:")
        print(repr(row["business_name"]))

        print("Normalized name:")
        print(repr(normalize_name(row["business_name"])))

        print("\nOriginal address:")
        print(repr(row["business_address"]))

        print("Normalized address:")
        print(repr(normalize_address(row["business_address"])))

        print("-" * 100)


def main():

    inspect_file(
        TRAIN_SOURCE1,
        "TRAIN SOURCE 1",
    )

    inspect_file(
        TRAIN_SOURCE2,
        "TRAIN SOURCE 2",
    )

    inspect_file(
        TRAIN_SOURCE3,
        "TRAIN SOURCE 3",
    )

    inspect_file(
        TEST_SOURCE1,
        "TEST SOURCE 1",
    )

    inspect_file(
        TEST_SOURCE2,
        "TEST SOURCE 2",
    )

    inspect_file(
        TEST_SOURCE3,
        "TEST SOURCE 3",
    )


if __name__ == "__main__":
    main()