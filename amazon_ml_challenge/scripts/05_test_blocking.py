from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
)

from src.blocking import generate_blocking_keys


def inspect_file(path: Path, name: str):

    print("\n" + "=" * 100)
    print(name)
    print("=" * 100)

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        nrows=10,
    )

    for _, row in df.iterrows():

        keys = generate_blocking_keys(
            row["country"],
            row["business_name"],
            row["business_address"],
        )

        print("\nID:", row["entity_id"])
        print("Name:", row["business_name"])
        print("Address:", row["business_address"])

        for key_name, key_value in keys.items():
            print(f"  {key_name:20s}: {key_value}")


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


if __name__ == "__main__":
    main()