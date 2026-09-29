from pathlib import Path
import sys
from collections import Counter

import pandas as pd


# ---------------------------------------------------------------------
# Project import
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

CHUNK_SIZE = 50_000


# ---------------------------------------------------------------------
# Generic source profiler
# ---------------------------------------------------------------------

def profile_source(path: Path, name: str) -> None:
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    if not path.exists():
        print(f"NOT FOUND: {path}")
        return

    print(f"File: {path}")
    print(f"Size: {path.stat().st_size / (1024 ** 2):.2f} MB")
    print(f"Chunk size: {CHUNK_SIZE:,}")

    row_count = 0

    country_counts = Counter()

    empty_name = 0
    empty_address = 0
    empty_country = 0

    leading_space_id = 0
    trailing_space_id = 0

    leading_space_name = 0
    trailing_space_name = 0

    leading_space_address = 0
    trailing_space_address = 0

    max_name_length = 0
    max_address_length = 0

    recovered_address_token = 0
    recovered_all_with_address = 0

    sample_rows = []

    reader = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    )

    for chunk_number, df in enumerate(reader, start=1):

        row_count += len(df)

        # -------------------------------------------------------------
        # Keep only a tiny sample
        # -------------------------------------------------------------

        if len(sample_rows) < 5:
            sample_rows.extend(
                df.head(5 - len(sample_rows)).to_dict("records")
            )

        # -------------------------------------------------------------
        # Country
        # -------------------------------------------------------------

        country_counts.update(df["country"])

        # -------------------------------------------------------------
        # Empty values
        # -------------------------------------------------------------

        empty_name += (df["business_name"].str.strip() == "").sum()
        empty_address += (df["business_address"].str.strip() == "").sum()
        empty_country += (df["country"].str.strip() == "").sum()

        # -------------------------------------------------------------
        # Whitespace diagnostics
        # -------------------------------------------------------------

        leading_space_id += (
            df["entity_id"] != df["entity_id"].str.lstrip()
        ).sum()

        trailing_space_id += (
            df["entity_id"] != df["entity_id"].str.rstrip()
        ).sum()

        leading_space_name += (
            df["business_name"] != df["business_name"].str.lstrip()
        ).sum()

        trailing_space_name += (
            df["business_name"] != df["business_name"].str.rstrip()
        ).sum()

        leading_space_address += (
            df["business_address"] != df["business_address"].str.lstrip()
        ).sum()

        trailing_space_address += (
            df["business_address"] != df["business_address"].str.rstrip()
        ).sum()

        # -------------------------------------------------------------
        # String lengths
        # -------------------------------------------------------------

        if len(df):
            max_name_length = max(
                max_name_length,
                int(df["business_name"].str.len().max()),
            )

            max_address_length = max(
                max_address_length,
                int(df["business_address"].str.len().max()),
            )

        # -------------------------------------------------------------
        # Progress
        # -------------------------------------------------------------

        if chunk_number % 10 == 0:
            print(
                f"  processed {row_count:,} rows...",
                flush=True,
            )

    # -----------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------

    print(f"\nTotal rows: {row_count:,}")

    print("\nCountries:")
    for country, count in country_counts.most_common():
        print(f"  {repr(country):20s} {count:,}")

    print("\nEmpty fields:")
    print(f"  business_name:    {empty_name:,}")
    print(f"  business_address: {empty_address:,}")
    print(f"  country:          {empty_country:,}")

    print("\nWhitespace issues:")
    print(f"  entity_id leading:       {leading_space_id:,}")
    print(f"  entity_id trailing:      {trailing_space_id:,}")
    print(f"  business_name leading:   {leading_space_name:,}")
    print(f"  business_name trailing:  {trailing_space_name:,}")
    print(f"  business_address leading:{leading_space_address:,}")
    print(f"  business_address trailing:{trailing_space_address:,}")

    print("\nMaximum string lengths:")
    print(f"  business_name:    {max_name_length}")
    print(f"  business_address: {max_address_length}")

    print("\nFirst few records:")
    for row in sample_rows:
        print(row)


# ---------------------------------------------------------------------
# Ground truth profiler
# ---------------------------------------------------------------------

def profile_ground_truth(path: Path) -> None:
    print("\n" + "=" * 80)
    print("TRAIN GROUND TRUTH")
    print("=" * 80)

    if not path.exists():
        print(f"NOT FOUND: {path}")
        return

    total_rows = 0
    singleton_count = 0
    matched_count = 0

    match_count_distribution = Counter()

    reader = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    )

    for df in reader:

        total_rows += len(df)

        for value in df["matched_entity_ids"]:

            value = value.strip()

            if not value:
                singleton_count += 1
                match_count_distribution[0] += 1
                continue

            ids = [
                x.strip()
                for x in value.split(",")
                if x.strip()
            ]

            n = len(ids)

            matched_count += 1
            match_count_distribution[n] += 1

    print(f"Total S1 rows: {total_rows:,}")
    print(f"S1 entities with >=1 match: {matched_count:,}")
    print(f"S1 singletons: {singleton_count:,}")

    print("\nNumber of matches per S1:")
    for n, count in sorted(match_count_distribution.items()):
        print(f"  {n:4d} matches -> {count:,} S1 entities")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    sources = [
        (TRAIN_SOURCE1, "TRAIN SOURCE 1"),
        (TRAIN_SOURCE2, "TRAIN SOURCE 2"),
        (TRAIN_SOURCE3, "TRAIN SOURCE 3"),
        (TEST_SOURCE1, "TEST SOURCE 1"),
        (TEST_SOURCE2, "TEST SOURCE 2"),
        (TEST_SOURCE3, "TEST SOURCE 3"),
    ]

    for path, name in sources:
        profile_source(path, name)

    profile_ground_truth(TRAIN_GROUND_TRUTH)


if __name__ == "__main__":
    main()