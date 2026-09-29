from pathlib import Path
import random

import pandas as pd

from src.config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
)
from src.normalization import (
    normalize_name,
    compact_text,
    normalize_address,
    token_list,
)


SEED = 42
VALIDATION_SAMPLE_SIZE = 5_000
CHUNK_SIZE = 100_000


COMMON_BUSINESS_TOKENS = {
    "llc", "inc", "incorporated", "corp",
    "corporation", "company", "co",
    "ltd", "limited", "private", "pvt",
    "plc", "llp",
}


def make_block_keys(country, name, address):
    compact = compact_text(name)

    keys = {
        "name_exact": [],
        "compact_name": [],
        "name_pair": [],
        "prefix_house": [],
    }

    if country and name:
        keys["name_exact"].append(
            f"{country}|name_exact|{name}"
        )

    if country and compact:
        keys["compact_name"].append(
            f"{country}|compact_name|{compact}"
        )

    tokens = [
        t
        for t in token_list(name)
        if len(t) >= 3
        and t not in COMMON_BUSINESS_TOKENS
    ]

    unique_tokens = sorted(set(tokens))

    for i in range(len(unique_tokens)):
        for j in range(i + 1, len(unique_tokens)):
            keys["name_pair"].append(
                f"{country}|name_pair|"
                f"{unique_tokens[i]}|{unique_tokens[j]}"
            )

    if country and compact and len(compact) >= 8:
        prefix = compact[:8]

        house_number = ""

        for token in address.split():
            if any(char.isdigit() for char in token):
                house_number = token
                break

        if house_number:
            keys["prefix_house"].append(
                f"{country}|prefix_house|"
                f"{prefix}|{house_number}"
            )

    return keys


def main():

    print("=" * 72)
    print("BLOCKING DIAGNOSTIC")
    print("=" * 72)

    # ------------------------------------------------------------
    # Load validation S1 sample
    # ------------------------------------------------------------

    validation_ids = [
        line.strip()
        for line in Path(
            "output/validation_s1_ids.txt"
        ).read_text().splitlines()
        if line.strip()
    ]

    rng = random.Random(SEED)
    rng.shuffle(validation_ids)

    validation_ids = validation_ids[
        :VALIDATION_SAMPLE_SIZE
    ]

    validation_id_set = set(validation_ids)

    print(
        f"\nValidation S1 sample: "
        f"{len(validation_ids):,}"
    )

    s1_df = pd.read_csv(
        TRAIN_SOURCE1,
        sep="\t",
        dtype=str,
    )

    s1_df = s1_df[
        s1_df["entity_id"].isin(validation_id_set)
    ]

    required_keys = {
        "name_exact": set(),
        "compact_name": set(),
        "name_pair": set(),
        "prefix_house": set(),
    }

    s1_block_counts = {
        "name_exact": 0,
        "compact_name": 0,
        "name_pair": 0,
        "prefix_house": 0,
    }

    for row in s1_df.itertuples(index=False):

        country = (
            str(row.country)
            .strip()
            .casefold()
        )

        name = normalize_name(
            row.business_name
        )

        address = normalize_address(
            row.business_address
        )

        block_keys = make_block_keys(
            country,
            name,
            address,
        )

        for block_type, keys in block_keys.items():

            if keys:
                s1_block_counts[
                    block_type
                ] += 1

                required_keys[
                    block_type
                ].update(keys)

    del s1_df

    print("\nS1 blocking coverage:")
    for block_type, count in s1_block_counts.items():
        print(
            f"  {block_type:15s}: "
            f"{count:,} / {len(validation_ids):,}"
        )

    print("\nRequired unique keys:")
    for block_type, keys in required_keys.items():
        print(
            f"  {block_type:15s}: "
            f"{len(keys):,}"
        )

    # ------------------------------------------------------------
    # Scan S2/S3
    # ------------------------------------------------------------

    matched_entity_ids = {
        "name_exact": set(),
        "compact_name": set(),
        "name_pair": set(),
        "prefix_house": set(),
        "any": set(),
    }

    matched_rows = {
        "name_exact": 0,
        "compact_name": 0,
        "name_pair": 0,
        "prefix_house": 0,
        "any": 0,
    }

    total_rows = 0

    for source_path in [
        TRAIN_SOURCE2,
        TRAIN_SOURCE3,
    ]:

        print(
            f"\nScanning {source_path.name}..."
        )

        for chunk in pd.read_csv(
            source_path,
            sep="\t",
            dtype=str,
            chunksize=CHUNK_SIZE,
        ):

            for row in chunk.itertuples(
                index=False
            ):

                total_rows += 1

                country = (
                    str(row.country)
                    .strip()
                    .casefold()
                )

                name = normalize_name(
                    row.business_name
                )

                address = normalize_address(
                    row.business_address
                )

                block_keys = make_block_keys(
                    country,
                    name,
                    address,
                )

                matched_any = False

                for block_type, keys in block_keys.items():

                    if not keys:
                        continue

                    if any(
                        key in required_keys[block_type]
                        for key in keys
                    ):

                        matched_entity_ids[
                            block_type
                        ].add(row.entity_id)

                        matched_rows[
                            block_type
                        ] += 1

                        matched_any = True

                if matched_any:
                    matched_entity_ids[
                        "any"
                    ].add(row.entity_id)

                    matched_rows[
                        "any"
                    ] += 1

            print(
                f"  scanned: {total_rows:,}"
                f" | any matched: "
                f"{matched_rows['any']:,}"
            )

    # ------------------------------------------------------------
    # Report
    # ------------------------------------------------------------

    print("\n" + "=" * 72)
    print("BLOCKING RESULTS")
    print("=" * 72)

    print(
        f"\nTotal S2/S3 rows scanned: "
        f"{total_rows:,}"
    )

    print("\nRows retained by each block:")
    for block_type in [
        "name_exact",
        "compact_name",
        "name_pair",
        "prefix_house",
        "any",
    ]:

        rows = matched_rows[block_type]

        print(
            f"  {block_type:15s}: "
            f"{rows:,} "
            f"({rows / total_rows:.4%})"
        )

    print("\nUnique entities retained by each block:")
    for block_type in [
        "name_exact",
        "compact_name",
        "name_pair",
        "prefix_house",
        "any",
    ]:

        count = len(
            matched_entity_ids[block_type]
        )

        print(
            f"  {block_type:15s}: "
            f"{count:,}"
        )

    print("\nOverlap with ANY block:")

    any_ids = matched_entity_ids["any"]

    for block_type in [
        "name_exact",
        "compact_name",
        "name_pair",
        "prefix_house",
    ]:

        ids = matched_entity_ids[
            block_type
        ]

        print(
            f"  {block_type:15s}: "
            f"{len(ids & any_ids):,}"
        )


if __name__ == "__main__":
    main()