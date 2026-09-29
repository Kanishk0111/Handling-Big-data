from pathlib import Path
import random

import pandas as pd

from src.config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TRAIN_GROUND_TRUTH,
)
from src.normalization import (
    normalize_name,
    normalize_address,
    compact_text,
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

    result = {
        "name_exact": set(),
        "compact_name": set(),
        "name_pair": set(),
        "prefix_house": set(),
    }

    if country and name:
        result["name_exact"].add(
            f"{country}|name_exact|{name}"
        )

    if country and compact:
        result["compact_name"].add(
            f"{country}|compact_name|{compact}"
        )

    tokens = [
        token
        for token in token_list(name)
        if len(token) >= 3
        and token not in COMMON_BUSINESS_TOKENS
    ]

    unique_tokens = sorted(set(tokens))

    for i in range(len(unique_tokens)):
        for j in range(i + 1, len(unique_tokens)):
            result["name_pair"].add(
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
            result["prefix_house"].add(
                f"{country}|prefix_house|"
                f"{prefix}|{house_number}"
            )

    return result


def main():

    print("=" * 72)
    print("BLOCKING RECALL ANALYSIS")
    print("=" * 72)

    # ------------------------------------------------------------
    # Validation S1 sample
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

    # ------------------------------------------------------------
    # Load S1 records
    # ------------------------------------------------------------

    s1_df = pd.read_csv(
        TRAIN_SOURCE1,
        sep="\t",
        dtype=str,
    )

    s1_df = s1_df[
        s1_df["entity_id"].isin(
            validation_id_set
        )
    ].copy()

    s1_keys = {}

    for row in s1_df.itertuples(
        index=False
    ):

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

        s1_keys[row.entity_id] = (
            make_block_keys(
                country,
                name,
                address,
            )
        )

    del s1_df

    # ------------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------------

    gt = pd.read_csv(
        TRAIN_GROUND_TRUTH,
        sep="\t",
        dtype=str,
    )

    gt["matched_entity_ids"] = (
        gt["matched_entity_ids"]
        .fillna("")
    )

    truth = {}

    for row in gt.itertuples(
        index=False
    ):

        if row.source1_entity_id not in validation_id_set:
            continue

        truth[
            row.source1_entity_id
        ] = {
            x.strip()
            for x in row.matched_entity_ids.split(",")
            if x.strip()
        }

    # ------------------------------------------------------------
    # Reverse lookup:
    #
    # source2/source3 entity ID -> S1 IDs
    #
    # Only validation true pairs are stored.
    # ------------------------------------------------------------

    true_pair_lookup = {}

    total_true_pairs = 0

    for s1_id, true_ids in truth.items():

        total_true_pairs += len(true_ids)

        for entity_id in true_ids:

            true_pair_lookup.setdefault(
                entity_id,
                set()
            ).add(s1_id)

    print(
        f"True validation pairs: "
        f"{total_true_pairs:,}"
    )

    # ------------------------------------------------------------
    # Track recovered true pairs
    # ------------------------------------------------------------

    recovered = {
        "name_exact": set(),
        "compact_name": set(),
        "name_pair": set(),
        "prefix_house": set(),
        "exact_compact_prefix_house": set(),
        "all": set(),
    }

    total_scanned = 0

    # ------------------------------------------------------------
    # Scan S2/S3 once
    # ------------------------------------------------------------

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

                total_scanned += 1

                s1_ids = true_pair_lookup.get(
                    row.entity_id
                )

                # This is the key optimization:
                # if this source record isn't a
                # ground-truth match for our
                # validation sample, it cannot
                # contribute to recall.
                if not s1_ids:
                    continue

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

                candidate_keys = make_block_keys(
                    country,
                    name,
                    address,
                )

                for s1_id in s1_ids:

                    required = s1_keys[s1_id]

                    hit = {}

                    for block_type in required:

                        hit[block_type] = bool(
                            candidate_keys[
                                block_type
                            ]
                            & required[
                                block_type
                            ]
                        )

                    pair = (
                        s1_id,
                        row.entity_id,
                    )

                    if hit["name_exact"]:
                        recovered[
                            "name_exact"
                        ].add(pair)

                    if hit["compact_name"]:
                        recovered[
                            "compact_name"
                        ].add(pair)

                    if hit["name_pair"]:
                        recovered[
                            "name_pair"
                        ].add(pair)

                    if hit["prefix_house"]:
                        recovered[
                            "prefix_house"
                        ].add(pair)

                    if (
                        hit["name_exact"]
                        or hit["compact_name"]
                        or hit["prefix_house"]
                    ):
                        recovered[
                            "exact_compact_prefix_house"
                        ].add(pair)

                    if any(hit.values()):
                        recovered[
                            "all"
                        ].add(pair)

            print(
                f"  scanned: "
                f"{total_scanned:,}"
            )

    # ------------------------------------------------------------
    # Report
    # ------------------------------------------------------------

    print("\n" + "=" * 72)
    print("BLOCKING RECALL RESULTS")
    print("=" * 72)

    strategies = [
        "name_exact",
        "compact_name",
        "name_pair",
        "prefix_house",
        "exact_compact_prefix_house",
        "all",
    ]

    for strategy in strategies:

        count = len(
            recovered[strategy]
        )

        recall = (
            count / total_true_pairs
            if total_true_pairs
            else 0.0
        )

        print(
            f"{strategy:30s}: "
            f"{count:8,} / "
            f"{total_true_pairs:,} "
            f"= {recall:.4%}"
        )

    # ------------------------------------------------------------
    # Incremental contribution
    # ------------------------------------------------------------

    baseline = recovered[
        "exact_compact_prefix_house"
    ]

    pair_only = recovered[
        "name_pair"
    ]

    pair_incremental = (
        pair_only - baseline
    )

    print("\nIncremental name_pair contribution:")
    print(
        f"  New true pairs recovered by "
        f"name_pair beyond exact/compact/"
        f"prefix_house: "
        f"{len(pair_incremental):,}"
    )

    print(
        f"  Incremental recall: "
        f"{len(pair_incremental) / total_true_pairs:.4%}"
    )


if __name__ == "__main__":
    main()
    