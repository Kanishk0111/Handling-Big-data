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

COMMON_ADDRESS_TOKENS = {
    "road", "rd", "street", "st",
    "avenue", "ave", "lane", "ln",
    "floor", "fl", "building", "bldg",
    "block", "plot", "house", "no",
    "number", "unit", "flat", "suite",
    "sector", "area", "phase",
    "village", "vill", "town",
}


def get_components(country, name, address):
    """
    Generate reusable blocking components for one record.
    """

    compact = compact_text(name)

    # ------------------------------------------------------------
    # Name pair
    # ------------------------------------------------------------

    name_pair_keys = set()

    tokens = [
        token
        for token in token_list(name)
        if len(token) >= 3
        and token not in COMMON_BUSINESS_TOKENS
    ]

    unique_tokens = sorted(set(tokens))

    for i in range(len(unique_tokens)):
        for j in range(i + 1, len(unique_tokens)):
            name_pair_keys.add(
                f"{country}|name_pair|"
                f"{unique_tokens[i]}|{unique_tokens[j]}"
            )

    # ------------------------------------------------------------
    # Address prefix
    # ------------------------------------------------------------

    address_tokens = address.split()

    address_prefix = ""

    if len(address_tokens) >= 3:
        address_prefix = (
            f"{country}|address_prefix|"
            f"{address_tokens[0]}|"
            f"{address_tokens[1]}|"
            f"{address_tokens[2]}"
        )

    # ------------------------------------------------------------
    # House number
    # ------------------------------------------------------------

    house_number = ""

    for token in address_tokens:
        if any(char.isdigit() for char in token):
            house_number = (
                f"{country}|house_number|{token}"
            )
            break

    # ------------------------------------------------------------
    # Name prefix
    # ------------------------------------------------------------

    name_prefix = ""

    if len(compact) >= 8:
        name_prefix = (
            f"{country}|name_prefix|"
            f"{compact[:8]}"
        )

    return {
        "name_pair": name_pair_keys,
        "address_prefix": (
            {address_prefix}
            if address_prefix
            else set()
        ),
        "house_number": (
            {house_number}
            if house_number
            else set()
        ),
        "name_prefix": (
            {name_prefix}
            if name_prefix
            else set()
        ),
    }


def make_composite_signals(components):
    """
    Return booleans for the composite blocking strategies.
    """

    name_pairs = components["name_pair"]
    address_prefix = components["address_prefix"]
    house_number = components["house_number"]
    name_prefix = components["name_prefix"]

    return {
        "name_pair": bool(name_pairs),

        "name_pair_and_address_prefix": (
            bool(name_pairs)
            and bool(address_prefix)
        ),

        "name_pair_and_house_number": (
            bool(name_pairs)
            and bool(house_number)
        ),

        "name_pair_and_address_signal": (
            bool(name_pairs)
            and (
                bool(address_prefix)
                or bool(house_number)
            )
        ),

        "name_prefix_and_address_prefix": (
            bool(name_prefix)
            and bool(address_prefix)
        ),
    }


def keys_intersect(a, b):
    return bool(a & b)


def main():

    print("=" * 72)
    print("COMPOSITE BLOCKING ANALYSIS")
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
    # Load S1
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

    s1_components = {}

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

        s1_components[row.entity_id] = (
            get_components(
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

        truth[row.source1_entity_id] = {
            x.strip()
            for x in row.matched_entity_ids.split(",")
            if x.strip()
        }

    total_true_pairs = sum(
        len(ids)
        for ids in truth.values()
    )

    print(
        f"True validation pairs: "
        f"{total_true_pairs:,}"
    )

    # ------------------------------------------------------------
    # Reverse ground-truth lookup
    # ------------------------------------------------------------

    true_pair_lookup = {}

    for s1_id, true_ids in truth.items():

        for entity_id in true_ids:

            true_pair_lookup.setdefault(
                entity_id,
                set()
            ).add(s1_id)

    # ------------------------------------------------------------
    # Track recovered pairs
    # ------------------------------------------------------------

    strategies = [
        "name_pair",
        "name_pair_and_address_prefix",
        "name_pair_and_house_number",
        "name_pair_and_address_signal",
        "name_prefix_and_address_prefix",
    ]

    recovered = {
        strategy: set()
        for strategy in strategies
    }

    total_scanned = 0

    # ------------------------------------------------------------
    # Scan S2/S3
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

                components = get_components(
                    country,
                    name,
                    address,
                )

                for s1_id in s1_ids:

                    required = s1_components[
                        s1_id
                    ]

                    name_pair_hit = (
                        keys_intersect(
                            components["name_pair"],
                            required["name_pair"],
                        )
                    )

                    address_prefix_hit = (
                        keys_intersect(
                            components["address_prefix"],
                            required["address_prefix"],
                        )
                    )

                    house_number_hit = (
                        keys_intersect(
                            components["house_number"],
                            required["house_number"],
                        )
                    )

                    name_prefix_hit = (
                        keys_intersect(
                            components["name_prefix"],
                            required["name_prefix"],
                        )
                    )

                    pair = (
                        s1_id,
                        row.entity_id,
                    )

                    if name_pair_hit:
                        recovered[
                            "name_pair"
                        ].add(pair)

                    if (
                        name_pair_hit
                        and address_prefix_hit
                    ):
                        recovered[
                            "name_pair_and_address_prefix"
                        ].add(pair)

                    if (
                        name_pair_hit
                        and house_number_hit
                    ):
                        recovered[
                            "name_pair_and_house_number"
                        ].add(pair)

                    if (
                        name_pair_hit
                        and (
                            address_prefix_hit
                            or house_number_hit
                        )
                    ):
                        recovered[
                            "name_pair_and_address_signal"
                        ].add(pair)

                    if (
                        name_prefix_hit
                        and address_prefix_hit
                    ):
                        recovered[
                            "name_prefix_and_address_prefix"
                        ].add(pair)

            print(
                f"  scanned: "
                f"{total_scanned:,}"
            )

    # ------------------------------------------------------------
    # Recall report
    # ------------------------------------------------------------

    print("\n" + "=" * 72)
    print("COMPOSITE BLOCKING RECALL")
    print("=" * 72)

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
            f"{strategy:35s}: "
            f"{count:8,} / "
            f"{total_true_pairs:,} "
            f"= {recall:.4%}"
        )

    # ------------------------------------------------------------
    # Incremental recall relative to
    # exact + compact + prefix_house
    # ------------------------------------------------------------

    # Reconstruct the baseline true-pair set.
    baseline = set()

    # This requires the previously measured
    # baseline components to be recomputed
    # only for true pairs.

    for entity_id, s1_ids in true_pair_lookup.items():

        # We don't have source records here,
        # so this section intentionally does not
        # estimate baseline incremental recall.
        pass

    print("\nDone.")
    print(
        "Use the recall values above to compare "
        "the composite name-pair strategies."
    )


if __name__ == "__main__":
    main()