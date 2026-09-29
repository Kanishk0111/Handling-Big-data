from pathlib import Path
import random
from collections import Counter

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
    token_list,
)


SEED = 42
VALIDATION_SAMPLE_SIZE = 5_000
CHUNK_SIZE = 100_000


COMMON_ADDRESS_TOKENS = {
    "road", "rd", "street", "st",
    "avenue", "ave", "lane", "ln",
    "floor", "fl", "building", "bldg",
    "block", "plot", "house", "no",
    "number", "unit", "flat", "suite",
    "sector", "area", "phase",
    "village", "vill", "town",
}


def address_token_keys(country, address):
    """
    Generate informative address-token blocking keys.

    Very common structural address words are excluded.
    """

    tokens = [
        token
        for token in token_list(address)
        if len(token) >= 4
        and token not in COMMON_ADDRESS_TOKENS
    ]

    return {
        f"{country}|address_token|{token}"
        for token in set(tokens)
        if country
    }


def main():

    print("=" * 72)
    print("ADDRESS-TOKEN RARITY ANALYSIS")
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
    # Required address tokens from S1
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
    ]

    required_address_keys = set()

    for row in s1_df.itertuples(
        index=False
    ):

        country = (
            str(row.country)
            .strip()
            .casefold()
        )

        address = normalize_address(
            row.business_address
        )

        required_address_keys.update(
            address_token_keys(
                country,
                address,
            )
        )

    del s1_df

    print(
        f"Required address-token keys: "
        f"{len(required_address_keys):,}"
    )

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
    # Reverse truth lookup
    # ------------------------------------------------------------

    true_pair_lookup = {}

    for s1_id, true_ids in truth.items():

        for entity_id in true_ids:

            true_pair_lookup.setdefault(
                entity_id,
                set()
            ).add(s1_id)

    # ------------------------------------------------------------
    # Scan source data
    # ------------------------------------------------------------

    token_frequency = Counter()

    true_pair_token_hits = []

    total_scanned = 0

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

                country = (
                    str(row.country)
                    .strip()
                    .casefold()
                )

                address = normalize_address(
                    row.business_address
                )

                keys = (
                    address_token_keys(
                        country,
                        address,
                    )
                    & required_address_keys
                )

                if keys:

                    for key in keys:
                        token_frequency[key] += 1

                # Only true pairs are needed for recall.
                s1_ids = true_pair_lookup.get(
                    row.entity_id
                )

                if s1_ids and keys:

                    for s1_id in s1_ids:

                        for key in keys:

                            true_pair_token_hits.append(
                                (
                                    s1_id,
                                    row.entity_id,
                                    key,
                                )
                            )

            print(
                f"  scanned: "
                f"{total_scanned:,}"
            )

    true_pair_token_hits = set(
        true_pair_token_hits
    )

    print(
        f"\nUnique true pair/token hits: "
        f"{len(true_pair_token_hits):,}"
    )

    # ------------------------------------------------------------
    # Distribution
    # ------------------------------------------------------------

    print("\n" + "=" * 72)
    print("ADDRESS-TOKEN BUCKET DISTRIBUTION")
    print("=" * 72)

    frequencies = list(
        token_frequency.values()
    )

    if frequencies:

        frequencies.sort()

        print(
            f"Min:    {min(frequencies):,}"
        )
        print(
            f"Median: {frequencies[len(frequencies)//2]:,}"
        )
        print(
            f"Max:    {max(frequencies):,}"
        )

    # ------------------------------------------------------------
    # Rarity thresholds
    # ------------------------------------------------------------

    thresholds = [
        5,
        10,
        20,
        25,
        50,
        100,
        200,
        500,
        1000,
    ]

    print("\n" + "=" * 72)
    print("RECALL BY ADDRESS-TOKEN RARITY")
    print("=" * 72)

    print(
        f"{'Max bucket':>12} "
        f"{'Keys':>10} "
        f"{'Approx rows':>15} "
        f"{'True pairs':>12} "
        f"{'Recall':>10}"
    )

    for threshold in thresholds:

        allowed_keys = {
            key
            for key, count
            in token_frequency.items()
            if count <= threshold
        }

        approximate_rows = sum(
            token_frequency[key]
            for key in allowed_keys
        )

        recovered_pairs = {
            (s1_id, entity_id)
            for (
                s1_id,
                entity_id,
                key,
            ) in true_pair_token_hits
            if key in allowed_keys
        }

        recall = (
            len(recovered_pairs)
            / total_true_pairs
            if total_true_pairs
            else 0.0
        )

        print(
            f"{threshold:>12,} "
            f"{len(allowed_keys):>10,} "
            f"{approximate_rows:>15,} "
            f"{len(recovered_pairs):>12,} "
            f"{recall:>9.4%}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()