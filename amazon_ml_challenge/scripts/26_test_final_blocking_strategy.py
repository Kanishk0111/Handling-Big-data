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
    compact_text,
    token_list,
)


SEED = 42
VALIDATION_SAMPLE_SIZE = 5_000
CHUNK_SIZE = 100_000

PAIR_MAX_BUCKET = 500


COMMON_BUSINESS_TOKENS = {
    "llc", "inc", "incorporated", "corp",
    "corporation", "company", "co",
    "ltd", "limited", "private", "pvt",
    "plc", "llp",
}


def components(country, name, address):

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
    print("FINAL BLOCKING STRATEGY TEST")
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
    # Load S1 blocking keys
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

    s1_components = {}

    required_keys = {
        "name_exact": set(),
        "compact_name": set(),
        "name_pair": set(),
        "prefix_house": set(),
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

        c = components(
            country,
            name,
            address,
        )

        s1_components[row.entity_id] = c

        for block_type, keys in c.items():
            required_keys[block_type].update(keys)

    del s1_df

    print("\nRequired keys:")

    for block_type, keys in required_keys.items():
        print(
            f"  {block_type:15s}: "
            f"{len(keys):,}"
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

    for row in gt.itertuples(index=False):

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
        f"\nTrue validation pairs: "
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
    # First determine name-pair frequencies
    # ------------------------------------------------------------

    pair_frequency = Counter()

    total_scanned = 0

    for source_path in [
        TRAIN_SOURCE2,
        TRAIN_SOURCE3,
    ]:

        print(
            f"\nFrequency scan: "
            f"{source_path.name}"
        )

        for chunk in pd.read_csv(
            source_path,
            sep="\t",
            dtype=str,
            chunksize=CHUNK_SIZE,
        ):

            for row in chunk.itertuples(index=False):

                total_scanned += 1

                country = (
                    str(row.country)
                    .strip()
                    .casefold()
                )

                name = normalize_name(
                    row.business_name
                )

                keys = (
                    components(
                        country,
                        name,
                        ""
                    )["name_pair"]
                    & required_keys["name_pair"]
                )

                for key in keys:
                    pair_frequency[key] += 1

            print(
                f"  scanned: "
                f"{total_scanned:,}"
            )

    allowed_pair_keys = {
        key
        for key, count in pair_frequency.items()
        if count <= PAIR_MAX_BUCKET
    }

    print(
        f"\nName-pair keys allowed at "
        f"bucket <= {PAIR_MAX_BUCKET}: "
        f"{len(allowed_pair_keys):,}"
    )

    print(
        f"Approx rows from allowed pairs: "
        f"{sum(pair_frequency[k] for k in allowed_pair_keys):,}"
    )

    # ------------------------------------------------------------
    # Second scan:
    # candidate volume + true recall
    # ------------------------------------------------------------

    strategies = {
        "baseline": set(),
        "rarity_500": set(),
        "unrestricted": set(),
    }

    candidate_rows = {
        "baseline": 0,
        "rarity_500": 0,
        "unrestricted": 0,
    }

    true_recovered = {
        "baseline": set(),
        "rarity_500": set(),
        "unrestricted": set(),
    }

    total_scanned = 0

    for source_path in [
        TRAIN_SOURCE2,
        TRAIN_SOURCE3,
    ]:

        print(
            f"\nEvaluation scan: "
            f"{source_path.name}"
        )

        for chunk in pd.read_csv(
            source_path,
            sep="\t",
            dtype=str,
            chunksize=CHUNK_SIZE,
        ):

            for row in chunk.itertuples(index=False):

                total_scanned += 1

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

                c = components(
                    country,
                    name,
                    address,
                )

                exact_hit = bool(
                    c["name_exact"]
                    & required_keys["name_exact"]
                )

                compact_hit = bool(
                    c["compact_name"]
                    & required_keys["compact_name"]
                )

                prefix_house_hit = bool(
                    c["prefix_house"]
                    & required_keys["prefix_house"]
                )

                pair_hit_any = bool(
                    c["name_pair"]
                    & required_keys["name_pair"]
                )

                pair_hit_500 = bool(
                    c["name_pair"]
                    & allowed_pair_keys
                )

                baseline_hit = (
                    exact_hit
                    or compact_hit
                    or prefix_house_hit
                )

                rarity_hit = (
                    baseline_hit
                    or pair_hit_500
                )

                unrestricted_hit = (
                    baseline_hit
                    or pair_hit_any
                )

                entity_id = row.entity_id

                if baseline_hit:
                    candidate_rows["baseline"] += 1

                    s1_ids = true_pair_lookup.get(
                        entity_id,
                        set()
                    )

                    for s1_id in s1_ids:
                        true_recovered["baseline"].add(
                            (s1_id, entity_id)
                        )

                if rarity_hit:
                    candidate_rows["rarity_500"] += 1

                    s1_ids = true_pair_lookup.get(
                        entity_id,
                        set()
                    )

                    for s1_id in s1_ids:
                        true_recovered["rarity_500"].add(
                            (s1_id, entity_id)
                        )

                if unrestricted_hit:
                    candidate_rows["unrestricted"] += 1

                    s1_ids = true_pair_lookup.get(
                        entity_id,
                        set()
                    )

                    for s1_id in s1_ids:
                        true_recovered["unrestricted"].add(
                            (s1_id, entity_id)
                        )

            print(
                f"  scanned: "
                f"{total_scanned:,}"
            )

    # ------------------------------------------------------------
    # Report
    # ------------------------------------------------------------

    print("\n" + "=" * 72)
    print("FINAL BLOCKING COMPARISON")
    print("=" * 72)

    print(
        f"\n{'Strategy':20s}"
        f"{'Rows':>15s}"
        f"{'Recall':>15s}"
    )

    for strategy in [
        "baseline",
        "rarity_500",
        "unrestricted",
    ]:

        recall = (
            len(true_recovered[strategy])
            / total_true_pairs
        )

        print(
            f"{strategy:20s}"
            f"{candidate_rows[strategy]:15,}"
            f"{recall:14.4%}"
        )

    print("\nAdditional recall from rarity_500:")
    print(
        len(
            true_recovered["rarity_500"]
            - true_recovered["baseline"]
        )
        / total_true_pairs
    )

    print("\nAdditional recall from unrestricted:")
    print(
        len(
            true_recovered["unrestricted"]
            - true_recovered["rarity_500"]
        )
        / total_true_pairs
    )


if __name__ == "__main__":
    main()