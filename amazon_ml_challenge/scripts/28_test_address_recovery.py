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

ADDRESS_THRESHOLDS = [
    25,
    50,
    100,
    200,
    500,
    1000,
]

NAME_PAIR_THRESHOLD = 500


COMMON_BUSINESS_TOKENS = {
    "llc", "inc", "incorporated", "corp",
    "corporation", "company", "co",
    "ltd", "limited", "private",
    "pvt", "plc", "llp",
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


def country_key(value):
    if value is None:
        return ""
    return str(value).strip().casefold()


def name_token_pair_keys(country, name):
    country = country_key(country)

    tokens = [
        token
        for token in token_list(name)
        if len(token) >= 3
        and token not in COMMON_BUSINESS_TOKENS
    ]

    if not country or len(tokens) < 2:
        return set()

    tokens = sorted(set(tokens))

    return {
        f"{country}|name_pair|{tokens[i]}|{tokens[j]}"
        for i in range(len(tokens))
        for j in range(i + 1, len(tokens))
    }


def prefix_house_key(country, name, address):
    country = country_key(country)
    compact_name = compact_text(name)

    if not country or len(compact_name) < 8:
        return None

    address = normalize_address(address)

    for token in address.split():
        if any(char.isdigit() for char in token):
            return (
                f"{country}|prefix_house|"
                f"{compact_name[:8]}|{token}"
            )

    return None


def address_token_keys(country, address):
    country = country_key(country)

    tokens = [
        token
        for token in token_list(address)
        if len(token) >= 4
        and token not in COMMON_ADDRESS_TOKENS
    ]

    if not country:
        return set()

    return {
        f"{country}|address_token|{token}"
        for token in set(tokens)
    }


def main():

    print("=" * 72)
    print("ADDRESS RECOVERY OVER NAME-RARITY-500 BLOCKING")
    print("=" * 72)

    # ============================================================
    # 1. Validation S1 sample
    # ============================================================

    validation_ids = [
        x.strip()
        for x in Path(
            "output/validation_s1_ids.txt"
        ).read_text().splitlines()
        if x.strip()
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

    # ============================================================
    # 2. Load validation S1 and build required blocking keys
    # ============================================================

    s1 = pd.read_csv(
        TRAIN_SOURCE1,
        sep="\t",
        dtype=str,
    )

    s1 = s1[
        s1["entity_id"].isin(validation_id_set)
    ]

    required_name_exact = set()
    required_name_compact = set()
    required_name_pairs = set()
    required_prefix_house = set()
    required_address_tokens = set()

    for row in s1.itertuples(index=False):

        country = country_key(row.country)
        name = normalize_name(row.business_name)
        address = normalize_address(
            row.business_address
        )

        if country and name:

            required_name_exact.add(
                f"{country}|name|{name}"
            )

            compact = compact_text(name)

            if compact:
                required_name_compact.add(
                    f"{country}|compact_name|{compact}"
                )

        required_name_pairs.update(
            name_token_pair_keys(
                country,
                name,
            )
        )

        prefix_house = prefix_house_key(
            country,
            name,
            address,
        )

        if prefix_house:
            required_prefix_house.add(
                prefix_house
            )

        required_address_tokens.update(
            address_token_keys(
                country,
                address,
            )
        )

    del s1

    print(
        f"Required exact-name keys: "
        f"{len(required_name_exact):,}"
    )
    print(
        f"Required compact-name keys: "
        f"{len(required_name_compact):,}"
    )
    print(
        f"Required name-pair keys: "
        f"{len(required_name_pairs):,}"
    )
    print(
        f"Required prefix-house keys: "
        f"{len(required_prefix_house):,}"
    )
    print(
        f"Required address-token keys: "
        f"{len(required_address_tokens):,}"
    )

    # ============================================================
    # 3. Ground truth
    # ============================================================

    gt = pd.read_csv(
        TRAIN_GROUND_TRUTH,
        sep="\t",
        dtype=str,
    )

    gt["matched_entity_ids"] = (
        gt["matched_entity_ids"]
        .fillna("")
    )

    true_entity_to_s1 = {}

    total_true_pairs = 0

    for row in gt.itertuples(index=False):

        s1_id = row.source1_entity_id

        if s1_id not in validation_id_set:
            continue

        matches = {
            x.strip()
            for x in row.matched_entity_ids.split(",")
            if x.strip()
        }

        total_true_pairs += len(matches)

        for entity_id in matches:
            true_entity_to_s1.setdefault(
                entity_id,
                set(),
            ).add(s1_id)

    del gt

    print(
        f"True validation pairs: "
        f"{total_true_pairs:,}"
    )

    # ============================================================
    # 4. First source scan:
    #
    #    Calculate frequencies of only the keys relevant to
    #    this 5,000-S1 validation sample.
    # ============================================================

    name_pair_frequency = {}
    address_token_frequency = {}

    total_scanned = 0

    print("\n" + "=" * 72)
    print("PASS 1: BLOCK FREQUENCIES")
    print("=" * 72)

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

                country = country_key(
                    row.country
                )

                name = normalize_name(
                    row.business_name
                )

                address = normalize_address(
                    row.business_address
                )

                # ------------------------------
                # Name-pair frequency
                # ------------------------------

                pair_keys = (
                    name_token_pair_keys(
                        country,
                        name,
                    )
                    & required_name_pairs
                )

                for key in pair_keys:

                    name_pair_frequency[key] = (
                        name_pair_frequency.get(
                            key,
                            0,
                        ) + 1
                    )

                # ------------------------------
                # Address-token frequency
                # ------------------------------

                addr_keys = (
                    address_token_keys(
                        country,
                        address,
                    )
                    & required_address_tokens
                )

                for key in addr_keys:

                    address_token_frequency[key] = (
                        address_token_frequency.get(
                            key,
                            0,
                        ) + 1
                    )

            print(
                f"  scanned: "
                f"{total_scanned:,}"
            )

    # ============================================================
    # 5. Select rarity-filtered keys
    # ============================================================

    allowed_name_pairs = {
        key
        for key, frequency
        in name_pair_frequency.items()
        if frequency <= NAME_PAIR_THRESHOLD
    }

    allowed_address_tokens = {
        threshold: {
            key
            for key, frequency
            in address_token_frequency.items()
            if frequency <= threshold
        }
        for threshold in ADDRESS_THRESHOLDS
    }

    print("\n" + "=" * 72)
    print("RARITY FILTERS")
    print("=" * 72)

    print(
        f"Name-pair <= {NAME_PAIR_THRESHOLD}: "
        f"{len(allowed_name_pairs):,} keys"
    )

    for threshold in ADDRESS_THRESHOLDS:
        print(
            f"Address-token <= {threshold:4}: "
            f"{len(allowed_address_tokens[threshold]):,} keys"
        )

    # ============================================================
    # 6. Second source scan:
    #
    #    Evaluate actual unique candidate source rows.
    #
    #    Baseline:
    #      exact name
    #      OR compact name
    #      OR prefix+house
    #      OR name-pair <= 500
    #
    #    Combined:
    #      baseline
    #      OR address-token <= threshold
    # ============================================================

    baseline_candidates = set()

    combined_candidates = {
        threshold: set()
        for threshold in ADDRESS_THRESHOLDS
    }

    baseline_true_pairs = set()

    combined_true_pairs = {
        threshold: set()
        for threshold in ADDRESS_THRESHOLDS
    }

    total_scanned = 0

    print("\n" + "=" * 72)
    print("PASS 2: CANDIDATE VOLUME + TRUE-PAIR RECOVERY")
    print("=" * 72)

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

                entity_id = row.entity_id

                country = country_key(
                    row.country
                )

                name = normalize_name(
                    row.business_name
                )

                address = normalize_address(
                    row.business_address
                )

                # =================================================
                # Baseline blocking
                # =================================================

                baseline_hit = False

                # Exact name
                if country and name:

                    exact_key = (
                        f"{country}|name|{name}"
                    )

                    if exact_key in required_name_exact:
                        baseline_hit = True

                    # Compact name
                    if not baseline_hit:

                        compact = compact_text(
                            name
                        )

                        if compact:

                            compact_key = (
                                f"{country}|compact_name|"
                                f"{compact}"
                            )

                            if (
                                compact_key
                                in required_name_compact
                            ):
                                baseline_hit = True

                # Prefix + house number
                if not baseline_hit:

                    prefix_house = prefix_house_key(
                        country,
                        name,
                        address,
                    )

                    if (
                        prefix_house
                        and prefix_house
                        in required_prefix_house
                    ):
                        baseline_hit = True

                # Name-pair rarity <= 500
                if not baseline_hit:

                    pair_keys = (
                        name_token_pair_keys(
                            country,
                            name,
                        )
                    )

                    if (
                        pair_keys
                        & allowed_name_pairs
                    ):
                        baseline_hit = True

                # =================================================
                # Address-token rarity
                # =================================================

                addr_keys = (
                    address_token_keys(
                        country,
                        address,
                    )
                    & required_address_tokens
                )

                # =================================================
                # Candidate sets
                # =================================================

                if baseline_hit:

                    baseline_candidates.add(
                        entity_id
                    )

                    true_s1s = (
                        true_entity_to_s1.get(
                            entity_id
                        )
                    )

                    if true_s1s:

                        for s1_id in true_s1s:

                            baseline_true_pairs.add(
                                (
                                    s1_id,
                                    entity_id,
                                )
                            )

                for threshold in ADDRESS_THRESHOLDS:

                    address_hit = bool(
                        addr_keys
                        & allowed_address_tokens[
                            threshold
                        ]
                    )

                    if baseline_hit or address_hit:

                        combined_candidates[
                            threshold
                        ].add(entity_id)

                        true_s1s = (
                            true_entity_to_s1.get(
                                entity_id
                            )
                        )

                        if true_s1s:

                            for s1_id in true_s1s:

                                combined_true_pairs[
                                    threshold
                                ].add(
                                    (
                                        s1_id,
                                        entity_id,
                                    )
                                )

            print(
                f"  scanned: "
                f"{total_scanned:,}"
            )

    # ============================================================
    # 7. Final report
    # ============================================================

    baseline_candidate_count = len(
        baseline_candidates
    )

    baseline_recall = (
        len(baseline_true_pairs)
        / total_true_pairs
    )

    print("\n" + "=" * 72)
    print("RESULT")
    print("=" * 72)

    print(
        f"\nBaseline name-rarity-500:"
    )
    print(
        f"  Candidate rows: "
        f"{baseline_candidate_count:,}"
    )
    print(
        f"  True pairs recovered: "
        f"{len(baseline_true_pairs):,}"
    )
    print(
        f"  Recall: "
        f"{baseline_recall:.4%}"
    )

    print("\n" + "-" * 72)

    print(
        f"{'Address':>12} "
        f"{'Candidates':>14} "
        f"{'Added':>14} "
        f"{'True pairs':>12} "
        f"{'Recall':>10} "
        f"{'Δ Recall':>10}"
    )

    for threshold in ADDRESS_THRESHOLDS:

        candidates = combined_candidates[
            threshold
        ]

        true_pairs = combined_true_pairs[
            threshold
        ]

        candidate_count = len(candidates)

        added_candidates = (
            candidate_count
            - baseline_candidate_count
        )

        recall = (
            len(true_pairs)
            / total_true_pairs
        )

        incremental_recall = (
            recall
            - baseline_recall
        )

        print(
            f"{threshold:>12,} "
            f"{candidate_count:>14,} "
            f"{added_candidates:>14,} "
            f"{len(true_pairs):>12,} "
            f"{recall:>9.4%} "
            f"{incremental_recall:>+9.4%}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()