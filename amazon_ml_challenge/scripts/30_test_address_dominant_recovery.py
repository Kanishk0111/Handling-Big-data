from pathlib import Path
import random

import pandas as pd
from rapidfuzz.fuzz import ratio, token_set_ratio

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
SAMPLE_SIZE = 5_000
CHUNK_SIZE = 100_000

NAME_PAIR_THRESHOLD = 500
ADDRESS_TOKEN_THRESHOLD = 100


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
    return str(value).strip().casefold()


def name_pair_keys(country, name):
    tokens = [
        x for x in token_list(name)
        if len(x) >= 3
        and x not in COMMON_BUSINESS_TOKENS
    ]

    tokens = sorted(set(tokens))

    if len(tokens) < 2:
        return set()

    return {
        f"{country}|name_pair|{tokens[i]}|{tokens[j]}"
        for i in range(len(tokens))
        for j in range(i + 1, len(tokens))
    }


def address_token_keys(country, address):
    tokens = [
        x for x in token_list(address)
        if len(x) >= 4
        and x not in COMMON_ADDRESS_TOKENS
    ]

    return {
        f"{country}|address_token|{x}"
        for x in set(tokens)
    }


def prefix_house_key(country, name, address):
    compact = compact_text(name)

    if len(compact) < 8:
        return None

    for token in normalize_address(address).split():
        if any(c.isdigit() for c in token):
            return (
                f"{country}|prefix_house|"
                f"{compact[:8]}|{token}"
            )

    return None


def token_overlap(a, b):
    a = set(token_list(a))
    b = set(token_list(b))

    if not a or not b:
        return 0.0

    return len(a & b) / min(len(a), len(b))


def containment(a, b):
    a = set(token_list(a))
    b = set(token_list(b))

    if not a or not b:
        return 0.0

    return len(a & b) / min(len(a), len(b))


def main():

    print("=" * 72)
    print("ADDRESS-DOMINANT RECOVERY")
    print("=" * 72)

    # ============================================================
    # 1. Validation sample
    # ============================================================

    ids = [
        x.strip()
        for x in Path(
            "output/validation_s1_ids.txt"
        ).read_text().splitlines()
        if x.strip()
    ]

    rng = random.Random(SEED)
    rng.shuffle(ids)

    ids = ids[:SAMPLE_SIZE]
    id_set = set(ids)

    print(
        f"\nValidation S1 sample: {len(ids):,}"
    )

    # ============================================================
    # 2. Load S1
    # ============================================================

    s1 = pd.read_csv(
        TRAIN_SOURCE1,
        sep="\t",
        dtype=str,
    )

    s1 = s1[
        s1["entity_id"].isin(id_set)
    ]

    s1_lookup = {}

    required_exact = set()
    required_compact = set()
    required_pairs = set()
    required_prefix_house = set()
    required_address_tokens = set()

    for row in s1.itertuples(index=False):

        country = country_key(row.country)
        name = normalize_name(row.business_name)
        address = normalize_address(
            row.business_address
        )

        s1_lookup[row.entity_id] = {
            "country": country,
            "name": name,
            "address": address,
        }

        required_exact.add(
            f"{country}|name|{name}"
        )

        compact = compact_text(name)

        if compact:
            required_compact.add(
                f"{country}|compact_name|{compact}"
            )

        required_pairs.update(
            name_pair_keys(country, name)
        )

        prefix = prefix_house_key(
            country,
            name,
            address,
        )

        if prefix:
            required_prefix_house.add(prefix)

        required_address_tokens.update(
            address_token_keys(
                country,
                address,
            )
        )

    del s1

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

        if s1_id not in id_set:
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
        f"True pairs: {total_true_pairs:,}"
    )

    # ============================================================
    # 4. Frequency pass
    # ============================================================

    name_pair_frequency = {}
    address_token_frequency = {}

    print("\nPASS 1: blocking frequencies")

    scanned = 0

    for source_path in [
        TRAIN_SOURCE2,
        TRAIN_SOURCE3,
    ]:

        for chunk in pd.read_csv(
            source_path,
            sep="\t",
            dtype=str,
            chunksize=CHUNK_SIZE,
        ):

            for row in chunk.itertuples(index=False):

                scanned += 1

                country = country_key(row.country)
                name = normalize_name(row.business_name)
                address = normalize_address(
                    row.business_address
                )

                for key in (
                    name_pair_keys(country, name)
                    & required_pairs
                ):
                    name_pair_frequency[key] = (
                        name_pair_frequency.get(key, 0)
                        + 1
                    )

                for key in (
                    address_token_keys(country, address)
                    & required_address_tokens
                ):
                    address_token_frequency[key] = (
                        address_token_frequency.get(key, 0)
                        + 1
                    )

        print(
            f"  {source_path.name}: done"
        )

    allowed_name_pairs = {
        k
        for k, v in name_pair_frequency.items()
        if v <= NAME_PAIR_THRESHOLD
    }

    allowed_address_tokens = {
        k
        for k, v in address_token_frequency.items()
        if v <= ADDRESS_TOKEN_THRESHOLD
    }

    # ============================================================
    # 5. Second pass
    #
    # For every source entity:
    #   - determine baseline block membership
    #   - if not baseline, calculate address similarities against
    #     validation S1 records that share an address token
    #
    # Only pairs with a shared relevant address token are scored.
    # ============================================================

    baseline_candidates = set()
    baseline_true_pairs = set()

    recovery_candidates = {
        "A": set(),
        "B": set(),
        "C": set(),
    }

    recovery_true_pairs = {
        "A": set(),
        "B": set(),
        "C": set(),
    }

    # Map address-token key -> validation S1 IDs.
    address_key_to_s1 = {}

    for s1_id, row in s1_lookup.items():

        keys = (
            address_token_keys(
                row["country"],
                row["address"],
            )
            & allowed_address_tokens
        )

        for key in keys:
            address_key_to_s1.setdefault(
                key,
                set(),
            ).add(s1_id)

    # Source entities that share a rare address token with
    # at least one validation S1.
    #
    # We only need to retain the S1 candidates for those source
    # entities, keeping memory bounded.
    source_to_s1 = {}

    print("\nPASS 2: candidate + address recovery")

    scanned = 0

    for source_path in [
        TRAIN_SOURCE2,
        TRAIN_SOURCE3,
    ]:

        for chunk in pd.read_csv(
            source_path,
            sep="\t",
            dtype=str,
            chunksize=CHUNK_SIZE,
        ):

            for row in chunk.itertuples(index=False):

                scanned += 1

                entity_id = row.entity_id
                country = country_key(row.country)
                name = normalize_name(row.business_name)
                address = normalize_address(
                    row.business_address
                )

                # ------------------------------------------------
                # Baseline
                # ------------------------------------------------

                baseline_hit = False

                exact_key = (
                    f"{country}|name|{name}"
                )

                if exact_key in required_exact:
                    baseline_hit = True

                if not baseline_hit:

                    compact = compact_text(name)

                    if compact:

                        compact_key = (
                            f"{country}|compact_name|{compact}"
                        )

                        if compact_key in required_compact:
                            baseline_hit = True

                if not baseline_hit:

                    prefix = prefix_house_key(
                        country,
                        name,
                        address,
                    )

                    if (
                        prefix
                        and prefix in required_prefix_house
                    ):
                        baseline_hit = True

                if not baseline_hit:

                    pairs = name_pair_keys(
                        country,
                        name,
                    )

                    if pairs & allowed_name_pairs:
                        baseline_hit = True

                # ------------------------------------------------
                # Address-token candidates
                # ------------------------------------------------

                addr_keys = (
                    address_token_keys(
                        country,
                        address,
                    )
                    & allowed_address_tokens
                )

                matched_s1 = set()

                for key in addr_keys:

                    matched_s1.update(
                        address_key_to_s1.get(
                            key,
                            set(),
                        )
                    )

                # ------------------------------------------------
                # Baseline candidate
                # ------------------------------------------------

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
                                (s1_id, entity_id)
                            )

                # ------------------------------------------------
                # Address-dominant recovery
                # ------------------------------------------------

                if matched_s1:

                    for s1_id in matched_s1:

                        s1_row = s1_lookup[s1_id]

                        address_ratio = (
                            ratio(
                                s1_row["address"],
                                address,
                            ) / 100.0
                        )

                        address_token_ratio = (
                            token_set_ratio(
                                s1_row["address"],
                                address,
                            ) / 100.0
                        )

                        address_overlap = (
                            token_overlap(
                                s1_row["address"],
                                address,
                            )
                        )

                        address_containment = (
                            containment(
                                s1_row["address"],
                                address,
                            )
                        )

                        # ----------------------------------------
                        # A
                        # ----------------------------------------

                        hit_a = (
                            address_token_ratio >= 0.95
                            and address_overlap >= 0.80
                        )

                        # ----------------------------------------
                        # B
                        # ----------------------------------------

                        hit_b = (
                            address_token_ratio >= 0.90
                            and address_overlap >= 0.80
                        )

                        # ----------------------------------------
                        # C
                        # ----------------------------------------

                        hit_c = (
                            address_token_ratio >= 0.90
                            and address_containment >= 0.80
                        )

                        true_s1s = (
                            true_entity_to_s1.get(
                                entity_id,
                                set(),
                            )
                        )

                        if hit_a:

                            recovery_candidates[
                                "A"
                            ].add(entity_id)

                            if s1_id in true_s1s:
                                recovery_true_pairs[
                                    "A"
                                ].add(
                                    (s1_id, entity_id)
                                )

                        if hit_b:

                            recovery_candidates[
                                "B"
                            ].add(entity_id)

                            if s1_id in true_s1s:
                                recovery_true_pairs[
                                    "B"
                                ].add(
                                    (s1_id, entity_id)
                                )

                        if hit_c:

                            recovery_candidates[
                                "C"
                            ].add(entity_id)

                            if s1_id in true_s1s:
                                recovery_true_pairs[
                                    "C"
                                ].add(
                                    (s1_id, entity_id)
                                )

            print(
                f"  {source_path.name}: done"
            )

    # ============================================================
    # 6. Final report
    # ============================================================

    baseline_count = len(
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
        f"\nBaseline:"
    )
    print(
        f"  Candidates: "
        f"{baseline_count:,}"
    )
    print(
        f"  True pairs: "
        f"{len(baseline_true_pairs):,}"
    )
    print(
        f"  Recall: "
        f"{baseline_recall:.4%}"
    )

    print("\n" + "-" * 72)

    print(
        f"{'Block':>8} "
        f"{'New candidates':>16} "
        f"{'New true pairs':>16} "
        f"{'Total recall':>14} "
        f"{'Δ recall':>12}"
    )

    for block in ["A", "B", "C"]:

        # Only candidates outside baseline count as genuinely new.
        new_candidates = (
            recovery_candidates[block]
            - baseline_candidates
        )

        combined_true = (
            baseline_true_pairs
            | recovery_true_pairs[block]
        )

        total_recall = (
            len(combined_true)
            / total_true_pairs
        )

        delta = (
            total_recall
            - baseline_recall
        )

        print(
            f"{block:>8} "
            f"{len(new_candidates):>16,} "
            f"{len(recovery_true_pairs[block]):>16,} "
            f"{total_recall:>13.4%} "
            f"{delta:>+11.4%}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()