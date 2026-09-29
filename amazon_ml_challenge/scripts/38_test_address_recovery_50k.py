from pathlib import Path
import random
from collections import defaultdict

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


# ================================================================
# CONFIG
# ================================================================

SEED = 42
SAMPLE_SIZE = 50_000
CHUNK_SIZE = 100_000

NAME_PAIR_THRESHOLD = 500
ADDRESS_TOKEN_THRESHOLD = 100

# Fuzzy recovery is allowed to use a somewhat broader address
# token bucket than the frozen <=100 baseline.
#
# The fuzzy address similarity then filters these candidates.
FUZZY_ADDRESS_TOKEN_THRESHOLD = 1000

# Recovery A
ADDRESS_SET_RATIO_A = 0.90
ADDRESS_OVERLAP_A = 0.80

# Recovery B
ADDRESS_SET_RATIO_B = 0.95
ADDRESS_OVERLAP_B = 0.80

# Recovery C
ADDRESS_SET_RATIO_C = 0.90
NAME_RATIO_C = 0.50


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


# ================================================================
# HELPERS
# ================================================================

def country_key(value):
    return str(value).strip().casefold()


def name_pair_keys(country, name):
    tokens = [
        token
        for token in token_list(name)
        if len(token) >= 3
        and token not in COMMON_BUSINESS_TOKENS
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


def prefix_house_key(country, name, address):
    compact = compact_text(name)

    if len(compact) < 8:
        return None

    for token in address.split():
        if any(char.isdigit() for char in token):
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


def load_validation_sample():
    ids = [
        line.strip()
        for line in Path(
            "output/validation_s1_ids.txt"
        ).read_text().splitlines()
        if line.strip()
    ]

    rng = random.Random(SEED)
    rng.shuffle(ids)

    return ids[:SAMPLE_SIZE]


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 80)
    print("50K ADDRESS-DOMINANT FUZZY RECOVERY EXPERIMENT V2")
    print("=" * 80)

    # ============================================================
    # 1. Validation sample
    # ============================================================

    validation_ids = load_validation_sample()
    validation_id_set = set(validation_ids)

    print(
        f"\nValidation S1 sample: "
        f"{len(validation_ids):,}"
    )

    # ============================================================
    # 2. Load S1
    # ============================================================

    s1_df = pd.read_csv(
        TRAIN_SOURCE1,
        sep="\t",
        dtype=str,
    )

    s1_df = s1_df[
        s1_df["entity_id"].isin(validation_id_set)
    ].copy()

    s1_lookup = {}

    required_name_exact = set()
    required_name_compact = set()
    required_prefix_house = set()
    required_name_pairs = set()
    required_address_tokens = set()

    # Reverse S1 indexes.
    name_exact_to_s1 = defaultdict(set)
    name_compact_to_s1 = defaultdict(set)
    prefix_house_to_s1 = defaultdict(set)
    name_pair_to_s1 = defaultdict(set)

    # Every S1 address token -> S1 IDs.
    address_token_to_s1 = defaultdict(set)

    for row in s1_df.itertuples(index=False):

        s1_id = row.entity_id

        country = country_key(row.country)
        name = normalize_name(row.business_name)
        address = normalize_address(row.business_address)

        s1_lookup[s1_id] = {
            "country": country,
            "name": name,
            "address": address,
        }

        # --------------------------------------------------------
        # Exact name
        # --------------------------------------------------------

        if country and name:

            key = f"{country}|name|{name}"

            required_name_exact.add(key)
            name_exact_to_s1[key].add(s1_id)

        # --------------------------------------------------------
        # Compact name
        # --------------------------------------------------------

        compact = compact_text(name)

        if country and compact:

            key = f"{country}|compact_name|{compact}"

            required_name_compact.add(key)
            name_compact_to_s1[key].add(s1_id)

        # --------------------------------------------------------
        # Prefix + house
        # --------------------------------------------------------

        prefix = prefix_house_key(
            country,
            name,
            address,
        )

        if prefix:

            required_prefix_house.add(prefix)
            prefix_house_to_s1[prefix].add(s1_id)

        # --------------------------------------------------------
        # Name pairs
        # --------------------------------------------------------

        pairs = name_pair_keys(
            country,
            name,
        )

        for key in pairs:

            required_name_pairs.add(key)
            name_pair_to_s1[key].add(s1_id)

        # --------------------------------------------------------
        # Address tokens
        # --------------------------------------------------------

        addr_keys = address_token_keys(
            country,
            address,
        )

        for key in addr_keys:

            required_address_tokens.add(key)
            address_token_to_s1[key].add(s1_id)

    del s1_df

    print(
        f"S1 entities loaded: "
        f"{len(s1_lookup):,}"
    )

    print(
        f"Required name-pair keys: "
        f"{len(required_name_pairs):,}"
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

    true_entity_to_s1 = defaultdict(set)

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

        for source_id in matches:

            true_entity_to_s1[
                source_id
            ].add(s1_id)

    del gt

    print(
        f"True validation pairs: "
        f"{total_true_pairs:,}"
    )

    # ============================================================
    # 4. PASS 1
    #
    # Build frequency tables only for keys relevant to this
    # validation sample.
    # ============================================================

    print("\n" + "=" * 80)
    print("PASS 1: RELEVANT BLOCK FREQUENCIES")
    print("=" * 80)

    name_pair_frequency = defaultdict(int)
    address_token_frequency = defaultdict(int)

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

            for row in chunk.itertuples(index=False):

                total_scanned += 1

                country = country_key(row.country)

                name = normalize_name(
                    row.business_name
                )

                address = normalize_address(
                    row.business_address
                )

                # ------------------------------------------------
                # Name-pair frequencies
                # ------------------------------------------------

                keys = (
                    name_pair_keys(
                        country,
                        name,
                    )
                    & required_name_pairs
                )

                for key in keys:
                    name_pair_frequency[key] += 1

                # ------------------------------------------------
                # Address-token frequencies
                # ------------------------------------------------

                keys = (
                    address_token_keys(
                        country,
                        address,
                    )
                    & required_address_tokens
                )

                for key in keys:
                    address_token_frequency[key] += 1

            print(
                f"  scanned: "
                f"{total_scanned:,}"
            )

    # ------------------------------------------------------------
    # Rarity-filtered keys
    # ------------------------------------------------------------

    allowed_name_pairs = {
        key
        for key, frequency
        in name_pair_frequency.items()
        if frequency <= NAME_PAIR_THRESHOLD
    }

    allowed_address_tokens = {
        key
        for key, frequency
        in address_token_frequency.items()
        if frequency <= ADDRESS_TOKEN_THRESHOLD
    }

    allowed_fuzzy_address_tokens = {
        key
        for key, frequency
        in address_token_frequency.items()
        if frequency <= FUZZY_ADDRESS_TOKEN_THRESHOLD
    }

    print("\n" + "-" * 80)

    print(
        f"Name-pair <= {NAME_PAIR_THRESHOLD}: "
        f"{len(allowed_name_pairs):,} keys"
    )

    print(
        f"Address-token <= {ADDRESS_TOKEN_THRESHOLD}: "
        f"{len(allowed_address_tokens):,} keys"
    )

    print(
        f"Fuzzy address-token <= "
        f"{FUZZY_ADDRESS_TOKEN_THRESHOLD}: "
        f"{len(allowed_fuzzy_address_tokens):,} keys"
    )

    # Free the full frequency dictionaries.
    del name_pair_frequency
    del address_token_frequency

    # ============================================================
    # 5. Candidate sets
    #
    # We store candidate pairs as (s1_id, source_id).
    #
    # For 50k this is acceptable and gives exact incremental
    # candidate counts.
    # ============================================================

    baseline_candidates = set()

    recovery_a_candidates = set()
    recovery_b_candidates = set()
    recovery_c_candidates = set()

    # ============================================================
    # 6. PASS 2
    #
    # Generate:
    #   - frozen baseline
    #   - shared-token fuzzy-address candidates
    #
    # For fuzzy candidates we DO NOT immediately calculate fuzzy
    # similarity. First collect the small shared-token candidate
    # universe.
    # ============================================================

    print("\n" + "=" * 80)
    print("PASS 2: CANDIDATE GENERATION")
    print("=" * 80)

    total_scanned = 0
    shared_token_pairs = 0

    fuzzy_candidate_pairs = set()

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

            for row in chunk.itertuples(index=False):

                total_scanned += 1

                source_id = row.entity_id

                country = country_key(row.country)
                name = normalize_name(row.business_name)
                address = normalize_address(row.business_address)

                # =================================================
                # BASELINE
                # =================================================

                baseline_s1_ids = set()

                # -------------------------------------------------
                # Exact name
                # -------------------------------------------------

                if country and name:

                    key = f"{country}|name|{name}"

                    baseline_s1_ids.update(
                        name_exact_to_s1.get(
                            key,
                            ()
                        )
                    )

                # -------------------------------------------------
                # Compact name
                # -------------------------------------------------

                compact = compact_text(name)

                if country and compact:

                    key = (
                        f"{country}|compact_name|"
                        f"{compact}"
                    )

                    baseline_s1_ids.update(
                        name_compact_to_s1.get(
                            key,
                            ()
                        )
                    )

                # -------------------------------------------------
                # Prefix + house
                # -------------------------------------------------

                prefix = prefix_house_key(
                    country,
                    name,
                    address,
                )

                if prefix:

                    baseline_s1_ids.update(
                        prefix_house_to_s1.get(
                            prefix,
                            ()
                        )
                    )

                # -------------------------------------------------
                # Name pair <= 500
                # -------------------------------------------------

                for key in name_pair_keys(
                    country,
                    name,
                ):

                    if key in allowed_name_pairs:

                        baseline_s1_ids.update(
                            name_pair_to_s1.get(
                                key,
                                ()
                            )
                        )

                # -------------------------------------------------
                # Address token <= 100
                # -------------------------------------------------

                for key in address_token_keys(
                    country,
                    address,
                ):

                    if key in allowed_address_tokens:

                        baseline_s1_ids.update(
                            address_token_to_s1.get(
                                key,
                                ()
                            )
                        )

                for s1_id in baseline_s1_ids:

                    pair = (
                        s1_id,
                        source_id,
                    )

                    baseline_candidates.add(pair)

                # =================================================
                # FUZZY ADDRESS CANDIDATES
                # =================================================

                fuzzy_s1_ids = set()

                for key in address_token_keys(
                    country,
                    address,
                ):

                    if key not in allowed_fuzzy_address_tokens:
                        continue

                    fuzzy_s1_ids.update(
                        address_token_to_s1.get(
                            key,
                            ()
                        )
                    )

                shared_token_pairs += len(fuzzy_s1_ids)

                for s1_id in fuzzy_s1_ids:

                    pair = (
                        s1_id,
                        source_id,
                    )

                    # Don't waste memory on candidates already
                    # captured by the baseline.
                    if pair not in baseline_candidates:

                        fuzzy_candidate_pairs.add(pair)

            print(
                f"  scanned: "
                f"{total_scanned:,}"
            )

    print("\n" + "-" * 80)

    print(
        f"Baseline candidate pairs: "
        f"{len(baseline_candidates):,}"
    )

    print(
        f"New shared-token fuzzy candidates: "
        f"{len(fuzzy_candidate_pairs):,}"
    )

    print(
        f"Raw shared-token comparisons: "
        f"{shared_token_pairs:,}"
    )

    # ============================================================
    # 7. PASS 3
    #
    # Re-read source data and only score source IDs that appear
    # in fuzzy_candidate_pairs.
    #
    # Build:
    #   source_id -> candidate S1 IDs
    #
    # so we never compare a source row against unrelated S1s.
    # ============================================================

    fuzzy_source_to_s1 = defaultdict(list)

    for s1_id, source_id in fuzzy_candidate_pairs:

        fuzzy_source_to_s1[source_id].append(s1_id)

    # ------------------------------------------------------------
    # We no longer need the complete fuzzy pair set.
    # ------------------------------------------------------------

    del fuzzy_candidate_pairs

    recovery_a_true = set()
    recovery_b_true = set()
    recovery_c_true = set()

    recovery_a_candidates = set()
    recovery_b_candidates = set()
    recovery_c_candidates = set()

    print("\n" + "=" * 80)
    print("PASS 3: FUZZY ADDRESS SCORING")
    print("=" * 80)

    total_scanned = 0
    scored_pairs = 0

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

            for row in chunk.itertuples(index=False):

                total_scanned += 1

                source_id = row.entity_id

                candidate_s1_ids = (
                    fuzzy_source_to_s1.get(
                        source_id
                    )
                )

                if not candidate_s1_ids:
                    continue

                country = country_key(row.country)
                name = normalize_name(row.business_name)
                address = normalize_address(row.business_address)

                for s1_id in candidate_s1_ids:

                    s1 = s1_lookup[s1_id]

                    # Country safety check.
                    if s1["country"] != country:
                        continue

                    scored_pairs += 1

                    pair = (
                        s1_id,
                        source_id,
                    )

                    address_set_ratio = (
                        token_set_ratio(
                            s1["address"],
                            address,
                        )
                        / 100.0
                    )

                    address_overlap = (
                        token_overlap(
                            s1["address"],
                            address,
                        )
                    )

                    # --------------------------------------------
                    # Strategy A
                    # --------------------------------------------

                    if (
                        address_set_ratio
                        >= ADDRESS_SET_RATIO_A
                        and address_overlap
                        >= ADDRESS_OVERLAP_A
                    ):

                        recovery_a_candidates.add(pair)

                        if (
                            s1_id
                            in true_entity_to_s1.get(
                                source_id,
                                ()
                            )
                        ):
                            recovery_a_true.add(pair)

                    # --------------------------------------------
                    # Strategy B
                    # --------------------------------------------

                    if (
                        address_set_ratio
                        >= ADDRESS_SET_RATIO_B
                        and address_overlap
                        >= ADDRESS_OVERLAP_B
                    ):

                        recovery_b_candidates.add(pair)

                        if (
                            s1_id
                            in true_entity_to_s1.get(
                                source_id,
                                ()
                            )
                        ):
                            recovery_b_true.add(pair)

                    # --------------------------------------------
                    # Strategy C
                    # --------------------------------------------

                    if (
                        address_set_ratio
                        >= ADDRESS_SET_RATIO_C
                    ):

                        name_ratio = (
                            ratio(
                                s1["name"],
                                name,
                            )
                            / 100.0
                        )

                        if name_ratio >= NAME_RATIO_C:

                            recovery_c_candidates.add(pair)

                            if (
                                s1_id
                                in true_entity_to_s1.get(
                                    source_id,
                                    ()
                                )
                            ):
                                recovery_c_true.add(pair)

            print(
                f"  scanned: "
                f"{total_scanned:,}"
            )

    # ============================================================
    # 8. Calculate baseline true-pair recall
    #
    # Rather than relying on source scanning during the baseline
    # pass, derive it from the candidate pairs we already have.
    # ============================================================

    baseline_true_pairs = {
        pair
        for pair in baseline_candidates
        if (
            pair[0]
            in true_entity_to_s1.get(
                pair[1],
                ()
            )
        )
    }

    baseline_recall = (
        len(baseline_true_pairs)
        / total_true_pairs
        if total_true_pairs
        else 0.0
    )

    # ============================================================
    # 9. Incremental recovery
    # ============================================================

    new_a = (
        recovery_a_candidates
        - baseline_candidates
    )

    new_b = (
        recovery_b_candidates
        - baseline_candidates
    )

    new_c = (
        recovery_c_candidates
        - baseline_candidates
    )

    new_true_a = (
        recovery_a_true
        - baseline_true_pairs
    )

    new_true_b = (
        recovery_b_true
        - baseline_true_pairs
    )

    new_true_c = (
        recovery_c_true
        - baseline_true_pairs
    )

    # ============================================================
    # 10. Final results
    # ============================================================

    print("\n" + "=" * 80)
    print("FINAL ADDRESS RECOVERY RESULTS")
    print("=" * 80)

    print("\nFrozen baseline:")
    print(
        f"  Candidate pairs: "
        f"{len(baseline_candidates):,}"
    )

    print(
        f"  True pairs recovered: "
        f"{len(baseline_true_pairs):,}"
    )

    print(
        f"  Blocking recall: "
        f"{baseline_recall:.4%}"
    )

    print("\n" + "-" * 80)

    print(
        f"{'Strategy':34s}"
        f"{'New cand.':>14s}"
        f"{'New true':>14s}"
        f"{'Final true':>14s}"
        f"{'Recall':>11s}"
        f"{'Cand/new true':>16s}"
    )

    results = [
        (
            "A: set>=.90 + overlap>=.80",
            new_a,
            new_true_a,
        ),
        (
            "B: set>=.95 + overlap>=.80",
            new_b,
            new_true_b,
        ),
        (
            "C: set>=.90 + name>=.50",
            new_c,
            new_true_c,
        ),
    ]

    for label, new_candidates, new_true in results:

        final_true_count = (
            len(baseline_true_pairs)
            + len(new_true)
        )

        final_recall = (
            final_true_count
            / total_true_pairs
            if total_true_pairs
            else 0.0
        )

        candidate_per_true = (
            len(new_candidates)
            / len(new_true)
            if new_true
            else float("inf")
        )

        print(
            f"{label:34s}"
            f"{len(new_candidates):>14,}"
            f"{len(new_true):>14,}"
            f"{final_true_count:>14,}"
            f"{final_recall:>10.4%}"
            f"{candidate_per_true:>16.1f}"
        )

    # ============================================================
    # 11. Additional useful diagnostics
    # ============================================================

    print("\n" + "-" * 80)

    print(
        f"Fuzzy candidate pairs scored: "
        f"{scored_pairs:,}"
    )

    print(
        f"Fuzzy candidate S1/source pairs before scoring: "
        f"{sum(len(v) for v in fuzzy_source_to_s1.values()):,}"
    )

    print(
        f"Fuzzy address-token frequency cap: "
        f"{FUZZY_ADDRESS_TOKEN_THRESHOLD:,}"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()