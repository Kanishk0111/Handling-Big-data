"""
39_score_address_recovery_50k.py

Score the frozen blocking strategy plus address-dominant
fuzzy recovery strategies on a 50K validation S1 sample.

Models:
    output/candidate_model.joblib

Expected model bundle:
    {
        "logistic": Pipeline(...),
        "hist_gradient_boosting": HistGradientBoostingClassifier(...),
        "features": [...],
        "random_state": 42,
    }

Primary model:
    hist_gradient_boosting

Validation:
    - frozen baseline blocker
    - recovery A
    - recovery B
    - recovery C
    - model probability thresholds
    - macro F0.5 per S1

Important:
    The feature definitions here MUST match the definitions
    used when candidate_model.joblib was trained.
"""

from pathlib import Path
from collections import defaultdict
import random
import sys
import joblib

import numpy as np
import pandas as pd

from rapidfuzz.fuzz import ratio, token_set_ratio


# =====================================================================
# Project imports
# =====================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TRAIN_GROUND_TRUTH,
    OUTPUT_DIR,
    TSV_SEPARATOR,
    DEFAULT_CHUNK_SIZE,
)

from src.normalization import (
    normalize_name,
    normalize_address,
    compact_text,
    token_list,
)


# =====================================================================
# Configuration
# =====================================================================

SEED = 42

SAMPLE_SIZE = 50_000

CHUNK_SIZE = DEFAULT_CHUNK_SIZE

MODEL_FILE = OUTPUT_DIR / "candidate_model.joblib"

SCORED_OUTPUT = (
    OUTPUT_DIR / "address_recovery_scores_50k.csv"
)

THRESHOLD_OUTPUT = (
    OUTPUT_DIR / "address_recovery_scoring_50k.csv"
)


# ---------------------------------------------------------------------
# Frozen baseline blocker
# ---------------------------------------------------------------------

NAME_PAIR_THRESHOLD = 500

ADDRESS_TOKEN_THRESHOLD = 100


# ---------------------------------------------------------------------
# Fuzzy address recovery
#
# A:
#   address token-set ratio >= 0.95
#   address overlap >= 0.80
#
# B:
#   address token-set ratio >= 0.90
#   address overlap >= 0.80
#
# C:
#   address token-set ratio >= 0.90
#   address containment >= 0.80
# ---------------------------------------------------------------------

FUZZY_ADDRESS_TOKEN_THRESHOLD = 1000

RECOVERY_A_TOKEN_RATIO = 0.95
RECOVERY_A_OVERLAP = 0.80

RECOVERY_B_TOKEN_RATIO = 0.90
RECOVERY_B_OVERLAP = 0.80

RECOVERY_C_TOKEN_RATIO = 0.90
RECOVERY_C_CONTAINMENT = 0.80


# ---------------------------------------------------------------------
# Probability thresholds to evaluate
# ---------------------------------------------------------------------

PROBABILITY_THRESHOLDS = [
    0.60,
    0.65,
    0.70,
    0.73,
    0.75,
    0.77,
    0.80,
    0.82,
    0.85,
    0.90,
]


# ---------------------------------------------------------------------
# Exact model features
#
# This is also checked against bundle["features"].
# ---------------------------------------------------------------------

MODEL_FEATURES = [
    "name_exact",
    "name_compact_exact",
    "name_ratio",
    "name_token_set_ratio",
    "name_token_overlap",
    "name_containment",
    "address_exact",
    "address_ratio",
    "address_token_set_ratio",
    "address_token_overlap",
    "address_containment",
]


# =====================================================================
# Blocking token definitions
# =====================================================================

COMMON_BUSINESS_TOKENS = {
    "llc",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "company",
    "co",
    "ltd",
    "limited",
    "private",
    "pvt",
    "plc",
    "llp",
}


COMMON_ADDRESS_TOKENS = {
    "road",
    "rd",
    "street",
    "st",
    "avenue",
    "ave",
    "lane",
    "ln",
    "floor",
    "fl",
    "building",
    "bldg",
    "block",
    "plot",
    "house",
    "no",
    "number",
    "unit",
    "flat",
    "suite",
    "sector",
    "area",
    "phase",
    "village",
    "vill",
    "town",
}


# =====================================================================
# Basic helpers
# =====================================================================

def country_key(value):
    return str(value).strip().casefold()


def safe_ratio(a, b):
    if not a or not b:
        return 0.0

    return ratio(a, b) / 100.0


def safe_token_set_ratio(a, b):
    if not a or not b:
        return 0.0

    return token_set_ratio(a, b) / 100.0


def token_overlap(a, b):
    """
    IMPORTANT:
    This is NOT Jaccard similarity.

    It matches the feature definition used by the
    training pipeline:

        intersection / min(len(A), len(B))
    """

    a = set(token_list(a))
    b = set(token_list(b))

    if not a or not b:
        return 0.0

    return len(a & b) / min(
        len(a),
        len(b),
    )


def containment(a, b):
    """
    Same containment definition used by training:

        intersection / min(len(A), len(B))
    """

    a = set(token_list(a))
    b = set(token_list(b))

    if not a or not b:
        return 0.0

    return len(a & b) / min(
        len(a),
        len(b),
    )


# =====================================================================
# Blocking helpers
# =====================================================================

def name_pair_keys(country, name):
    tokens = [
        x
        for x in token_list(name)
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
        x
        for x in token_list(address)
        if len(x) >= 4
        and x not in COMMON_ADDRESS_TOKENS
    ]

    return {
        f"{country}|address_token|{x}"
        for x in set(tokens)
    }


def prefix_house_key(
    country,
    name,
    address,
):
    compact = compact_text(name)

    if len(compact) < 8:
        return None

    for token in normalize_address(address).split():

        if any(
            char.isdigit()
            for char in token
        ):
            return (
                f"{country}|prefix_house|"
                f"{compact[:8]}|{token}"
            )

    return None


# =====================================================================
# Ground truth
# =====================================================================

def load_ground_truth():
    print("\nLoading ground truth...")

    gt = pd.read_csv(
        TRAIN_GROUND_TRUTH,
        sep=TSV_SEPARATOR,
        dtype=str,
    )

    gt.columns = [
        c.strip()
        for c in gt.columns
    ]

    print(
        f"Ground truth rows: "
        f"{len(gt):,}"
    )

    columns = list(gt.columns)

    source1_col = None
    source2_col = None
    source3_col = None

    for column in columns:

        lc = column.lower()

        if "source1" in lc:
            source1_col = column

        elif "source2" in lc:
            source2_col = column

        elif "source3" in lc:
            source3_col = column

    if source1_col is None:
        source1_col = columns[0]

    if source2_col is None and len(columns) >= 2:
        source2_col = columns[1]

    if source3_col is None and len(columns) >= 3:
        source3_col = columns[2]

    print(
        "Ground-truth columns:"
    )

    print(
        f"  source1: {source1_col}"
    )

    print(
        f"  source2: {source2_col}"
    )

    print(
        f"  source3: {source3_col}"
    )

    truth = defaultdict(set)

    for row in gt.itertuples(
        index=False,
        name=None,
    ):

        s1_id = (
            str(
                row[
                    gt.columns.get_loc(
                        source1_col
                    )
                ]
            ).strip()
        )

        if not s1_id or s1_id.lower() == "nan":
            continue

        if source2_col is not None:

            value = row[
                gt.columns.get_loc(
                    source2_col
                )
            ]

            if (
                pd.notna(value)
                and str(value).strip()
            ):
                truth[s1_id].add(
                    str(value).strip()
                )

        if source3_col is not None:

            value = row[
                gt.columns.get_loc(
                    source3_col
                )
            ]

            if (
                pd.notna(value)
                and str(value).strip()
            ):
                truth[s1_id].add(
                    str(value).strip()
                )

    print(
        f"Truth S1 entities: "
        f"{len(truth):,}"
    )

    print(
        f"Truth pairs: "
        f"{sum(len(v) for v in truth.values()):,}"
    )

    return truth


# =====================================================================
# Validation S1 sample
# =====================================================================

def load_validation_s1_sample():
    validation_file = (
        OUTPUT_DIR /
        "validation_s1_ids.txt"
    )

    ids = [
        x.strip()
        for x in validation_file.read_text().splitlines()
        if x.strip()
    ]

    rng = random.Random(SEED)

    rng.shuffle(ids)

    ids = ids[:SAMPLE_SIZE]

    return ids


# =====================================================================
# Load S1 records
# =====================================================================

def load_s1_records(s1_ids):
    wanted = set(s1_ids)

    records = {}

    print(
        "\nLoading validation S1 records..."
    )

    for chunk in pd.read_csv(
        TRAIN_SOURCE1,
        sep=TSV_SEPARATOR,
        dtype=str,
        chunksize=CHUNK_SIZE,
    ):

        for row in chunk.itertuples(
            index=False
        ):

            entity_id = str(
                row.entity_id
            ).strip()

            if entity_id not in wanted:
                continue

            records[entity_id] = {
                "country": country_key(
                    row.country
                ),
                "name": normalize_name(
                    row.business_name
                ),
                "address": normalize_address(
                    row.business_address
                ),
            }

            if len(records) == len(wanted):
                return records

    return records


# =====================================================================
# Prepare S1 blocking structures
# =====================================================================

def build_s1_blocking_structures(
    s1_records,
    truth,
):
    required_exact = set()
    required_compact = set()
    required_prefix_house = set()

    name_pair_frequency = defaultdict(int)

    address_token_frequency = defaultdict(int)

    required_address_tokens = set()

    true_entity_to_s1 = defaultdict(set)

    for s1_id, row in s1_records.items():

        country = row["country"]
        name = row["name"]
        address = row["address"]

        # -------------------------------------------------------------
        # Exact name
        # -------------------------------------------------------------

        required_exact.add(
            f"{country}|name|{name}"
        )

        # -------------------------------------------------------------
        # Compact name
        # -------------------------------------------------------------

        compact = compact_text(name)

        if compact:
            required_compact.add(
                f"{country}|compact_name|{compact}"
            )

        # -------------------------------------------------------------
        # Prefix + house number
        # -------------------------------------------------------------

        prefix = prefix_house_key(
            country,
            name,
            address,
        )

        if prefix:
            required_prefix_house.add(
                prefix
            )

        # -------------------------------------------------------------
        # Name pair frequency
        # -------------------------------------------------------------

        for key in name_pair_keys(
            country,
            name,
        ):
            name_pair_frequency[key] += 1

        # -------------------------------------------------------------
        # Address token frequency
        # -------------------------------------------------------------

        for key in address_token_keys(
            country,
            address,
        ):
            address_token_frequency[key] += 1

    allowed_name_pairs = {
        key
        for key, count
        in name_pair_frequency.items()
        if count <= NAME_PAIR_THRESHOLD
    }

    allowed_address_tokens = {
        key
        for key, count
        in address_token_frequency.items()
        if count <= ADDRESS_TOKEN_THRESHOLD
    }

    fuzzy_address_tokens = {
        key
        for key, count
        in address_token_frequency.items()
        if count <= FUZZY_ADDRESS_TOKEN_THRESHOLD
    }

    # -----------------------------------------------------------------
    # Address token -> S1 mapping
    # -----------------------------------------------------------------

    address_key_to_s1 = defaultdict(set)

    for s1_id, row in s1_records.items():

        keys = (
            address_token_keys(
                row["country"],
                row["address"],
            )
            & allowed_address_tokens
        )

        for key in keys:
            address_key_to_s1[key].add(
                s1_id
            )

    # -----------------------------------------------------------------
    # Fuzzy address token -> S1 mapping
    # -----------------------------------------------------------------

    fuzzy_address_key_to_s1 = defaultdict(set)

    for s1_id, row in s1_records.items():

        keys = (
            address_token_keys(
                row["country"],
                row["address"],
            )
            & fuzzy_address_tokens
        )

        for key in keys:
            fuzzy_address_key_to_s1[key].add(
                s1_id
            )

    # -----------------------------------------------------------------
    # True entity -> S1
    # -----------------------------------------------------------------

    for s1_id, source_ids in truth.items():

        if s1_id not in s1_records:
            continue

        for source_id in source_ids:
            true_entity_to_s1[
                source_id
            ].add(
                s1_id
            )

    return {
        "required_exact": required_exact,
        "required_compact": required_compact,
        "required_prefix_house": required_prefix_house,
        "allowed_name_pairs": allowed_name_pairs,
        "allowed_address_tokens": allowed_address_tokens,
        "fuzzy_address_tokens": fuzzy_address_tokens,
        "address_key_to_s1": address_key_to_s1,
        "fuzzy_address_key_to_s1": fuzzy_address_key_to_s1,
        "true_entity_to_s1": true_entity_to_s1,
    }


# =====================================================================
# Frozen baseline test
# =====================================================================

def baseline_hit(
    country,
    name,
    address,
    structures,
):
    # Exact name
    exact_key = (
        f"{country}|name|{name}"
    )

    if (
        exact_key
        in structures["required_exact"]
    ):
        return True

    # Compact name
    compact = compact_text(name)

    if compact:

        compact_key = (
            f"{country}|compact_name|{compact}"
        )

        if (
            compact_key
            in structures["required_compact"]
        ):
            return True

    # Prefix + house
    prefix = prefix_house_key(
        country,
        name,
        address,
    )

    if (
        prefix
        and prefix
        in structures["required_prefix_house"]
    ):
        return True

    # Name pair <= 500
    pairs = name_pair_keys(
        country,
        name,
    )

    if (
        pairs
        & structures["allowed_name_pairs"]
    ):
        return True

    return False


# =====================================================================
# Model feature generation
# =====================================================================

def make_features(
    s1_row,
    source_row,
):
    name1 = s1_row["name"]
    name2 = source_row["name"]

    address1 = s1_row["address"]
    address2 = source_row["address"]

    return {
        "name_exact": float(
            name1 == name2
        ),

        "name_compact_exact": float(
            compact_text(name1)
            == compact_text(name2)
        ),

        "name_ratio": safe_ratio(
            name1,
            name2,
        ),

        "name_token_set_ratio":
            safe_token_set_ratio(
                name1,
                name2,
            ),

        "name_token_overlap":
            token_overlap(
                name1,
                name2,
            ),

        "name_containment":
            containment(
                name1,
                name2,
            ),

        "address_exact": float(
            address1 == address2
        ),

        "address_ratio": safe_ratio(
            address1,
            address2,
        ),

        "address_token_set_ratio":
            safe_token_set_ratio(
                address1,
                address2,
            ),

        "address_token_overlap":
            token_overlap(
                address1,
                address2,
            ),

        "address_containment":
            containment(
                address1,
                address2,
            ),
    }


# =====================================================================
# Evaluate address recovery condition
# =====================================================================

def recovery_hits(
    s1_row,
    source_row,
):
    address1 = s1_row["address"]
    address2 = source_row["address"]

    address_token_ratio = (
        safe_token_set_ratio(
            address1,
            address2,
        )
    )

    address_overlap = (
        token_overlap(
            address1,
            address2,
        )
    )

    address_containment = (
        containment(
            address1,
            address2,
        )
    )

    hit_a = (
        address_token_ratio
        >= RECOVERY_A_TOKEN_RATIO
        and
        address_overlap
        >= RECOVERY_A_OVERLAP
    )

    hit_b = (
        address_token_ratio
        >= RECOVERY_B_TOKEN_RATIO
        and
        address_overlap
        >= RECOVERY_B_OVERLAP
    )

    hit_c = (
        address_token_ratio
        >= RECOVERY_C_TOKEN_RATIO
        and
        address_containment
        >= RECOVERY_C_CONTAINMENT
    )

    return (
        hit_a,
        hit_b,
        hit_c,
    )


# =====================================================================
# Scan source files and generate candidates
# =====================================================================

def generate_candidates(
    s1_records,
    structures,
    truth,
):
    """
    Generates the candidate pairs needed for:

        baseline
        A
        B
        C

    Candidate dictionaries contain:
        baseline
        A
        B
        C

    Each value is a list of:
        (s1_id, source_id)

    We also retain the normalized source row needed
    later for feature scoring.
    """

    baseline_candidates = set()
    recovery_a_candidates = set()
    recovery_b_candidates = set()
    recovery_c_candidates = set()

    # -----------------------------------------------------------------
    # These are useful diagnostics.
    # -----------------------------------------------------------------

    baseline_true = set()
    recovery_a_true = set()
    recovery_b_true = set()
    recovery_c_true = set()

    source_rows = {}

    scanned = 0

    for source_path in [
        TRAIN_SOURCE2,
        TRAIN_SOURCE3,
    ]:

        print(
            f"\nScanning "
            f"{source_path.name}..."
        )

        for chunk in pd.read_csv(
            source_path,
            sep=TSV_SEPARATOR,
            dtype=str,
            chunksize=CHUNK_SIZE,
        ):

            for row in chunk.itertuples(
                index=False
            ):

                scanned += 1

                source_id = str(
                    row.entity_id
                ).strip()

                country = country_key(
                    row.country
                )

                name = normalize_name(
                    row.business_name
                )

                address = normalize_address(
                    row.business_address
                )

                source = {
                    "source_id": source_id,
                    "country": country,
                    "name": name,
                    "address": address,
                    "source_type": (
                        "source2"
                        if source_path
                        == TRAIN_SOURCE2
                        else "source3"
                    ),
                }

                # =====================================================
                # 1. Frozen baseline
                # =====================================================

                is_baseline = baseline_hit(
                    country,
                    name,
                    address,
                    structures,
                )

                if is_baseline:

                    true_s1s = (
                        structures[
                            "true_entity_to_s1"
                        ].get(
                            source_id,
                            set(),
                        )
                    )

                    for s1_id in s1_records:

                        # We cannot create the complete
                        # cross-product here.
                        #
                        # Instead use name/address blocks
                        # below to identify S1s.
                        pass

                # =====================================================
                # 2. Address-token candidates
                # =====================================================

                base_addr_keys = (
                    address_token_keys(
                        country,
                        address,
                    )
                    & structures[
                        "allowed_address_tokens"
                    ]
                )

                matched_base_s1 = set()

                for key in base_addr_keys:

                    matched_base_s1.update(
                        structures[
                            "address_key_to_s1"
                        ].get(
                            key,
                            set(),
                        )
                    )

                # =====================================================
                # 3. Fuzzy address candidates
                # =====================================================

                fuzzy_addr_keys = (
                    address_token_keys(
                        country,
                        address,
                    )
                    & structures[
                        "fuzzy_address_tokens"
                    ]
                )

                matched_fuzzy_s1 = set()

                for key in fuzzy_addr_keys:

                    matched_fuzzy_s1.update(
                        structures[
                            "fuzzy_address_key_to_s1"
                        ].get(
                            key,
                            set(),
                        )
                    )

                # =====================================================
                # 4. We need name-based S1 lookup as well.
                #
                # Build it lazily outside this loop below.
                # =====================================================

                # Save only source rows that participate in
                # address recovery. Baseline rows are handled
                # separately by the name maps.
                #
                # This keeps memory bounded.
                if (
                    is_baseline
                    or matched_base_s1
                    or matched_fuzzy_s1
                ):
                    source_rows[
                        source_id
                    ] = source

                # -----------------------------------------------------
                # Address-token baseline candidates
                # -----------------------------------------------------

                for s1_id in matched_base_s1:

                    pair = (
                        s1_id,
                        source_id,
                    )

                    baseline_candidates.add(
                        pair
                    )

                    if (
                        source_id
                        in truth.get(
                            s1_id,
                            set(),
                        )
                    ):
                        baseline_true.add(
                            pair
                        )

                # -----------------------------------------------------
                # Fuzzy recovery
                # -----------------------------------------------------

                for s1_id in matched_fuzzy_s1:

                    s1_row = s1_records[
                        s1_id
                    ]

                    hit_a, hit_b, hit_c = (
                        recovery_hits(
                            s1_row,
                            source,
                        )
                    )

                    if hit_a:

                        pair = (
                            s1_id,
                            source_id,
                        )

                        recovery_a_candidates.add(
                            pair
                        )

                        if (
                            source_id
                            in truth.get(
                                s1_id,
                                set(),
                            )
                        ):
                            recovery_a_true.add(
                                pair
                            )

                    if hit_b:

                        pair = (
                            s1_id,
                            source_id,
                        )

                        recovery_b_candidates.add(
                            pair
                        )

                        if (
                            source_id
                            in truth.get(
                                s1_id,
                                set(),
                            )
                        ):
                            recovery_b_true.add(
                                pair
                            )

                    if hit_c:

                        pair = (
                            s1_id,
                            source_id,
                        )

                        recovery_c_candidates.add(
                            pair
                        )

                        if (
                            source_id
                            in truth.get(
                                s1_id,
                                set(),
                            )
                        ):
                            recovery_c_true.add(
                                pair
                            )

            print(
                f"  scanned: "
                f"{scanned:,}"
            )

    # =================================================================
    # IMPORTANT:
    #
    # The baseline name blocks need their own S1 lookup because a
    # source row can match many S1s through name-pair keys.
    #
    # Build those maps and scan sources AGAIN only for baseline
    # candidates.
    # =================================================================

    print(
        "\nBuilding name-based S1 maps..."
    )

    exact_to_s1 = defaultdict(set)
    compact_to_s1 = defaultdict(set)
    prefix_to_s1 = defaultdict(set)
    name_pair_to_s1 = defaultdict(set)

    for s1_id, row in s1_records.items():

        country = row["country"]
        name = row["name"]
        address = row["address"]

        exact_to_s1[
            f"{country}|name|{name}"
        ].add(s1_id)

        compact = compact_text(name)

        if compact:

            compact_to_s1[
                f"{country}|compact_name|{compact}"
            ].add(s1_id)

        prefix = prefix_house_key(
            country,
            name,
            address,
        )

        if prefix:

            prefix_to_s1[
                prefix
            ].add(s1_id)

        for key in (
            name_pair_keys(
                country,
                name,
            )
            & structures[
                "allowed_name_pairs"
            ]
        ):

            name_pair_to_s1[
                key
            ].add(s1_id)

    print(
        "Scanning sources for baseline "
        "name candidates..."
    )

    scanned = 0

    for source_path in [
        TRAIN_SOURCE2,
        TRAIN_SOURCE3,
    ]:

        for chunk in pd.read_csv(
            source_path,
            sep=TSV_SEPARATOR,
            dtype=str,
            chunksize=CHUNK_SIZE,
        ):

            for row in chunk.itertuples(
                index=False
            ):

                scanned += 1

                source_id = str(
                    row.entity_id
                ).strip()

                country = country_key(
                    row.country
                )

                name = normalize_name(
                    row.business_name
                )

                address = normalize_address(
                    row.business_address
                )

                candidate_s1 = set()

                # Exact
                candidate_s1.update(
                    exact_to_s1.get(
                        f"{country}|name|{name}",
                        set(),
                    )
                )

                # Compact
                compact = compact_text(name)

                if compact:

                    candidate_s1.update(
                        compact_to_s1.get(
                            f"{country}|compact_name|{compact}",
                            set(),
                        )
                    )

                # Prefix + house
                prefix = prefix_house_key(
                    country,
                    name,
                    address,
                )

                if prefix:

                    candidate_s1.update(
                        prefix_to_s1.get(
                            prefix,
                            set(),
                        )
                    )

                # Name pairs
                for key in (
                    name_pair_keys(
                        country,
                        name,
                    )
                    & structures[
                        "allowed_name_pairs"
                    ]
                ):

                    candidate_s1.update(
                        name_pair_to_s1.get(
                            key,
                            set(),
                        )
                    )

                for s1_id in candidate_s1:

                    pair = (
                        s1_id,
                        source_id,
                    )

                    baseline_candidates.add(
                        pair
                    )

                    if (
                        source_id
                        in truth.get(
                            s1_id,
                            set(),
                        )
                    ):
                        baseline_true.add(
                            pair
                        )

                    if source_id not in source_rows:

                        source_rows[
                            source_id
                        ] = {
                            "source_id":
                                source_id,
                            "country":
                                country,
                            "name":
                                name,
                            "address":
                                address,
                            "source_type":
                                (
                                    "source2"
                                    if source_path
                                    == TRAIN_SOURCE2
                                    else "source3"
                                ),
                        }

    # =================================================================
    # Return
    # =================================================================

    return {
        "baseline": baseline_candidates,
        "A": recovery_a_candidates,
        "B": recovery_b_candidates,
        "C": recovery_c_candidates,
        "baseline_true": baseline_true,
        "A_true": recovery_a_true,
        "B_true": recovery_b_true,
        "C_true": recovery_c_true,
        "source_rows": source_rows,
    }


# =====================================================================
# Build final candidate sets
# =====================================================================

def build_final_candidate_sets(
    candidate_data,
):
    baseline = candidate_data[
        "baseline"
    ]

    A = baseline | candidate_data["A"]
    B = baseline | candidate_data["B"]
    C = baseline | candidate_data["C"]

    return {
        "baseline": baseline,
        "A": A,
        "B": B,
        "C": C,
    }


# =====================================================================
# Score candidates
# =====================================================================

def score_candidate_pairs(
    candidate_pairs,
    s1_records,
    source_rows,
    model,
    model_features,
):
    """
    Score a candidate pair set.

    Returns DataFrame with:
        s1_id
        source_id
        probability
        model features
    """

    print(
        f"\nScoring "
        f"{len(candidate_pairs):,} "
        f"candidate pairs..."
    )

    rows = []

    batch = []

    metadata = []

    BATCH_SIZE = 20_000

    def flush_batch():

        nonlocal batch
        nonlocal metadata
        nonlocal rows

        if not batch:
            return

        X = pd.DataFrame(
            batch
        )[model_features]

        probabilities = model.predict_proba(
            X
        )[:, 1]

        for meta, probability in zip(
            metadata,
            probabilities,
        ):

            result = {
                "s1_id": meta["s1_id"],
                "source_id": meta["source_id"],
                "source_type": meta[
                    "source_type"
                ],
                "probability": float(
                    probability
                ),
            }

            result.update(
                meta["features"]
            )

            rows.append(result)

        batch = []
        metadata = []

    processed = 0

    for s1_id, source_id in candidate_pairs:

        s1_row = s1_records.get(
            s1_id
        )

        source_row = source_rows.get(
            source_id
        )

        if (
            s1_row is None
            or source_row is None
        ):
            continue

        features = make_features(
            s1_row,
            source_row,
        )

        batch.append(features)

        metadata.append({
            "s1_id": s1_id,
            "source_id": source_id,
            "source_type":
                source_row[
                    "source_type"
                ],
            "features": features,
        })

        if len(batch) >= BATCH_SIZE:

            flush_batch()

        processed += 1

        if (
            processed % 500_000
            == 0
        ):

            print(
                f"  processed: "
                f"{processed:,}"
            )

    flush_batch()

    result = pd.DataFrame(rows)

    return result


# =====================================================================
# F0.5
# =====================================================================

def f05(
    precision,
    recall,
):
    beta = 0.5

    if precision <= 0 and recall <= 0:
        return 0.0

    denominator = (
        beta * beta * precision
        + recall
    )

    if denominator == 0:
        return 0.0

    return (
        (1 + beta * beta)
        * precision
        * recall
        / denominator
    )


# =====================================================================
# Macro F0.5
# =====================================================================

def evaluate_predictions(
    scored,
    truth,
    threshold,
    s1_ids,
):
    """
    Challenge-style entity-level evaluation.

    For every S1:
        actual = true source IDs
        predicted = model-positive source IDs

    F0.5 is calculated separately for every S1,
    then averaged across S1 entities.

    Singleton / zero-match entities are therefore
    explicitly represented.
    """

    predicted_by_s1 = defaultdict(set)

    selected = scored[
        scored["probability"]
        >= threshold
    ]

    for row in selected.itertuples(
        index=False
    ):

        predicted_by_s1[
            row.s1_id
        ].add(
            row.source_id
        )

    scores = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    predicted_s1_count = 0

    for s1_id in s1_ids:

        actual = set(
            truth.get(
                s1_id,
                set(),
            )
        )

        predicted = (
            predicted_by_s1.get(
                s1_id,
                set(),
            )
        )

        if predicted:
            predicted_s1_count += 1

        tp = len(
            actual & predicted
        )

        fp = len(
            predicted - actual
        )

        fn = len(
            actual - predicted
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        if tp == 0:

            if (
                len(actual) == 0
                and len(predicted) == 0
            ):
                score = 1.0

            else:
                score = 0.0

        else:

            precision = (
                tp
                / len(predicted)
                if predicted
                else 0.0
            )

            recall = (
                tp
                / len(actual)
                if actual
                else 0.0
            )

            score = f05(
                precision,
                recall,
            )

        scores.append(score)

    macro_f05 = (
        float(np.mean(scores))
        if scores
        else 0.0
    )

    pair_precision = (
        total_tp
        / (total_tp + total_fp)
        if total_tp + total_fp
        else 0.0
    )

    pair_recall = (
        total_tp
        / (total_tp + total_fn)
        if total_tp + total_fn
        else 0.0
    )

    return {
        "threshold": threshold,
        "macro_f05": macro_f05,
        "pair_precision": pair_precision,
        "pair_recall": pair_recall,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "predicted_pairs": int(
            len(selected)
        ),
        "predicted_s1": predicted_s1_count,
        "empty_s1": len(s1_ids)
        - predicted_s1_count,
    }


# =====================================================================
# Main
# =====================================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print(
        "SCORE ADDRESS RECOVERY — "
        "50K VALIDATION"
    )
    print("=" * 72)

    # ================================================================
    # 1. Load model bundle
    # ================================================================

    print(
        "\nLoading candidate model..."
    )

    bundle = joblib.load(
        MODEL_FILE
    )

    if not isinstance(
        bundle,
        dict,
    ):
        raise RuntimeError(
            "Expected candidate_model.joblib "
            "to contain a dictionary."
        )

    print(
        "Bundle keys:"
    )

    for key in bundle:
        print(
            f"  {key!r}: "
            f"{type(bundle[key]).__name__}"
        )

    required_keys = {
        "hist_gradient_boosting",
        "features",
    }

    missing = (
        required_keys
        - set(bundle.keys())
    )

    if missing:

        raise RuntimeError(
            "candidate_model.joblib "
            "is missing keys: "
            f"{sorted(missing)}"
        )

    model = bundle[
        "hist_gradient_boosting"
    ]

    model_features = list(
        bundle["features"]
    )

    print(
        f"\nModel type: "
        f"{type(model).__name__}"
    )

    print(
        f"Model features: "
        f"{len(model_features)}"
    )

    print(
        "Model feature order:"
    )

    for i, feature in enumerate(
        model_features,
        start=1,
    ):
        print(
            f"  {i:2d}. {feature}"
        )

    if not hasattr(
        model,
        "predict_proba",
    ):

        raise RuntimeError(
            "HistGradientBoosting model "
            "does not provide predict_proba()."
        )

    if (
        model_features
        != MODEL_FEATURES
    ):

        raise RuntimeError(
            "\nFeature mismatch!\n"
            f"Saved model features:\n"
            f"{model_features}\n\n"
            f"Expected features:\n"
            f"{MODEL_FEATURES}"
        )

    # ================================================================
    # 2. Validation sample
    # ================================================================

    s1_ids = (
        load_validation_s1_sample()
    )

    s1_id_set = set(
        s1_ids
    )

    print(
        f"\nValidation S1 sample: "
        f"{len(s1_ids):,}"
    )

    s1_records = (
        load_s1_records(
            s1_ids
        )
    )

    if len(s1_records) != len(
        s1_ids
    ):

        missing_s1 = (
            s1_id_set
            - set(s1_records)
        )

        raise RuntimeError(
            "Could not load all sampled "
            "S1 records. Missing: "
            f"{len(missing_s1):,}"
        )

    print(
        f"S1 records loaded: "
        f"{len(s1_records):,}"
    )

    # ================================================================
    # 3. Ground truth
    # ================================================================

    truth = load_ground_truth()

    sample_truth = {
        s1_id: truth.get(
            s1_id,
            set(),
        )
        for s1_id in s1_ids
    }

    total_true_pairs = sum(
        len(v)
        for v in sample_truth.values()
    )

    print(
        f"\nTrue validation pairs: "
        f"{total_true_pairs:,}"
    )

    # ================================================================
    # 4. Blocking structures
    # ================================================================

    print(
        "\nBuilding blocking structures..."
    )

    structures = (
        build_s1_blocking_structures(
            s1_records,
            sample_truth,
        )
    )

    print(
        f"Allowed name-pair keys: "
        f"{len(structures['allowed_name_pairs']):,}"
    )

    print(
        f"Allowed address-token keys: "
        f"{len(structures['allowed_address_tokens']):,}"
    )

    print(
        f"Fuzzy address-token keys: "
        f"{len(structures['fuzzy_address_tokens']):,}"
    )

    # ================================================================
    # 5. Generate candidates
    # ================================================================

    candidate_data = (
        generate_candidates(
            s1_records,
            structures,
            sample_truth,
        )
    )

    candidate_sets = (
        build_final_candidate_sets(
            candidate_data
        )
    )

    print(
        "\n" + "=" * 72
    )

    print(
        "BLOCKING SUMMARY"
    )

    print(
        "=" * 72
    )

    for strategy in [
        "baseline",
        "A",
        "B",
        "C",
    ]:

        pairs = candidate_sets[
            strategy
        ]

        true_pairs = (
            pairs
            & set(
                (
                    s1_id,
                    source_id
                )
                for s1_id, source_ids
                in sample_truth.items()
                for source_id
                in source_ids
            )
        )

        recall = (
            len(true_pairs)
            / total_true_pairs
            if total_true_pairs
            else 0.0
        )

        print(
            f"\n{strategy}:"
        )

        print(
            f"  Candidate pairs: "
            f"{len(pairs):,}"
        )

        print(
            f"  True pairs recovered: "
            f"{len(true_pairs):,}"
        )

        print(
            f"  Blocking recall: "
            f"{recall:.4%}"
        )

    # ================================================================
    # 6. Score each final strategy
    # ================================================================

    scored_by_strategy = {}

    for strategy in [
        "baseline",
        "A",
        "B",
        "C",
    ]:

        print(
            "\n" + "=" * 72
        )

        print(
            f"SCORING STRATEGY: "
            f"{strategy}"
        )

        print(
            "=" * 72
        )

        scored = score_candidate_pairs(
            candidate_sets[
                strategy
            ],
            s1_records,
            candidate_data[
                "source_rows"
            ],
            model,
            model_features,
        )

        scored[
            "strategy"
        ] = strategy

        scored_by_strategy[
            strategy
        ] = scored

        print(
            f"Scored rows: "
            f"{len(scored):,}"
        )

    # ================================================================
    # 7. Save scoring data immediately
    # ================================================================

    print(
        "\nSaving scored candidate data..."
    )

    all_scored = pd.concat(
        scored_by_strategy.values(),
        ignore_index=True,
    )

    # Avoid duplicated rows if the same pair appears
    # in multiple strategies.
    all_scored = (
        all_scored
        .drop_duplicates(
            subset=[
                "strategy",
                "s1_id",
                "source_id",
            ]
        )
    )

    all_scored.to_csv(
        SCORED_OUTPUT,
        index=False,
    )

    print(
        f"Saved:\n"
        f"  {SCORED_OUTPUT}"
    )

    # ================================================================
    # 8. Evaluate thresholds
    # ================================================================

    print(
        "\n" + "=" * 72
    )

    print(
        "THRESHOLD EVALUATION"
    )

    print(
        "=" * 72
    )

    results = []

    for strategy in [
        "baseline",
        "A",
        "B",
        "C",
    ]:

        scored = scored_by_strategy[
            strategy
        ]

        print(
            f"\nStrategy: {strategy}"
        )

        print(
            "-" * 72
        )

        for threshold in (
            PROBABILITY_THRESHOLDS
        ):

            metrics = (
                evaluate_predictions(
                    scored,
                    sample_truth,
                    threshold,
                    s1_ids,
                )
            )

            metrics[
                "strategy"
            ] = strategy

            results.append(
                metrics
            )

            print(
                f"  threshold={threshold:.2f} "
                f"| macro_F0.5="
                f"{metrics['macro_f05']:.6f} "
                f"| pair_P="
                f"{metrics['pair_precision']:.6f} "
                f"| pair_R="
                f"{metrics['pair_recall']:.6f} "
                f"| predicted="
                f"{metrics['predicted_pairs']:,}"
            )

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        THRESHOLD_OUTPUT,
        index=False,
    )

    # ================================================================
    # 9. Best result per strategy
    # ================================================================

    print(
        "\n" + "=" * 72
    )

    print(
        "BEST RESULT PER STRATEGY"
    )

    print(
        "=" * 72
    )

    for strategy in [
        "baseline",
        "A",
        "B",
        "C",
    ]:

        subset = results_df[
            results_df["strategy"]
            == strategy
        ]

        best = subset.loc[
            subset["macro_f05"].idxmax()
        ]

        print(
            f"\n{strategy}:"
        )

        print(
            f"  threshold: "
            f"{best['threshold']:.2f}"
        )

        print(
            f"  macro F0.5: "
            f"{best['macro_f05']:.6f}"
        )

        print(
            f"  pair precision: "
            f"{best['pair_precision']:.6f}"
        )

        print(
            f"  pair recall: "
            f"{best['pair_recall']:.6f}"
        )

        print(
            f"  predicted pairs: "
            f"{int(best['predicted_pairs']):,}"
        )

        print(
            f"  predicted S1: "
            f"{int(best['predicted_s1']):,}"
        )

        print(
            f"  empty S1: "
            f"{int(best['empty_s1']):,}"
        )

    # ================================================================
    # 10. Overall best tested combination
    # ================================================================

    best_idx = results_df[
        "macro_f05"
    ].idxmax()

    best = results_df.loc[
        best_idx
    ]

    print(
        "\n" + "=" * 72
    )

    print(
        "BEST TESTED COMBINATION"
    )

    print(
        "=" * 72
    )

    print(
        f"Strategy: "
        f"{best['strategy']}"
    )

    print(
        f"Threshold: "
        f"{best['threshold']:.2f}"
    )

    print(
        f"Macro F0.5: "
        f"{best['macro_f05']:.6f}"
    )

    print(
        f"Pair precision: "
        f"{best['pair_precision']:.6f}"
    )

    print(
        f"Pair recall: "
        f"{best['pair_recall']:.6f}"
    )

    print(
        f"Predicted pairs: "
        f"{int(best['predicted_pairs']):,}"
    )

    print(
        f"Predicted S1: "
        f"{int(best['predicted_s1']):,}"
    )

    print(
        f"Empty S1: "
        f"{int(best['empty_s1']):,}"
    )

    print(
        f"\nThreshold results saved:\n"
        f"  {THRESHOLD_OUTPUT}"
    )

    print(
        "\nDone."
    )


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    main()