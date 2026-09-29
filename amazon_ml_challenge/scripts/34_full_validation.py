from pathlib import Path

import joblib
import numpy as np
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


# ============================================================
# CONFIG
# ============================================================

VALIDATION_IDS_PATH = Path(
    "output/validation_s1_ids.txt"
)

MODEL_PATH = Path(
    "output/candidate_model.joblib"
)

OUTPUT_DIR = Path("output")

CHUNK_SIZE = 100_000

# IMPORTANT:
# Start with 50k S1 entities.
# This is large enough to validate entity-level behavior,
# while avoiding the 45 repeated full scans from v1.
VALIDATION_SAMPLE_SIZE = 50_000

RANDOM_STATE = 42

NAME_PAIR_THRESHOLD = 500
ADDRESS_TOKEN_THRESHOLD = 100

ADDRESS_RECOVERY_SET_RATIO = 0.90
ADDRESS_RECOVERY_OVERLAP = 0.80


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


# ============================================================
# BLOCKING HELPERS
# ============================================================

def country_key(value):
    return str(value).strip().casefold()


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

    for token in normalize_address(
        address
    ).split():

        if any(
            c.isdigit()
            for c in token
        ):

            return (
                f"{country}|prefix_house|"
                f"{compact[:8]}|{token}"
            )

    return None


# ============================================================
# FEATURE HELPERS
# ============================================================

def token_overlap(a, b):

    a = set(token_list(a))
    b = set(token_list(b))

    if not a or not b:
        return 0.0

    return len(a & b) / min(
        len(a),
        len(b),
    )


def containment(a, b):

    return token_overlap(a, b)


def address_recovery_match(
    address1,
    address2,
):

    if not address1 or not address2:
        return False

    overlap = token_overlap(
        address1,
        address2,
    )

    if overlap < ADDRESS_RECOVERY_OVERLAP:
        return False

    set_ratio = (
        token_set_ratio(
            address1,
            address2,
        )
        / 100.0
    )

    return (
        set_ratio
        >= ADDRESS_RECOVERY_SET_RATIO
    )


def make_features(
    s1,
    source,
):

    name1 = s1["name"]
    name2 = source["name"]

    address1 = s1["address"]
    address2 = source["address"]

    compact1 = compact_text(name1)
    compact2 = compact_text(name2)

    return {
        "name_exact": float(
            name1 == name2
            and bool(name1)
        ),

        "name_compact_exact": float(
            compact1 == compact2
            and bool(compact1)
        ),

        "name_ratio": (
            ratio(
                name1,
                name2,
            )
            / 100.0
        ),

        "name_token_set_ratio": (
            token_set_ratio(
                name1,
                name2,
            )
            / 100.0
        ),

        "name_token_overlap": (
            token_overlap(
                name1,
                name2,
            )
        ),

        "name_containment": (
            containment(
                name1,
                name2,
            )
        ),

        "address_exact": float(
            address1 == address2
            and bool(address1)
        ),

        "address_ratio": (
            ratio(
                address1,
                address2,
            )
            / 100.0
        ),

        "address_token_set_ratio": (
            token_set_ratio(
                address1,
                address2,
            )
            / 100.0
        ),

        "address_token_overlap": (
            token_overlap(
                address1,
                address2,
            )
        ),

        "address_containment": (
            containment(
                address1,
                address2,
            )
        ),
    }


# ============================================================
# LOAD S1
# ============================================================

def load_validation_s1():

    ids = [
        x.strip()
        for x in VALIDATION_IDS_PATH
        .read_text()
        .splitlines()
        if x.strip()
    ]

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    if len(ids) > VALIDATION_SAMPLE_SIZE:

        selected = rng.choice(
            ids,
            size=VALIDATION_SAMPLE_SIZE,
            replace=False,
        )

        ids = selected.tolist()

    id_set = set(ids)

    print(
        f"Validation S1 sample: "
        f"{len(ids):,}"
    )

    s1 = pd.read_csv(
        TRAIN_SOURCE1,
        sep="\t",
        dtype=str,
    )

    s1 = s1[
        s1["entity_id"].isin(id_set)
    ]

    lookup = {}

    for row in s1.itertuples(
        index=False
    ):

        lookup[row.entity_id] = {
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

    return ids, lookup


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_truth(
    validation_ids
):

    id_set = set(
        validation_ids
    )

    gt = pd.read_csv(
        TRAIN_GROUND_TRUTH,
        sep="\t",
        dtype=str,
    )

    gt[
        "matched_entity_ids"
    ] = (
        gt[
            "matched_entity_ids"
        ].fillna("")
    )

    truth = {}

    for row in gt.itertuples(
        index=False
    ):

        s1_id = (
            row.source1_entity_id
        )

        if s1_id not in id_set:
            continue

        truth[s1_id] = {
            x.strip()
            for x in (
                row.matched_entity_ids
                .split(",")
            )
            if x.strip()
        }

    return truth


# ============================================================
# BUILD S1 INVERSE MAPS
# ============================================================

def build_s1_maps(
    s1_lookup
):

    exact_to_s1 = {}
    compact_to_s1 = {}
    prefix_to_s1 = {}
    pair_to_s1 = {}
    address_to_s1 = {}

    required_pairs = set()
    required_address_tokens = set()

    for s1_id, row in s1_lookup.items():

        country = row["country"]
        name = row["name"]
        address = row["address"]

        exact_key = (
            f"{country}|name|{name}"
        )

        exact_to_s1.setdefault(
            exact_key,
            set(),
        ).add(s1_id)

        compact = compact_text(name)

        if compact:

            compact_key = (
                f"{country}|compact_name|{compact}"
            )

            compact_to_s1.setdefault(
                compact_key,
                set(),
            ).add(s1_id)

        prefix = prefix_house_key(
            country,
            name,
            address,
        )

        if prefix:

            prefix_to_s1.setdefault(
                prefix,
                set(),
            ).add(s1_id)

        pairs = name_pair_keys(
            country,
            name,
        )

        required_pairs.update(
            pairs
        )

        for key in pairs:

            pair_to_s1.setdefault(
                key,
                set(),
            ).add(s1_id)

        address_keys = (
            address_token_keys(
                country,
                address,
            )
        )

        required_address_tokens.update(
            address_keys
        )

        for key in address_keys:

            address_to_s1.setdefault(
                key,
                set(),
            ).add(s1_id)

    return {
        "exact": exact_to_s1,
        "compact": compact_to_s1,
        "prefix": prefix_to_s1,
        "pair": pair_to_s1,
        "address": address_to_s1,
        "required_pairs": required_pairs,
        "required_address_tokens": (
            required_address_tokens
        ),
    }


# ============================================================
# ONE PASS: FREQUENCY COUNT
# ============================================================

def build_allowed_keys(
    maps
):

    name_frequency = {}
    address_frequency = {}

    print()
    print(
        "PASS 1: frequency scan"
    )

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

            for row in chunk.itertuples(
                index=False
            ):

                country = country_key(
                    row.country
                )

                name = normalize_name(
                    row.business_name
                )

                address = normalize_address(
                    row.business_address
                )

                for key in (
                    name_pair_keys(
                        country,
                        name,
                    )
                    & maps["required_pairs"]
                ):

                    name_frequency[key] = (
                        name_frequency.get(
                            key,
                            0,
                        )
                        + 1
                    )

                for key in (
                    address_token_keys(
                        country,
                        address,
                    )
                    & maps[
                        "required_address_tokens"
                    ]
                ):

                    address_frequency[key] = (
                        address_frequency.get(
                            key,
                            0,
                        )
                        + 1
                    )

            scanned += len(chunk)

        print(
            f"  {source_path.name}: "
            f"{scanned:,} rows"
        )

    allowed_pairs = {
        key
        for key, count
        in name_frequency.items()
        if count <= NAME_PAIR_THRESHOLD
    }

    allowed_address = {
        key
        for key, count
        in address_frequency.items()
        if count <= ADDRESS_TOKEN_THRESHOLD
    }

    print(
        f"Allowed name-pair keys: "
        f"{len(allowed_pairs):,}"
    )

    print(
        f"Allowed address-token keys: "
        f"{len(allowed_address):,}"
    )

    return (
        allowed_pairs,
        allowed_address,
    )


# ============================================================
# ONE SOURCE SCAN FOR CANDIDATES
# ============================================================

def generate_candidates(
    s1_lookup,
    maps,
    allowed_pairs,
    allowed_address,
):

    candidate_map = {}

    def add_candidate(
        source_id,
        s1_id,
    ):

        candidate_map.setdefault(
            source_id,
            set(),
        ).add(s1_id)

    print()
    print(
        "PASS 2: candidate generation"
    )

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

            for row in chunk.itertuples(
                index=False
            ):

                source_id = row.entity_id

                country = country_key(
                    row.country
                )

                name = normalize_name(
                    row.business_name
                )

                address = normalize_address(
                    row.business_address
                )

                # ------------------------------------------------
                # Exact name
                # ------------------------------------------------

                exact_key = (
                    f"{country}|name|{name}"
                )

                for s1_id in maps[
                    "exact"
                ].get(
                    exact_key,
                    (),
                ):

                    add_candidate(
                        source_id,
                        s1_id,
                    )

                # ------------------------------------------------
                # Compact name
                # ------------------------------------------------

                compact = compact_text(
                    name
                )

                if compact:

                    compact_key = (
                        f"{country}|compact_name|{compact}"
                    )

                    for s1_id in maps[
                        "compact"
                    ].get(
                        compact_key,
                        (),
                    ):

                        add_candidate(
                            source_id,
                            s1_id,
                        )

                # ------------------------------------------------
                # Prefix + house
                # ------------------------------------------------

                prefix = prefix_house_key(
                    country,
                    name,
                    address,
                )

                if prefix:

                    for s1_id in maps[
                        "prefix"
                    ].get(
                        prefix,
                        (),
                    ):

                        add_candidate(
                            source_id,
                            s1_id,
                        )

                # ------------------------------------------------
                # Name pair <= 500
                # ------------------------------------------------

                for key in (
                    name_pair_keys(
                        country,
                        name,
                    )
                    & allowed_pairs
                ):

                    for s1_id in maps[
                        "pair"
                    ].get(
                        key,
                        (),
                    ):

                        add_candidate(
                            source_id,
                            s1_id,
                        )

                # ------------------------------------------------
                # Address token <= 100
                # ------------------------------------------------

                addr_keys = (
                    address_token_keys(
                        country,
                        address,
                    )
                    & allowed_address
                )

                for key in addr_keys:

                    for s1_id in maps[
                        "address"
                    ].get(
                        key,
                        (),
                    ):

                        add_candidate(
                            source_id,
                            s1_id,
                        )

                # ------------------------------------------------
                # Address-dominant recovery
                # ------------------------------------------------

                for key in addr_keys:

                    for s1_id in maps[
                        "address"
                    ].get(
                        key,
                        (),
                    ):

                        s1_row = (
                            s1_lookup[s1_id]
                        )

                        if address_recovery_match(
                            s1_row["address"],
                            address,
                        ):

                            add_candidate(
                                source_id,
                                s1_id,
                            )

            scanned += len(chunk)

        print(
            f"  {source_path.name}: "
            f"{scanned:,} rows scanned"
        )

    print(
        f"Unique candidate source entities: "
        f"{len(candidate_map):,}"
    )

    return candidate_map


# ============================================================
# SCORE CANDIDATES
# ============================================================

def score_candidates(
    candidate_map,
    s1_lookup,
    truth,
    model,
):

    classifier = model[
        "hist_gradient_boosting"
    ]

    results = []

    print()
    print(
        "PASS 3: scoring candidates"
    )

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

            feature_rows = []
            metadata = []

            for row in chunk.itertuples(
                index=False
            ):

                source_id = row.entity_id

                s1_ids = candidate_map.get(
                    source_id
                )

                if not s1_ids:
                    continue

                source = {
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

                for s1_id in s1_ids:

                    feature_rows.append(
                        make_features(
                            s1_lookup[s1_id],
                            source,
                        )
                    )

                    metadata.append(
                        (
                            s1_id,
                            source_id,
                        )
                    )

            if feature_rows:

                X = pd.DataFrame(
                    feature_rows
                )[
                    MODEL_FEATURES
                ].astype(float)

                probabilities = (
                    classifier.predict_proba(
                        X
                    )[:, 1]
                )

                for (
                    (s1_id, source_id),
                    probability,
                ) in zip(
                    metadata,
                    probabilities,
                ):

                    results.append(
                        {
                            "s1_id": s1_id,
                            "source_id": source_id,
                            "probability": float(
                                probability
                            ),
                            "label": int(
                                source_id
                                in truth.get(
                                    s1_id,
                                    set(),
                                )
                            ),
                        }
                    )

            scanned += len(chunk)

        print(
            f"  {source_path.name}: "
            f"{scanned:,} rows scanned"
        )

    return pd.DataFrame(
        results
    )


# ============================================================
# ENTITY-LEVEL METRICS
# ============================================================

def calculate_metrics(
    scored,
    truth,
    validation_ids,
    threshold,
):

    f05_values = []
    precision_values = []
    recall_values = []

    empty_correct = 0

    beta2 = 0.25

    for s1_id in validation_ids:

        true_ids = truth.get(
            s1_id,
            set(),
        )

        subset = scored[
            (
                scored["s1_id"]
                == s1_id
            )
            & (
                scored["probability"]
                >= threshold
            )
        ]

        predicted_ids = set(
            subset["source_id"]
        )

        if (
            not true_ids
            and not predicted_ids
        ):

            empty_correct += 1

            f05_values.append(
                1.0
            )

            precision_values.append(
                1.0
            )

            recall_values.append(
                1.0
            )

            continue

        tp = len(
            true_ids
            & predicted_ids
        )

        fp = len(
            predicted_ids
            - true_ids
        )

        fn = len(
            true_ids
            - predicted_ids
        )

        precision = (
            tp / (tp + fp)
            if tp + fp
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn
            else 0.0
        )

        denominator = (
            beta2 * precision
            + recall
        )

        f05 = (
            1.25
            * precision
            * recall
            / denominator
            if denominator
            else 0.0
        )

        f05_values.append(
            f05
        )

        precision_values.append(
            precision
        )

        recall_values.append(
            recall
        )

    return {
        "threshold": threshold,
        "macro_f0.5": np.mean(
            f05_values
        ),
        "mean_precision": np.mean(
            precision_values
        ),
        "mean_recall": np.mean(
            recall_values
        ),
        "empty_correct": empty_correct,
        "entities": len(
            validation_ids
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "FAST ENTITY-LEVEL VALIDATION"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # Load S1 sample
    # --------------------------------------------------------

    validation_ids, s1_lookup = (
        load_validation_s1()
    )

    print(
        f"S1 loaded: "
        f"{len(s1_lookup):,}"
    )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    truth = load_truth(
        validation_ids
    )

    total_true_pairs = sum(
        len(
            truth.get(
                s1_id,
                set(),
            )
        )
        for s1_id in validation_ids
    )

    print(
        f"True pairs: "
        f"{total_true_pairs:,}"
    )

    # --------------------------------------------------------
    # Build S1 maps
    # --------------------------------------------------------

    print()
    print(
        "Building S1 blocking maps..."
    )

    maps = build_s1_maps(
        s1_lookup
    )

    # --------------------------------------------------------
    # Frequency scan
    # --------------------------------------------------------

    (
        allowed_pairs,
        allowed_address,
    ) = build_allowed_keys(
        maps
    )

    # --------------------------------------------------------
    # Candidate generation
    # --------------------------------------------------------

    candidate_map = (
        generate_candidates(
            s1_lookup,
            maps,
            allowed_pairs,
            allowed_address,
        )
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = joblib.load(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    scored = score_candidates(
        candidate_map,
        s1_lookup,
        truth,
        model,
    )

    print()
    print("=" * 80)
    print("SAVING SCORED PAIRS BEFORE THRESHOLD EVALUATION")
    print("=" * 80)

    scored_output = OUTPUT_DIR / "full_validation_scored_pairs_v2.csv"

    scored.to_csv(
        scored_output,
        index=False,
    )

    print(f"Saved scored pairs: {scored_output}")
    print(f"Rows saved: {len(scored):,}")

    print("Scoring complete. Run scripts/35_evaluate_validation_fast.py separately.")
    raise SystemExit(0)
    print()
    print(
        f"Scored candidate pairs: "
        f"{len(scored):,}"
    )

    # --------------------------------------------------------
    # Blocking recall
    # --------------------------------------------------------

    reachable_true_pairs = int(
        scored["label"].sum()
    )

    blocking_recall = (
        reachable_true_pairs
        / total_true_pairs
        if total_true_pairs
        else 0.0
    )

    print()
    print(
        f"Reachable true pairs: "
        f"{reachable_true_pairs:,}"
    )

    print(
        f"Blocking recall: "
        f"{blocking_recall:.6f}"
    )

    # --------------------------------------------------------
    # Threshold evaluation
    # --------------------------------------------------------

    thresholds = [
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.73,
        0.75,
        0.78,
        0.80,
        0.82,
        0.85,
        0.88,
        0.90,
        0.92,
        0.94,
        0.95,
        0.97,
        0.99,
    ]

    metrics = []

    for threshold in thresholds:

        metrics.append(
            calculate_metrics(
                scored,
                truth,
                validation_ids,
                threshold,
            )
        )

    metrics_df = pd.DataFrame(
        metrics
    )

    metrics_df[
        "blocking_recall"
    ] = blocking_recall

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    metrics_path = (
        OUTPUT_DIR
        / "full_validation_metrics_v2.csv"
    )

    scored_path = (
        OUTPUT_DIR
        / "full_validation_scored_pairs_v2.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    scored.to_csv(
        scored_path,
        index=False,
    )

    # --------------------------------------------------------
    # Best
    # --------------------------------------------------------

    best_index = (
        metrics_df[
            "macro_f0.5"
        ].idxmax()
    )

    best = metrics_df.loc[
        best_index
    ]

    print()
    print("=" * 72)
    print(
        "ENTITY-LEVEL VALIDATION RESULTS"
    )
    print("=" * 72)

    print(
        metrics_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 72)
    print(
        "BEST THRESHOLD"
    )
    print("=" * 72)

    print(
        f"Threshold: "
        f"{best['threshold']:.2f}"
    )

    print(
        f"Macro F0.5: "
        f"{best['macro_f0.5']:.6f}"
    )

    print(
        f"Mean precision: "
        f"{best['mean_precision']:.6f}"
    )

    print(
        f"Mean recall: "
        f"{best['mean_recall']:.6f}"
    )

    print(
        f"Correct empty predictions: "
        f"{int(best['empty_correct']):,}"
    )

    print()
    print(
        f"Saved: {metrics_path}"
    )

    print(
        f"Saved: {scored_path}"
    )


if __name__ == "__main__":
    main()