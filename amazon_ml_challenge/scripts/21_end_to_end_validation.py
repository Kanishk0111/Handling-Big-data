from pathlib import Path
import random

import joblib
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from src.config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TRAIN_GROUND_TRUTH,
)
from src.normalization import normalize_name, normalize_address


SEED = 42

# Keep the first end-to-end experiment small.
VALIDATION_SAMPLE_SIZE = 5_000

# Process the 10M candidate rows in chunks.
CHUNK_SIZE = 100_000

MODEL_FILE = Path(
    "output/entity_match_baseline.joblib"
)

VALIDATION_IDS_FILE = Path(
    "output/validation_s1_ids.txt"
)

OUTPUT_FILE = Path(
    "output/end_to_end_validation_predictions.csv"
)


# ------------------------------------------------------------
# Blocking
# ------------------------------------------------------------

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


def country_key(value):
    if value is None:
        return ""

    return str(value).strip().casefold()


def compact_text(value):
    if not value:
        return ""

    return "".join(
        str(value).split()
    )


def token_list(value):
    if not value:
        return []

    return str(value).split()


def make_block_keys(row):
    country = country_key(row["country"])
    name = normalize_name(row["business_name"])
    compact = compact_text(row["business_name"])
    address = normalize_address(row["business_address"])

    keys = []

    if country and name:
        keys.append(
            f"{country}|name_exact|{name}"
        )

    if country and compact:
        keys.append(
            f"{country}|compact_name|{compact}"
        )

    # Informative name-token pairs.
    tokens = [
        t
        for t in token_list(row["business_name"])
        if len(t) >= 3
        and t not in COMMON_BUSINESS_TOKENS
    ]

    unique_tokens = sorted(set(tokens))

    for i in range(len(unique_tokens)):
        for j in range(i + 1, len(unique_tokens)):
            keys.append(
                f"{country}|name_pair|"
                f"{unique_tokens[i]}|{unique_tokens[j]}"
            )

    # Conditional fuzzy-recovery block:
    # name prefix AND house number must agree.
    if country and compact and len(compact) >= 8:
        prefix = compact[:8]

        house_number = ""

        for token in address.split():
            if any(char.isdigit() for char in token):
                house_number = token
                break

        if house_number:
            keys.append(
                f"{country}|prefix_house|"
                f"{prefix}|{house_number}"
            )

    return keys


# ------------------------------------------------------------
# Features
# ------------------------------------------------------------

def token_set(text):
    if not text:
        return set()

    return set(text.split())


def overlap(a, b):
    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def containment(a, b):
    if not a or not b:
        return 0.0

    return len(a & b) / min(
        len(a),
        len(b),
    )


def make_features(s1, candidate):

    name1 = s1["name"]
    name2 = candidate["name"]

    address1 = s1["address"]
    address2 = candidate["address"]

    name_tokens1 = token_set(name1)
    name_tokens2 = token_set(name2)

    address_tokens1 = token_set(address1)
    address_tokens2 = token_set(address2)

    return {
        # Kept for feature computation compatibility,
        # but NOT passed to the saved baseline model.
        "country_same": int(
            s1["country"] == candidate["country"]
        ),

        "name_exact": int(
            bool(name1)
            and name1 == name2
        ),

        "name_compact_exact": int(
            bool(name1)
            and name1.replace(" ", "")
            == name2.replace(" ", "")
        ),

        "name_ratio": (
            fuzz.ratio(
                name1,
                name2,
            ) / 100.0
        ),

        "name_token_set_ratio": (
            fuzz.token_set_ratio(
                name1,
                name2,
            ) / 100.0
        ),

        "name_token_overlap": overlap(
            name_tokens1,
            name_tokens2,
        ),

        "name_containment": containment(
            name_tokens1,
            name_tokens2,
        ),

        "address_exact": int(
            bool(address1)
            and address1 == address2
        ),

        "address_ratio": (
            fuzz.ratio(
                address1,
                address2,
            ) / 100.0
        ),

        "address_token_set_ratio": (
            fuzz.token_set_ratio(
                address1,
                address2,
            ) / 100.0
        ),

        "address_token_overlap": overlap(
            address_tokens1,
            address_tokens2,
        ),

        "address_containment": containment(
            address_tokens1,
            address_tokens2,
        ),
    }


# ------------------------------------------------------------
# F0.5
# ------------------------------------------------------------

def f05(precision, recall):

    if precision == 0 and recall == 0:
        return 0.0

    beta = 0.5

    return (
        (1 + beta ** 2)
        * precision
        * recall
        / (
            beta ** 2 * precision
            + recall
        )
    )


def entity_f05(predicted, truth):

    predicted = set(predicted)
    truth = set(truth)

    # Challenge singleton behavior:
    # correct empty prediction for empty truth = 1.0.
    if not predicted and not truth:
        return 1.0

    if not predicted and truth:
        return 0.0

    if predicted and not truth:
        return 0.0

    tp = len(predicted & truth)

    precision = (
        tp / len(predicted)
        if predicted
        else 0.0
    )

    recall = (
        tp / len(truth)
        if truth
        else 0.0
    )

    return f05(
        precision,
        recall,
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 72)
    print("END-TO-END ENTITY RESOLUTION VALIDATION")
    print("=" * 72)

    # --------------------------------------------------------
    # Load provisional model
    # --------------------------------------------------------

    print("\nLoading provisional model...")

    bundle = joblib.load(
        MODEL_FILE
    )

    model = bundle["model"]

    # IMPORTANT:
    # The saved model was actually trained on 11 features.
    # Do NOT pass country_same to this model.
    model_features = [
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

    print(
        "Model features:",
        len(model_features),
    )

    # --------------------------------------------------------
    # Validation IDs
    # --------------------------------------------------------

    validation_ids = [
        line.strip()
        for line in VALIDATION_IDS_FILE.read_text()
        .splitlines()
        if line.strip()
    ]

    rng = random.Random(SEED)

    rng.shuffle(validation_ids)

    validation_ids = validation_ids[
        :VALIDATION_SAMPLE_SIZE
    ]

    validation_id_set = set(
        validation_ids
    )

    print(
        f"\nValidation S1 sample: "
        f"{len(validation_ids):,}"
    )

    # --------------------------------------------------------
    # Load S1 sample
    # --------------------------------------------------------

    print("\nLoading S1 validation records...")

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

    s1_records = {}

    required_keys = set()

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

        record = {
            "entity_id": row.entity_id,
            "country": country,
            "name": name,
            "address": address,
        }

        s1_records[
            row.entity_id
        ] = record

        block_row = {
            "country": country,
            "business_name": name,
            "business_address": address,
        }

        required_keys.update(
            make_block_keys(
                block_row
            )
        )

    del s1_df

    print(
        f"S1 records loaded: "
        f"{len(s1_records):,}"
    )

    print(
        f"Required blocking keys: "
        f"{len(required_keys):,}"
    )

    # --------------------------------------------------------
    # Candidate index
    # --------------------------------------------------------

    # IMPORTANT:
    # candidate_records stores each candidate once.
    # candidate_index maps blocking keys -> entity IDs.
    # We only index keys actually needed by sampled S1s.
    # We do NOT construct the global index again.
    # --------------------------------------------------------

    candidate_records = {}
    candidate_index = {}

    total_source_rows = 0
    retained_candidate_rows = 0

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

                total_source_rows += 1

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

                block_row = {
                    "country": country,
                    "business_name": name,
                    "business_address": address,
                }

                keys = make_block_keys(
                    block_row
                )

                matching_keys = (
                    set(keys) & required_keys
                )

                if not matching_keys:
                    continue

                entity_id = row.entity_id

                candidate = {
                    "entity_id": entity_id,
                    "country": country,
                    "name": name,
                    "address": address,
                }

                candidate_records[
                    entity_id
                ] = candidate

                retained_candidate_rows += 1

                for key in matching_keys:

                    bucket = candidate_index.setdefault(
                        key,
                        set()
                    )

                    bucket.add(entity_id)

            print(
                f"  scanned: "
                f"{total_source_rows:,}"
                f" | retained: "
                f"{retained_candidate_rows:,}"
            )

    print(
        "\nCandidate index keys:",
        f"{len(candidate_index):,}",
    )

    print(
        "Candidate records:",
        f"{len(candidate_records):,}",
    )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    print("\nLoading ground truth...")

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

        truth[
            row.source1_entity_id
        ] = {
            x.strip()
            for x in row.matched_entity_ids.split(",")
            if x.strip()
        }

    # --------------------------------------------------------
    # Generate candidates and score
    # --------------------------------------------------------

    print(
        "\nGenerating and scoring candidates..."
    )

    results = []

    candidate_counts = []

    blocking_recovered_pairs = 0
    total_true_pairs = 0

    for counter, s1_id in enumerate(
        validation_ids,
        start=1,
    ):

        s1 = s1_records[s1_id]

        true_ids = truth.get(
            s1_id,
            set(),
        )

        total_true_pairs += len(
            true_ids
        )

        candidate_ids = set()

        block_row = {
            "country": s1["country"],
            "business_name": s1["name"],
            "business_address": s1["address"],
        }

        for key in make_block_keys(
            block_row
        ):

            bucket = candidate_index.get(
                key
            )

            if bucket:
                candidate_ids.update(
                    bucket
                )

        candidate_counts.append(
            len(candidate_ids)
        )

        blocking_recovered_pairs += len(
            candidate_ids & true_ids
        )

        scored = []

        # Direct candidate retrieval.
        # No repeated search through blocking buckets.
        for candidate_id in candidate_ids:

            candidate = candidate_records[
                candidate_id
            ]

            feature_row = make_features(
                s1,
                candidate,
            )

            X = pd.DataFrame(
                [feature_row]
            )[model_features]

            score = float(
                model.predict_proba(X)[0, 1]
            )

            scored.append(
                (
                    candidate_id,
                    score,
                )
            )

        # ----------------------------------------------------
        # Preliminary decision rule.
        #
        # Keep threshold at 0.50 for this diagnostic run.
        # Do NOT tune it yet.
        # ----------------------------------------------------

        predicted = [
            candidate_id
            for candidate_id, score in scored
            if score >= 0.50
        ]

        score_values = [
            score
            for _, score in scored
        ]

        top_score = (
            max(score_values)
            if score_values
            else 0.0
        )

        results.append({
            "source1_entity_id": s1_id,
            "true_count": len(true_ids),
            "candidate_count": len(candidate_ids),
            "true_candidates": len(
                candidate_ids & true_ids
            ),
            "predicted_count": len(predicted),
            "correct_predictions": len(
                set(predicted) & true_ids
            ),
            "f05": entity_f05(
                predicted,
                true_ids,
            ),
            "top_score": top_score,
        })

        if counter % 250 == 0:
            print(
                f"  Processed "
                f"{counter:,} / "
                f"{len(validation_ids):,}"
            )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    blocking_recall = (
        blocking_recovered_pairs
        / total_true_pairs
        if total_true_pairs
        else 0.0
    )

    macro_f05 = results_df[
        "f05"
    ].mean()

    mean_candidates = results_df[
        "candidate_count"
    ].mean()

    median_candidates = results_df[
        "candidate_count"
    ].median()

    p95_candidates = results_df[
        "candidate_count"
    ].quantile(0.95)

    no_candidate_rate = (
        (
            results_df[
                "candidate_count"
            ] == 0
        ).mean()
    )

    singleton_rows = (
        results_df["true_count"] == 0
    )

    singleton_accuracy = (
        results_df.loc[
            singleton_rows,
            "predicted_count"
        ] == 0
    ).mean()

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("END-TO-END VALIDATION RESULTS")
    print("=" * 72)

    print(
        f"Validation S1s: "
        f"{len(results_df):,}"
    )

    print(
        f"S2/S3 rows scanned: "
        f"{total_source_rows:,}"
    )

    print(
        f"Candidate index keys: "
        f"{len(candidate_index):,}"
    )

    print(
        f"Candidate pairs: "
        f"{sum(candidate_counts):,}"
    )

    print(
        f"True pairs: "
        f"{total_true_pairs:,}"
    )

    print(
        f"True pairs recovered: "
        f"{blocking_recovered_pairs:,}"
    )

    print(
        f"Blocking recall: "
        f"{blocking_recall:.4f}"
    )

    print(
        f"Mean candidates / S1: "
        f"{mean_candidates:.2f}"
    )

    print(
        f"Median candidates / S1: "
        f"{median_candidates:.0f}"
    )

    print(
        f"P95 candidates / S1: "
        f"{p95_candidates:.0f}"
    )

    print(
        f"No-candidate rate: "
        f"{no_candidate_rate:.4f}"
    )

    print(
        f"Singleton accuracy: "
        f"{singleton_accuracy:.4f}"
    )

    print(
        f"Entity-level F0.5: "
        f"{macro_f05:.4f}"
    )

    print(
        f"\nSaved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()