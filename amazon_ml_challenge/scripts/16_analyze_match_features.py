"""
Analyze similarity features on TRUE S1 -> S2/S3 matches.

This is a bounded training experiment.

It does NOT:
    - build a full candidate index
    - scan source files for every S1
    - perform fuzzy matching over the whole dataset

It:
    1. Samples S1 rows from ground truth.
    2. Retrieves only their known matched S2/S3 records.
    3. Computes similarity features.
    4. Saves the resulting feature table.

This gives us the feature distributions needed to design
the scoring model.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from src.config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TRAIN_GROUND_TRUTH,
    TSV_SEPARATOR,
)

from src.normalization import (
    normalize_name,
    normalize_address,
    token_set,
)


RANDOM_STATE = 42

# Keep this bounded.
S1_SAMPLE_SIZE = 5_000

CHUNK_SIZE = 100_000

OUTPUT_PATH = Path(
    "output/match_feature_sample.csv"
)


USECOLS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]


# ---------------------------------------------------------------------
# Similarity features
# ---------------------------------------------------------------------

def jaro_winkler(a, b):
    if not a or not b:
        return 0.0

    return fuzz.WRatio(a, b) / 100.0


def ratio(a, b):
    if not a or not b:
        return 0.0

    return fuzz.ratio(a, b) / 100.0


def token_set_ratio(a, b):
    if not a or not b:
        return 0.0

    return fuzz.token_set_ratio(a, b) / 100.0


def token_overlap(a, b):
    """
    Jaccard overlap of normalized tokens.
    """

    a_tokens = token_set(a)
    b_tokens = token_set(b)

    if not a_tokens or not b_tokens:
        return 0.0

    return len(a_tokens & b_tokens) / len(
        a_tokens | b_tokens
    )


def containment(a, b):
    """
    Fraction of the smaller token set contained in the larger.
    """

    a_tokens = token_set(a)
    b_tokens = token_set(b)

    if not a_tokens or not b_tokens:
        return 0.0

    intersection = len(a_tokens & b_tokens)

    return intersection / min(
        len(a_tokens),
        len(b_tokens),
    )


def build_features(s1_row, source_row):

    s1_name = normalize_name(
        s1_row["business_name"]
    )

    source_name = normalize_name(
        source_row["business_name"]
    )

    s1_address = normalize_address(
        s1_row["business_address"]
    )

    source_address = normalize_address(
        source_row["business_address"]
    )

    s1_compact_name = s1_name.replace(
        " ",
        "",
    )

    source_compact_name = source_name.replace(
        " ",
        "",
    )

    return {
        "s1_entity_id": s1_row["entity_id"],
        "matched_entity_id": source_row["entity_id"],
        "source": source_row["entity_id"][:2],

        "country_same": int(
            str(s1_row["country"]).casefold()
            ==
            str(source_row["country"]).casefold()
        ),

        # ---------------------------------------------------------
        # Name
        # ---------------------------------------------------------

        "name_exact": int(
            bool(s1_name)
            and s1_name == source_name
        ),

        "name_compact_exact": int(
            bool(s1_compact_name)
            and
            s1_compact_name == source_compact_name
        ),

        "name_ratio": ratio(
            s1_name,
            source_name,
        ),

        "name_token_set_ratio": token_set_ratio(
            s1_name,
            source_name,
        ),

        "name_token_overlap": token_overlap(
            s1_name,
            source_name,
        ),

        "name_containment": containment(
            s1_name,
            source_name,
        ),

        # ---------------------------------------------------------
        # Address
        # ---------------------------------------------------------

        "address_exact": int(
            bool(s1_address)
            and s1_address == source_address
        ),

        "address_ratio": ratio(
            s1_address,
            source_address,
        ),

        "address_token_set_ratio": token_set_ratio(
            s1_address,
            source_address,
        ),

        "address_token_overlap": token_overlap(
            s1_address,
            source_address,
        ),

        "address_containment": containment(
            s1_address,
            source_address,
        ),

        "s1_name": s1_name,
        "matched_name": source_name,
        "s1_address": s1_address,
        "matched_address": source_address,
    }


# ---------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------

def load_sample_ground_truth():

    print("Loading ground truth...")

    gt = pd.read_csv(
        TRAIN_GROUND_TRUTH,
        sep=TSV_SEPARATOR,
        dtype=str,
        keep_default_na=False,
    )

    sample = gt.sample(
        n=S1_SAMPLE_SIZE,
        random_state=RANDOM_STATE,
    ).copy()

    sample.reset_index(drop=True, inplace=True)

    print(
        f"Sampled S1 records: {len(sample):,}"
    )

    # -------------------------------------------------------------
    # Convert comma-separated matches into individual pairs.
    # -------------------------------------------------------------

    pairs = []

    for row in sample.itertuples(index=False):

        if not row.matched_entity_ids:
            continue

        for matched_id in row.matched_entity_ids.split(","):

            matched_id = matched_id.strip()

            if matched_id:
                pairs.append(
                    (
                        row.source1_entity_id,
                        matched_id,
                    )
                )

    pairs_df = pd.DataFrame(
        pairs,
        columns=[
            "s1_entity_id",
            "matched_entity_id",
        ],
    )

    print(
        f"True pairs in sample: {len(pairs_df):,}"
    )

    return sample, pairs_df


# ---------------------------------------------------------------------
# Retrieve S1 records
# ---------------------------------------------------------------------

def load_s1_records(sample):

    s1_ids = set(
        sample["source1_entity_id"]
    )

    result = {}

    print("\nRetrieving S1 records...")

    for chunk in pd.read_csv(
        TRAIN_SOURCE1,
        sep=TSV_SEPARATOR,
        usecols=USECOLS,
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    ):

        mask = chunk["entity_id"].isin(
            s1_ids
        )

        if mask.any():

            for row in chunk.loc[
                mask
            ].to_dict("records"):

                result[row["entity_id"]] = row

    print(
        f"Retrieved S1 records: {len(result):,}"
    )

    return result


# ---------------------------------------------------------------------
# Retrieve matched S2/S3 records
# ---------------------------------------------------------------------

def load_matched_source_records(
    path,
    target_ids,
):

    result = {}

    print(
        f"\nRetrieving matched records from "
        f"{path.name}..."
    )

    for chunk_no, chunk in enumerate(
        pd.read_csv(
            path,
            sep=TSV_SEPARATOR,
            usecols=USECOLS,
            dtype=str,
            keep_default_na=False,
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):

        mask = chunk["entity_id"].isin(
            target_ids
        )

        if mask.any():

            for row in chunk.loc[
                mask
            ].to_dict("records"):

                result[row["entity_id"]] = row

        if chunk_no % 10 == 0:
            print(
                f"  processed "
                f"{chunk_no * CHUNK_SIZE:,} rows...",
                flush=True,
            )

    print(
        f"Retrieved records: {len(result):,}"
    )

    return result


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 72)
    print("TRUE-MATCH FEATURE ANALYSIS")
    print("=" * 72)

    sample, pairs = (
        load_sample_ground_truth()
    )

    # -------------------------------------------------------------
    # Retrieve source records
    # -------------------------------------------------------------

    s1_records = load_s1_records(
        sample
    )

    s2_ids = {
        entity_id
        for entity_id in pairs[
            "matched_entity_id"
        ]
        if entity_id.startswith("S2-")
    }

    s3_ids = {
        entity_id
        for entity_id in pairs[
            "matched_entity_id"
        ]
        if entity_id.startswith("S3-")
    }

    s2_records = load_matched_source_records(
        TRAIN_SOURCE2,
        s2_ids,
    )

    s3_records = load_matched_source_records(
        TRAIN_SOURCE3,
        s3_ids,
    )

    source_records = {
        **s2_records,
        **s3_records,
    }

    # -------------------------------------------------------------
    # Compute features
    # -------------------------------------------------------------

    print("\nComputing similarity features...")

    features = []

    missing_s1 = 0
    missing_source = 0

    for row in pairs.itertuples(
        index=False
    ):

        s1 = s1_records.get(
            row.s1_entity_id
        )

        source = source_records.get(
            row.matched_entity_id
        )

        if s1 is None:
            missing_s1 += 1
            continue

        if source is None:
            missing_source += 1
            continue

        features.append(
            build_features(
                s1,
                source,
            )
        )

    features_df = pd.DataFrame(
        features
    )

    print(
        f"\nFeature rows: {len(features_df):,}"
    )

    if missing_s1:
        print(
            f"Missing S1 records: "
            f"{missing_s1:,}"
        )

    if missing_source:
        print(
            f"Missing matched source records: "
            f"{missing_source:,}"
        )

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------

    numeric_cols = [
        "country_same",
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

    print()
    print("=" * 72)
    print("TRUE-MATCH FEATURE DISTRIBUTIONS")
    print("=" * 72)

    summary = (
        features_df[numeric_cols]
        .describe(
            percentiles=[
                0.01,
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
                0.99,
            ]
        )
        .T
    )

    print(
        summary.to_string()
    )

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    features_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()