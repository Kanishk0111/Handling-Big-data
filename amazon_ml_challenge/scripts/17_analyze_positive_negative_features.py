"""
Compare TRUE matches against controlled false pairs.

The positive pairs come directly from the previously generated
match_feature_sample.csv.

Negative pairs are created by shuffling matched source records
among different S1 records.

This is a diagnostic experiment for feature separation.
It is NOT the final classifier-training dataset.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from src.normalization import (
    token_set,
    normalize_name,
    normalize_address,
)


INPUT_PATH = Path(
    "output/match_feature_sample.csv"
)

OUTPUT_PATH = Path(
    "output/positive_negative_feature_sample.csv"
)

RANDOM_STATE = 42


def ratio(a, b):
    if not a or not b:
        return 0.0

    return fuzz.ratio(a, b) / 100.0


def token_set_ratio(a, b):
    if not a or not b:
        return 0.0

    return fuzz.token_set_ratio(a, b) / 100.0


def token_overlap(a, b):
    a_tokens = token_set(a)
    b_tokens = token_set(b)

    if not a_tokens or not b_tokens:
        return 0.0

    return len(a_tokens & b_tokens) / len(
        a_tokens | b_tokens
    )


def containment(a, b):
    a_tokens = token_set(a)
    b_tokens = token_set(b)

    if not a_tokens or not b_tokens:
        return 0.0

    return len(a_tokens & b_tokens) / min(
        len(a_tokens),
        len(b_tokens),
    )


def calculate_features(
    s1_name,
    source_name,
    s1_address,
    source_address,
):
    s1_name = normalize_name(s1_name)
    source_name = normalize_name(source_name)

    s1_address = normalize_address(s1_address)
    source_address = normalize_address(source_address)

    s1_compact = s1_name.replace(" ", "")
    source_compact = source_name.replace(" ", "")

    return {
        "name_exact": int(
            bool(s1_name)
            and s1_name == source_name
        ),

        "name_compact_exact": int(
            bool(s1_compact)
            and s1_compact == source_compact
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
    }


def main():

    print("=" * 72)
    print("POSITIVE VS CONTROLLED NEGATIVE FEATURE ANALYSIS")
    print("=" * 72)

    df = pd.read_csv(
        INPUT_PATH,
        dtype=str,
        keep_default_na=False,
    )

    print(
        f"Positive rows: {len(df):,}"
    )

    # -------------------------------------------------------------
    # Positive examples
    # -------------------------------------------------------------

    positive = df[
        [
            "s1_entity_id",
            "matched_entity_id",
            "s1_name",
            "matched_name",
            "s1_address",
            "matched_address",
        ]
    ].copy()

    positive["label"] = 1

    # -------------------------------------------------------------
    # Create controlled negatives.
    #
    # Shift source records by a large offset so that each S1 gets
    # another sampled source record.
    #
    # We also require the source entity to differ.
    # -------------------------------------------------------------

    n = len(df)

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    permutation = rng.permutation(n)

    # Fix accidental self-pairs.
    for i in range(n):
        if permutation[i] == i:
            j = (i + 1) % n

            permutation[i], permutation[j] = (
                permutation[j],
                permutation[i],
            )

    negative = pd.DataFrame(
        {
            "s1_entity_id": df[
                "s1_entity_id"
            ].values,

            "matched_entity_id": df[
                "matched_entity_id"
            ].values[permutation],

            "s1_name": df[
                "s1_name"
            ].values,

            "matched_name": df[
                "matched_name"
            ].values[permutation],

            "s1_address": df[
                "s1_address"
            ].values,

            "matched_address": df[
                "matched_address"
            ].values[permutation],
        }
    )

    negative["label"] = 0

    # -------------------------------------------------------------
    # Calculate features for negatives.
    # -------------------------------------------------------------

    print(
        "\nCalculating negative features..."
    )

    negative_features = []

    for row in negative.itertuples(
        index=False
    ):

        features = calculate_features(
            row.s1_name,
            row.matched_name,
            row.s1_address,
            row.matched_address,
        )

        negative_features.append(
            features
        )

    negative_features = pd.DataFrame(
        negative_features
    )

    negative = pd.concat(
        [
            negative.reset_index(drop=True),
            negative_features,
        ],
        axis=1,
    )

    # -------------------------------------------------------------
    # Positives already have features, so select them.
    # -------------------------------------------------------------

    feature_columns = [
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

    positive_features = positive[
        [
            "s1_entity_id",
            "matched_entity_id",
            "s1_name",
            "matched_name",
            "s1_address",
            "matched_address",
            "label",
        ]
    ].copy()

    positive_features = pd.concat(
        [
            positive_features.reset_index(drop=True),
            df[
                feature_columns
            ].reset_index(drop=True),
        ],
        axis=1,
    )

    # -------------------------------------------------------------
    # Combine
    # -------------------------------------------------------------

    combined = pd.concat(
        [
            positive_features,
            negative,
        ],
        ignore_index=True,
    )

    combined.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------

    print()
    print("=" * 72)
    print("FEATURE COMPARISON")
    print("=" * 72)

    summary = (
        combined
        .groupby("label")[feature_columns]
        .agg(
            [
                "mean",
                "median",
                lambda x: np.percentile(x, 95),
            ]
        )
    )

    print(
        summary.to_string()
    )

    print()
    print(
        "1 = true match, 0 = controlled negative"
    )

    print()
    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()