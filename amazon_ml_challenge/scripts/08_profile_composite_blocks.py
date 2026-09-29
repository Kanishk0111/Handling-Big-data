from pathlib import Path
import sys
from collections import defaultdict

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TRAIN_GROUND_TRUTH,
    DEFAULT_CHUNK_SIZE,
)
from src.normalization import compact_text, normalize_name
from src.blocking import (
    country_key,
    token_list,
)


SAMPLE_SIZE = 20_000
RANDOM_STATE = 42


BLOCKS = [
    "exact_name",
    "compact_name",
    "name_prefix_8_house",
    "name_prefix_8_address",
    "name_prefix_10",
    "name_prefix_12",
]


def make_key(
    country,
    business_name,
    business_address,
    block,
):
    country = country_key(country)
    name = normalize_name(business_name)
    compact = compact_text(business_name)

    address_tokens = token_list(
        business_address
    )

    if block == "exact_name":
        if not country or not name:
            return ""
        return f"{country}|name|{name}"

    if block == "compact_name":
        if not country or not compact:
            return ""
        return f"{country}|compact|{compact}"

    if block == "name_prefix_8_house":
        if not country or len(compact) < 8:
            return ""

        house = ""

        for token in normalize_name(
            business_address
        ).split():
            if any(char.isdigit() for char in token):
                house = token
                break

        if not house:
            return ""

        return (
            f"{country}|np8|{compact[:8]}"
            f"|house|{house}"
        )

    if block == "name_prefix_8_address":
        if not country or len(compact) < 8:
            return ""

        if len(address_tokens) < 3:
            return ""

        return (
            f"{country}|np8|{compact[:8]}"
            f"|addr|"
            f"{address_tokens[0]}|"
            f"{address_tokens[1]}|"
            f"{address_tokens[2]}"
        )

    if block == "name_prefix_10":
        if not country or len(compact) < 10:
            return ""
        return (
            f"{country}|np10|{compact[:10]}"
        )

    if block == "name_prefix_12":
        if not country or len(compact) < 12:
            return ""
        return (
            f"{country}|np12|{compact[:12]}"
        )

    raise ValueError(
        f"Unknown block: {block}"
    )


def load_s1_sample():
    df = pd.read_csv(
        TRAIN_SOURCE1,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
    )

    return df.sample(
        n=min(SAMPLE_SIZE, len(df)),
        random_state=RANDOM_STATE,
    ).copy()


def load_truth_for_sample(sample):
    sample_ids = set(
        sample["entity_id"]
    )

    truth_df = pd.read_csv(
        TRAIN_GROUND_TRUTH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    truth = {}

    for row in truth_df.itertuples(index=False):

        if row.source1_entity_id not in sample_ids:
            continue

        matches = set()

        if row.matched_entity_ids:
            text = str(
                row.matched_entity_ids
            )

            for separator in [
                ",",
                "|",
                ";",
            ]:
                text = text.replace(
                    separator,
                    " ",
                )

            for token in text.split():
                if (
                    token.startswith("S2-")
                    or token.startswith("S3-")
                ):
                    matches.add(token)

        truth[row.source1_entity_id] = matches

    return truth


def build_sample_block_keys(sample):
    """
    block -> key -> set(S1 IDs)

    Only sample S1 records are indexed.
    """

    indexes = {
        block: defaultdict(set)
        for block in BLOCKS
    }

    for row in sample.itertuples(
        index=False
    ):
        for block in BLOCKS:

            key = make_key(
                row.country,
                row.business_name,
                row.business_address,
                block,
            )

            if key:
                indexes[block][key].add(
                    row.entity_id
                )

    return indexes


def build_candidate_sets(
    sample,
    source_path,
    indexes,
):
    """
    Build exact deduplicated candidate sets
    for every sampled S1 entity.

    Only source records sharing a block with at
    least one sampled S1 record are retained.
    """

    candidates = {
        block: defaultdict(set)
        for block in BLOCKS
    }

    # Reverse lookup:
    # block -> key -> sampled S1 IDs
    for chunk in pd.read_csv(
        source_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
        chunksize=DEFAULT_CHUNK_SIZE,
    ):

        for row in chunk.itertuples(
            index=False
        ):

            for block in BLOCKS:

                key = make_key(
                    row.country,
                    row.business_name,
                    row.business_address,
                    block,
                )

                if not key:
                    continue

                s1_ids = indexes[
                    block
                ].get(key)

                if not s1_ids:
                    continue

                for s1_id in s1_ids:
                    candidates[block][
                        s1_id
                    ].add(
                        row.entity_id
                    )

    return candidates


def evaluate(
    sample,
    truth,
    candidates,
):
    results = []

    total_truth = sum(
        len(v)
        for v in truth.values()
    )

    for block in BLOCKS:

        block_candidates = candidates[
            block
        ]

        candidate_counts = []

        recovered = 0

        for s1_id in sample[
            "entity_id"
        ]:

            ids = block_candidates.get(
                s1_id,
                set(),
            )

            candidate_counts.append(
                len(ids)
            )

            recovered += len(
                ids.intersection(
                    truth.get(
                        s1_id,
                        set(),
                    )
                )
            )

        counts = np.asarray(
            candidate_counts
        )

        recall = (
            recovered / total_truth
            if total_truth
            else 0
        )

        results.append(
            {
                "block": block,
                "mean": counts.mean(),
                "median": np.percentile(
                    counts,
                    50,
                ),
                "p95": np.percentile(
                    counts,
                    95,
                ),
                "p99": np.percentile(
                    counts,
                    99,
                ),
                "max": counts.max(),
                "zero_s1": (
                    counts == 0
                ).sum(),
                "recovered_truth": recovered,
                "truth_recall": recall,
            }
        )

    # Union of all blocks.
    union_counts = []

    union_recovered = 0

    for s1_id in sample[
        "entity_id"
    ]:

        combined = set()

        for block in BLOCKS:

            combined.update(
                candidates[
                    block
                ].get(
                    s1_id,
                    set(),
                )
            )

        union_counts.append(
            len(combined)
        )

        union_recovered += len(
            combined.intersection(
                truth.get(
                    s1_id,
                    set(),
                )
            )
        )

    union_counts = np.asarray(
        union_counts
    )

    results.append(
        {
            "block": "ALL BLOCKS UNION",
            "mean": union_counts.mean(),
            "median": np.percentile(
                union_counts,
                50,
            ),
            "p95": np.percentile(
                union_counts,
                95,
            ),
            "p99": np.percentile(
                union_counts,
                99,
            ),
            "max": union_counts.max(),
            "zero_s1": (
                union_counts == 0
            ).sum(),
            "recovered_truth": union_recovered,
            "truth_recall": (
                union_recovered
                / total_truth
            ),
        }
    )

    return pd.DataFrame(results)


def main():

    print("Loading S1 sample...")

    sample = load_s1_sample()

    print(
        f"Sample size: "
        f"{len(sample):,}"
    )

    print(
        "\nLoading ground truth "
        "for sample..."
    )

    truth = load_truth_for_sample(
        sample
    )

    total_truth = sum(
        len(v)
        for v in truth.values()
    )

    print(
        f"True pairs for sample: "
        f"{total_truth:,}"
    )

    print(
        "\nBuilding sample S1 block indexes..."
    )

    indexes = build_sample_block_keys(
        sample
    )

    print(
        "\nScanning S2..."
    )

    s2_candidates = build_candidate_sets(
        sample,
        TRAIN_SOURCE2,
        indexes,
    )

    print(
        "\nScanning S3..."
    )

    s3_candidates = build_candidate_sets(
        sample,
        TRAIN_SOURCE3,
        indexes,
    )

    # Merge S2 + S3 candidate sets.
    candidates = {
        block: defaultdict(set)
        for block in BLOCKS
    }

    for block in BLOCKS:

        all_s1_ids = set(
            s2_candidates[block]
        ) | set(
            s3_candidates[block]
        )

        for s1_id in all_s1_ids:

            candidates[
                block
            ][s1_id] = (
                s2_candidates[
                    block
                ].get(
                    s1_id,
                    set(),
                )
                |
                s3_candidates[
                    block
                ].get(
                    s1_id,
                    set(),
                )
            )

    print(
        "\nCalculating exact candidate "
        "statistics..."
    )

    result = evaluate(
        sample,
        truth,
        candidates,
    )

    result["truth_recall"] *= 100

    print(
        "\n" + "=" * 110
    )
    print(
        "COMPOSITE BLOCK RESULTS"
    )
    print(
        "=" * 110
    )

    print(
        result.to_string(
            index=False,
            formatters={
                "mean": "{:,.1f}".format,
                "median": "{:,.0f}".format,
                "p95": "{:,.0f}".format,
                "p99": "{:,.0f}".format,
                "max": "{:,.0f}".format,
                "truth_recall": "{:.4f}%".format,
            },
        )
    )


if __name__ == "__main__":
    main()