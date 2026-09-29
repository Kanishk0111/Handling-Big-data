from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TRAIN_GROUND_TRUTH,
    DEFAULT_CHUNK_SIZE,
)
from src.blocking import generate_blocking_keys


BASE_STRATEGIES = [
    "name_prefix",
    "address_prefix",
    "house_number",
]

COMBINATIONS = {
    "name_prefix": {"name_prefix"},
    "address_prefix": {"address_prefix"},
    "house_number": {"house_number"},
    "name_prefix + house_number": {
        "name_prefix",
        "house_number",
    },
    "name_prefix + address_prefix": {
        "name_prefix",
        "address_prefix",
    },
    "address_prefix + house_number": {
        "address_prefix",
        "house_number",
    },
    "ALL THREE": {
        "name_prefix",
        "address_prefix",
        "house_number",
    },
}


def load_ground_truth():
    df = pd.read_csv(
        TRAIN_GROUND_TRUTH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    truth = {}

    for row in df.itertuples(index=False):
        matches = set()

        if row.matched_entity_ids:
            text = str(row.matched_entity_ids)

            for separator in [",", "|", ";"]:
                text = text.replace(separator, " ")

            for token in text.split():
                if token.startswith("S2-") or token.startswith("S3-"):
                    matches.add(token)

        truth[row.source1_entity_id] = matches

    return truth


def load_s1_keys():
    print("Loading S1...")

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

    print(f"S1 rows: {len(df):,}")

    keys_by_id = {}

    for row in df.itertuples(index=False):
        keys_by_id[row.entity_id] = generate_blocking_keys(
            row.country,
            row.business_name,
            row.business_address,
        )

    return keys_by_id


def collect_true_source_ids(truth):
    s2_ids = set()
    s3_ids = set()

    for matches in truth.values():
        for entity_id in matches:
            if entity_id.startswith("S2-"):
                s2_ids.add(entity_id)
            elif entity_id.startswith("S3-"):
                s3_ids.add(entity_id)

    return s2_ids, s3_ids


def evaluate_source(
    source_path,
    source_name,
    needed_ids,
    s1_keys,
    truth,
):
    """
    Evaluate all blocking combinations in one pass.

    Only ground-truth source records are evaluated.
    """

    recovered = {
        name: 0
        for name in COMBINATIONS
    }

    true_reverse = {}

    for s1_id, matches in truth.items():
        for source_id in matches:
            if source_id in needed_ids:
                true_reverse.setdefault(
                    source_id,
                    []
                ).append(s1_id)

    processed = 0
    relevant = 0

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
        processed += len(chunk)

        relevant_chunk = chunk[
            chunk["entity_id"].isin(needed_ids)
        ]

        if len(relevant_chunk) == 0:
            continue

        relevant += len(relevant_chunk)

        for row in relevant_chunk.itertuples(index=False):

            source_id = row.entity_id

            source_keys = generate_blocking_keys(
                row.country,
                row.business_name,
                row.business_address,
            )

            true_s1_ids = true_reverse.get(
                source_id,
                [],
            )

            for s1_id in true_s1_ids:

                s1_key_dict = s1_keys[s1_id]

                for combination_name, strategies in COMBINATIONS.items():

                    # A true pair is recovered if ANY key in the
                    # combination agrees.
                    matched = False

                    for strategy in strategies:

                        source_key = source_keys[strategy]
                        s1_key = s1_key_dict[strategy]

                        if (
                            source_key
                            and s1_key
                            and source_key == s1_key
                        ):
                            matched = True
                            break

                    if matched:
                        recovered[combination_name] += 1

        if processed % 1_000_000 == 0:
            print(
                f"  {source_name}: "
                f"{processed:,} rows scanned; "
                f"{relevant:,} true-match records"
            )

    print(
        f"  {source_name}: finished "
        f"({processed:,} rows scanned)"
    )

    return recovered


def main():

    print("Loading ground truth...")
    truth = load_ground_truth()

    total_true_pairs = sum(
        len(matches)
        for matches in truth.values()
    )

    print(
        f"Ground-truth S1 entities: "
        f"{len(truth):,}"
    )

    print(
        f"Total true pairs: "
        f"{total_true_pairs:,}"
    )

    print("\nCollecting true source IDs...")

    s2_ids, s3_ids = collect_true_source_ids(truth)

    print(
        f"Unique S2 IDs in ground truth: "
        f"{len(s2_ids):,}"
    )

    print(
        f"Unique S3 IDs in ground truth: "
        f"{len(s3_ids):,}"
    )

    s1_keys = load_s1_keys()

    print("\n" + "=" * 100)
    print("EVALUATING S2")
    print("=" * 100)

    s2_recovered = evaluate_source(
        TRAIN_SOURCE2,
        "S2",
        s2_ids,
        s1_keys,
        truth,
    )

    print("\n" + "=" * 100)
    print("EVALUATING S3")
    print("=" * 100)

    s3_recovered = evaluate_source(
        TRAIN_SOURCE3,
        "S3",
        s3_ids,
        s1_keys,
        truth,
    )

    print("\n" + "=" * 100)
    print("BLOCKING RECALL")
    print("=" * 100)

    results = []

    for combination_name in COMBINATIONS:

        recovered = (
            s2_recovered[combination_name]
            + s3_recovered[combination_name]
        )

        recall = (
            recovered / total_true_pairs
        )

        results.append(
            {
                "strategy": combination_name,
                "recovered_pairs": recovered,
                "total_true_pairs": total_true_pairs,
                "recall": recall * 100,
            }
        )

    result_df = pd.DataFrame(results)

    print(
        result_df.to_string(
            index=False,
            formatters={
                "recall": "{:.4f}%".format
            },
        )
    )


if __name__ == "__main__":
    main()