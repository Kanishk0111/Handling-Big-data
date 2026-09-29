from pathlib import Path

import pandas as pd

from src.config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TRAIN_GROUND_TRUTH,
)
from src.blocking import (
    generate_blocking_keys,
    name_tokens_key,
    name_token_pair_keys,
)


RANDOM_STATE = 42
SAMPLE_SIZE = 20_000
MAX_EXAMPLES = 40


def get_keys(row):
    keys = generate_blocking_keys(
        row["country"],
        row["business_name"],
        row["business_address"],
    )

    token_keys = name_tokens_key(
        row["country"],
        row["business_name"],
    )

    pair_keys = name_token_pair_keys(
        row["country"],
        row["business_name"],
    )

    return keys, token_keys, pair_keys


def main():
    print("=" * 80)
    print("ANALYZE MISSED TRUE MATCHES")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Ground truth sample
    # ------------------------------------------------------------------

    gt = pd.read_csv(
        TRAIN_GROUND_TRUTH,
        sep="\t",
        dtype=str,
    )

    gt = gt.sample(
        n=min(SAMPLE_SIZE, len(gt)),
        random_state=RANDOM_STATE,
    )

    # ------------------------------------------------------------------
    # Extract true pairs
    # ------------------------------------------------------------------

    true_pairs = []

    for _, row in gt.iterrows():
        matched = str(row["matched_entity_ids"])

        if not matched or matched.lower() == "nan":
            continue

        for entity_id in matched.split(","):
            entity_id = entity_id.strip()

            if entity_id:
                true_pairs.append(
                    (
                        row["source1_entity_id"],
                        entity_id,
                    )
                )

    truth = pd.DataFrame(
        true_pairs,
        columns=[
            "source1_entity_id",
            "matched_entity_id",
        ],
    )

    print(f"Sampled S1 entities : {len(gt):,}")
    print(f"True matched pairs  : {len(truth):,}")

    # ------------------------------------------------------------------
    # Load sampled S1
    # ------------------------------------------------------------------

    s1_ids = set(gt["source1_entity_id"])

    print("\nLoading S1 sample...")

    s1_parts = []

    for chunk in pd.read_csv(
        TRAIN_SOURCE1,
        sep="\t",
        dtype=str,
        chunksize=100_000,
    ):
        part = chunk[
            chunk["entity_id"].isin(s1_ids)
        ]

        if not part.empty:
            s1_parts.append(part)

    s1 = pd.concat(
        s1_parts,
        ignore_index=True,
    )

    s1_by_id = {
        row["entity_id"]: row
        for _, row in s1.iterrows()
    }

    # ------------------------------------------------------------------
    # Load true S2/S3 records
    # ------------------------------------------------------------------

    true_ids = set(
        truth["matched_entity_id"]
    )

    source_files = [
        ("S2", TRAIN_SOURCE2),
        ("S3", TRAIN_SOURCE3),
    ]

    source_parts = []

    for source_name, path in source_files:

        print(f"Scanning {source_name}...")

        parts = []

        for chunk in pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            chunksize=100_000,
        ):
            part = chunk[
                chunk["entity_id"].isin(true_ids)
            ]

            if not part.empty:
                part = part.copy()
                part["source"] = source_name
                parts.append(part)

        if parts:
            source_parts.append(
                pd.concat(
                    parts,
                    ignore_index=True,
                )
            )

    sources = pd.concat(
        source_parts,
        ignore_index=True,
    )

    source_by_id = {
        row["entity_id"]: row
        for _, row in sources.iterrows()
    }

    # ------------------------------------------------------------------
    # Find missed pairs
    # ------------------------------------------------------------------

    missed = []

    for _, pair in truth.iterrows():

        s1_row = s1_by_id.get(
            pair["source1_entity_id"]
        )

        source_row = source_by_id.get(
            pair["matched_entity_id"]
        )

        if s1_row is None or source_row is None:
            continue

        s1_blocks, s1_tokens, s1_pairs = get_keys(
            s1_row
        )

        source_blocks, source_tokens, source_pairs = get_keys(
            source_row
        )

        # ALL blocking rules currently available
        s1_all = (
            set(v for v in s1_blocks.values() if v)
            | set(s1_tokens)
            | set(s1_pairs)
        )

        source_all = (
            set(v for v in source_blocks.values() if v)
            | set(source_tokens)
            | set(source_pairs)
        )

        if not (s1_all & source_all):
            missed.append(
                {
                    "s1_id": pair["source1_entity_id"],
                    "matched_id": pair["matched_entity_id"],
                    "source": source_row["source"],

                    "country": s1_row["country"],

                    "s1_name": s1_row["business_name"],
                    "matched_name": source_row["business_name"],

                    "s1_address": s1_row["business_address"],
                    "matched_address": source_row["business_address"],

                    "s1_name_prefix": s1_blocks["name_prefix"],
                    "matched_name_prefix": source_blocks["name_prefix"],

                    "s1_house": s1_blocks["house_number"],
                    "matched_house": source_blocks["house_number"],

                    "s1_tokens": ", ".join(s1_tokens),
                    "matched_tokens": ", ".join(source_tokens),

                    "s1_token_pairs": ", ".join(s1_pairs),
                    "matched_token_pairs": ", ".join(source_pairs),
                }
            )

    print("\n" + "=" * 80)
    print("MISSED PAIRS")
    print("=" * 80)

    print(f"Total missed: {len(missed):,}")

    if not missed:
        print("No missed pairs found.")
        return

    # Deterministic sample of misses
    missed_df = pd.DataFrame(missed)

    sample = missed_df.sample(
        n=min(MAX_EXAMPLES, len(missed_df)),
        random_state=RANDOM_STATE,
    )

    for i, row in enumerate(
        sample.to_dict("records"),
        start=1,
    ):
        print("\n" + "-" * 80)
        print(f"MISS {i}")

        print(f"S1 ID       : {row['s1_id']}")
        print(f"Matched ID  : {row['matched_id']}")
        print(f"Source      : {row['source']}")
        print(f"Country     : {row['country']}")

        print(f"\nS1 name     : {row['s1_name']}")
        print(f"Match name  : {row['matched_name']}")

        print(f"\nS1 address  : {row['s1_address']}")
        print(f"Match addr  : {row['matched_address']}")

        print("\nName prefixes:")
        print(f"  S1    : {row['s1_name_prefix']}")
        print(f"  Match : {row['matched_name_prefix']}")

        print("\nHouse numbers:")
        print(f"  S1    : {row['s1_house']}")
        print(f"  Match : {row['matched_house']}")

        print("\nName tokens:")
        print(f"  S1    : {row['s1_tokens']}")
        print(f"  Match : {row['matched_tokens']}")

        print("\nToken pairs:")
        print(f"  S1    : {row['s1_token_pairs']}")
        print(f"  Match : {row['matched_token_pairs']}")

    # ------------------------------------------------------------------
    # Save complete miss set
    # ------------------------------------------------------------------

    output_path = Path("output") / "blocking_misses_sample.csv"

    missed_df.to_csv(
        output_path,
        index=False,
    )

    print("\n" + "=" * 80)
    print(f"Full miss list saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()