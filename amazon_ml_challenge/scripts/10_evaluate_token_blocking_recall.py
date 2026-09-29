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
    address_tokens_key,
)


RANDOM_STATE = 42
SAMPLE_SIZE = 20_000


def build_key_set(country, name, address):
    keys = set()

    blocks = generate_blocking_keys(
        country,
        name,
        address,
    )

    for value in blocks.values():
        if value:
            keys.add(value)

    for key in name_tokens_key(country, name):
        keys.add(key)

    for key in name_token_pair_keys(country, name):
        keys.add(key)

    for key in address_tokens_key(country, address):
        keys.add(key)

    return keys

def main():
    print("=" * 80)
    print("TOKEN BLOCKING TRUE-PAIR RECALL EVALUATION")
    print("=" * 80)

    print("\nLoading ground truth...")

    gt = pd.read_csv(
        TRAIN_GROUND_TRUTH,
        sep="\t",
        dtype=str,
    )

    gt = gt.sample(
        n=min(SAMPLE_SIZE, len(gt)),
        random_state=RANDOM_STATE,
    )

    print(f"Sampled S1 entities: {len(gt):,}")

    # ------------------------------------------------------------------
    # Extract true S2/S3 IDs from the sampled ground truth
    # ------------------------------------------------------------------

    true_pairs = []

    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        matched = str(row["matched_entity_ids"])

        if not matched or matched.lower() == "nan":
            continue

        for entity_id in matched.split(","):
            entity_id = entity_id.strip()

            if entity_id:
                true_pairs.append(
                    (s1_id, entity_id)
                )

    truth = pd.DataFrame(
        true_pairs,
        columns=["source1_entity_id", "matched_entity_id"],
    )

    print(f"True matched pairs: {len(truth):,}")

    # ------------------------------------------------------------------
    # Load sampled S1 records
    # ------------------------------------------------------------------

    s1_ids = set(gt["source1_entity_id"])

    print("\nLoading sampled S1 records...")

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

    s1 = pd.concat(s1_parts, ignore_index=True)

    print(f"Loaded S1 records: {len(s1):,}")

    # ------------------------------------------------------------------
    # Load only true matched S2/S3 records
    # ------------------------------------------------------------------

    true_ids = set(truth["matched_entity_id"])

    print(f"\nUnique true S2/S3 IDs: {len(true_ids):,}")

    source_files = [
        ("S2", TRAIN_SOURCE2),
        ("S3", TRAIN_SOURCE3),
    ]

    true_sources = []

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
                parts.append(part)

        if parts:
            source = pd.concat(parts, ignore_index=True)
            source["source"] = source_name
            true_sources.append(source)

            print(
                f"  Found {len(source):,} true {source_name} records"
            )

    candidates = pd.concat(
        true_sources,
        ignore_index=True,
    )

    print(
        f"\nLoaded true matched source records: "
        f"{len(candidates):,}"
    )

    # ------------------------------------------------------------------
    # Build lookup by entity ID
    # ------------------------------------------------------------------

    candidate_by_id = {
        row["entity_id"]: row
        for _, row in candidates.iterrows()
    }

    s1_by_id = {
        row["entity_id"]: row
        for _, row in s1.iterrows()
    }

    # ------------------------------------------------------------------
    # Evaluate each true pair
    # ------------------------------------------------------------------

    recovered_all = 0
    recovered_token = 0
    recovered_pair = 0
    recovered_prefix_house = 0
    recovered_address_token = 0
    recovered_all_with_address = 0

    for _, pair in truth.iterrows():

        s1_id = pair["source1_entity_id"]
        matched_id = pair["matched_entity_id"]

        s1_row = s1_by_id.get(s1_id)
        matched_row = candidate_by_id.get(matched_id)

        if s1_row is None or matched_row is None:
            continue

        s1_keys = build_key_set(
            s1_row["country"],
            s1_row["business_name"],
            s1_row["business_address"],
        )

        matched_keys = build_key_set(
            matched_row["country"],
            matched_row["business_name"],
            matched_row["business_address"],
        )
        if s1_keys & matched_keys:
            recovered_all += 1
        # All current blocks
        s1_address_keys = set(
            address_tokens_key(
                s1_row["country"],
                s1_row["business_address"],
            )
        )

        matched_address_keys = set(
            address_tokens_key(
                matched_row["country"],
                matched_row["business_address"],
            )
        )

        if s1_address_keys & matched_address_keys:
            recovered_address_token += 1
        # Token-only
        s1_token_keys = set(
            name_tokens_key(
                s1_row["country"],
                s1_row["business_name"],
            )
        )

        s1_pair_keys = set(
            name_token_pair_keys(
                s1_row["country"],
                s1_row["business_name"],
            )
        )

        matched_token_keys = set(
            name_tokens_key(
                matched_row["country"],
                matched_row["business_name"],
            )
        )

        matched_pair_keys = set(
            name_token_pair_keys(
                matched_row["country"],
                matched_row["business_name"],
            )
        )

        if (
            s1_token_keys & matched_token_keys
            or s1_pair_keys & matched_pair_keys
        ):
            recovered_token += 1

        # Token-pair only
        if s1_pair_keys & matched_pair_keys:
            recovered_pair += 1

        # Existing strong combination
        s1_prefix_house = {
            generate_blocking_keys(
                s1_row["country"],
                s1_row["business_name"],
                s1_row["business_address"],
            )["name_prefix"],
            generate_blocking_keys(
                s1_row["country"],
                s1_row["business_name"],
                s1_row["business_address"],
            )["house_number"],
        }

        matched_prefix_house = {
            generate_blocking_keys(
                matched_row["country"],
                matched_row["business_name"],
                matched_row["business_address"],
            )["name_prefix"],
            generate_blocking_keys(
                matched_row["country"],
                matched_row["business_name"],
                matched_row["business_address"],
            )["house_number"],
        }

        s1_prefix_house.discard("")
        matched_prefix_house.discard("")

        if s1_prefix_house & matched_prefix_house:
            recovered_prefix_house += 1

    total = len(truth)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    results = [
        (
            "Token + token-pair",
            recovered_token,
        ),
        (
            "Token-pair only",
            recovered_pair,
        ),
        (
            "Name-prefix + house-number",
            recovered_prefix_house,
        ),
        (
            "ALL blocks + address tokens",
            recovered_all,
        ),(
            "Address-token only",
            recovered_address_token,
        ),
    ]

    for name, recovered in results:
        recall = recovered / total if total else 0

        print(
            f"{name:35s} "
            f"{recovered:10,} / {total:,} "
            f"= {recall:.4%}"
        )


if __name__ == "__main__":
    main()