from pathlib import Path
import random

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
S1_SAMPLE_SIZE = 5_000
MAX_HARD_NEGATIVES_PER_S1 = 3

random.seed(SEED)


def tokens(text):
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
    return len(a & b) / min(len(a), len(b))


def features(s1, candidate):
    name1 = s1["name"]
    name2 = candidate["name"]

    addr1 = s1["address"]
    addr2 = candidate["address"]

    nt1 = tokens(name1)
    nt2 = tokens(name2)

    at1 = tokens(addr1)
    at2 = tokens(addr2)

    return {
        "country_same": int(
            s1["country"] == candidate["country"]
        ),
        "name_exact": int(
            bool(name1) and name1 == name2
        ),
        "name_compact_exact": int(
            bool(name1)
            and name1.replace(" ", "")
            == name2.replace(" ", "")
        ),
        "name_ratio": fuzz.ratio(name1, name2) / 100.0,
        "name_token_set_ratio": (
            fuzz.token_set_ratio(name1, name2) / 100.0
        ),
        "name_token_overlap": overlap(nt1, nt2),
        "name_containment": containment(nt1, nt2),
        "address_exact": int(
            bool(addr1) and addr1 == addr2
        ),
        "address_ratio": fuzz.ratio(addr1, addr2) / 100.0,
        "address_token_set_ratio": (
            fuzz.token_set_ratio(addr1, addr2) / 100.0
        ),
        "address_token_overlap": overlap(at1, at2),
        "address_containment": containment(at1, at2),
    }


def block_keys(row):
    country = row["country"]

    name = row["name"]
    address = row["address"]

    keys = []

    # Name prefix
    if len(name) >= 8:
        compact = name.replace(" ", "")
        if len(compact) >= 8:
            keys.append(
                f"{country}|np|{compact[:8]}"
            )

    # Individual name tokens
    name_tokens = [
        x for x in name.split()
        if len(x) >= 3
    ]

    for token in set(name_tokens):
        keys.append(
            f"{country}|nt|{token}"
        )

    # Name token pairs
    unique_tokens = sorted(set(name_tokens))

    for i in range(len(unique_tokens)):
        for j in range(i + 1, len(unique_tokens)):
            keys.append(
                f"{country}|npair|"
                f"{unique_tokens[i]}|{unique_tokens[j]}"
            )

    # Address tokens
    for token in set(address.split()):
        if len(token) >= 4:
            keys.append(
                f"{country}|at|{token}"
            )

    return keys


def prepare_source(path):
    print(f"  Loading {path.name}")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
    )

    # Normalize once.
    df["name"] = (
        df["business_name"]
        .fillna("")
        .map(normalize_name)
    )

    df["address"] = (
        df["business_address"]
        .fillna("")
        .map(normalize_address)
    )

    df["country"] = (
        df["country"]
        .fillna("")
        .str.casefold()
    )

    # Convert only the columns we need into Python records.
    records = []

    for row in df[
        [
            "entity_id",
            "name",
            "address",
            "country",
        ]
    ].itertuples(index=False, name=None):

        records.append(row)

    return records


def main():

    print("=" * 72)
    print("FAST HARD-NEGATIVE SAMPLE")
    print("=" * 72)

    # ------------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------------

    print("\nLoading ground truth...")

    gt = pd.read_csv(
        TRAIN_GROUND_TRUTH,
        sep="\t",
        dtype=str,
        usecols=[
            "source1_entity_id",
            "matched_entity_ids",
        ],
    )

    gt["matched_entity_ids"] = (
        gt["matched_entity_ids"]
        .fillna("")
    )

    gt_map = {}

    for row in gt.itertuples(index=False):

        matches = {
            x.strip()
            for x in row.matched_entity_ids.split(",")
            if x.strip()
        }

        gt_map[row.source1_entity_id] = matches

    print(f"Ground truth rows: {len(gt_map):,}")

    # ------------------------------------------------------------
    # S1 sample
    # ------------------------------------------------------------

    print("\nLoading S1...")

    s1_df = pd.read_csv(
        TRAIN_SOURCE1,
        sep="\t",
        dtype=str,
    )

    s1_df = s1_df.sample(
        n=min(S1_SAMPLE_SIZE, len(s1_df)),
        random_state=SEED,
    )

    s1_records = []

    for row in s1_df.itertuples(index=False):

        s1_records.append({
            "entity_id": row.entity_id,
            "name": normalize_name(
                row.business_name
            ),
            "address": normalize_address(
                row.business_address
            ),
            "country": str(
                row.country
            ).casefold(),
        })

    print(f"S1 sample: {len(s1_records):,}")

    del s1_df

    # ------------------------------------------------------------
    # Candidate sources
    # ------------------------------------------------------------

    print("\nLoading S2/S3...")

    s2 = prepare_source(TRAIN_SOURCE2)
    s3 = prepare_source(TRAIN_SOURCE3)

    candidates = s2 + s3

    del s2
    del s3

    print(
        f"Total candidates: {len(candidates):,}"
    )

    # ------------------------------------------------------------
    # Blocking index
    # ------------------------------------------------------------

    print("\nBuilding blocking index...")

    block_index = {}

    for idx, (
        entity_id,
        name,
        address,
        country,
    ) in enumerate(candidates):

        row = {
            "name": name,
            "address": address,
            "country": country,
        }

        for key in block_keys(row):

            bucket = block_index.get(key)

            if bucket is None:
                block_index[key] = [idx]
            else:
                bucket.append(idx)

        if (idx + 1) % 500_000 == 0:
            print(
                f"  Indexed {idx + 1:,}"
            )

    print(
        f"Blocking keys: {len(block_index):,}"
    )

    # ------------------------------------------------------------
    # Hard negatives
    # ------------------------------------------------------------

    print("\nGenerating hard negatives...")

    output_rows = []

    for i, s1 in enumerate(s1_records):

        s1_id = s1["entity_id"]

        true_ids = gt_map.get(
            s1_id,
            set(),
        )

        candidate_indices = set()

        for key in block_keys(s1):

            bucket = block_index.get(key)

            if bucket:
                candidate_indices.update(bucket)

        # Remove true matches.
        hard_indices = [
            idx
            for idx in candidate_indices
            if candidates[idx][0] not in true_ids
        ]

        # We don't need every collision.
        if len(hard_indices) > MAX_HARD_NEGATIVES_PER_S1:
            hard_indices = random.sample(
                hard_indices,
                MAX_HARD_NEGATIVES_PER_S1,
            )

        for idx in hard_indices:

            (
                candidate_id,
                candidate_name,
                candidate_address,
                candidate_country,
            ) = candidates[idx]

            candidate = {
                "entity_id": candidate_id,
                "name": candidate_name,
                "address": candidate_address,
                "country": candidate_country,
            }

            output_rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": candidate_id,
                "candidate_source": (
                    "S2"
                    if candidate_id.startswith("S2-")
                    else "S3"
                ),
                "label": 0,
                **features(s1, candidate),
            })

        if (i + 1) % 250 == 0:
            print(
                f"  Processed "
                f"{i + 1:,} / {len(s1_records):,}"
                f" | hard negatives: "
                f"{len(output_rows):,}"
            )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    output = pd.DataFrame(output_rows)

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    output_path = (
        output_dir
        / "hard_negative_feature_sample.csv"
    )

    output.to_csv(
        output_path,
        index=False,
    )

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)

    print(
        f"Hard-negative rows: "
        f"{len(output):,}"
    )

    print(
        f"Saved: {output_path}"
    )

    if len(output):

        feature_columns = [
            "name_ratio",
            "name_token_set_ratio",
            "name_token_overlap",
            "name_containment",
            "address_ratio",
            "address_token_set_ratio",
            "address_token_overlap",
            "address_containment",
        ]

        print("\nHARD-NEGATIVE FEATURE SUMMARY")
        print("=" * 72)

        print(
            output[feature_columns]
            .describe(
                percentiles=[
                    0.05,
                    0.50,
                    0.95,
                ]
            )
            .T[
                [
                    "mean",
                    "50%",
                    "5%",
                    "95%",
                    "max",
                ]
            ]
        )


if __name__ == "__main__":
    main()