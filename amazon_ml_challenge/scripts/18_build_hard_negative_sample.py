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
from src.normalization import (
    normalize_name,
    normalize_address,
    token_list,
)


SEED = 42
S1_SAMPLE_SIZE = 5_000
MAX_HARD_NEGATIVES_PER_S1 = 5

random.seed(SEED)


def name_token_set(name):
    return set(token_list(name))


def overlap(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def containment(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def make_features(s1, candidate):
    s1_name = normalize_name(s1["business_name"])
    s2_name = normalize_name(candidate["business_name"])

    s1_addr = normalize_address(s1["business_address"])
    s2_addr = normalize_address(candidate["business_address"])

    name_tokens_1 = name_token_set(s1_name)
    name_tokens_2 = name_token_set(s2_name)

    addr_tokens_1 = name_token_set(s1_addr)
    addr_tokens_2 = name_token_set(s2_addr)

    return {
        "country_same": int(
            str(s1["country"]).casefold()
            == str(candidate["country"]).casefold()
        ),
        "name_exact": int(s1_name == s2_name and bool(s1_name)),
        "name_compact_exact": int(
            s1_name.replace(" ", "")
            == s2_name.replace(" ", "")
            and bool(s1_name)
        ),
        "name_ratio": fuzz.ratio(s1_name, s2_name) / 100.0,
        "name_token_set_ratio": (
            fuzz.token_set_ratio(s1_name, s2_name) / 100.0
        ),
        "name_token_overlap": overlap(name_tokens_1, name_tokens_2),
        "name_containment": containment(name_tokens_1, name_tokens_2),
        "address_exact": int(
            s1_addr == s2_addr and bool(s1_addr)
        ),
        "address_ratio": fuzz.ratio(s1_addr, s2_addr) / 100.0,
        "address_token_set_ratio": (
            fuzz.token_set_ratio(s1_addr, s2_addr) / 100.0
        ),
        "address_token_overlap": overlap(
            addr_tokens_1,
            addr_tokens_2,
        ),
        "address_containment": containment(
            addr_tokens_1,
            addr_tokens_2,
        ),
    }


def build_block_keys(row):
    country = str(row["country"]).casefold()

    name = normalize_name(row["business_name"])
    address = normalize_address(row["business_address"])

    keys = set()

    if name:
        compact = name.replace(" ", "")

        if len(compact) >= 8:
            keys.add(
                f"{country}|name_prefix|{compact[:8]}"
            )

        tokens = [
            t for t in name.split()
            if len(t) >= 3
        ]

        for token in tokens:
            keys.add(
                f"{country}|name_token|{token}"
            )

        unique_tokens = sorted(set(tokens))

        for i in range(len(unique_tokens)):
            for j in range(i + 1, len(unique_tokens)):
                keys.add(
                    f"{country}|name_pair|"
                    f"{unique_tokens[i]}|{unique_tokens[j]}"
                )

    if address:
        for token in set(address.split()):
            if len(token) >= 4:
                keys.add(
                    f"{country}|address_token|{token}"
                )

    return keys


def main():

    print("=" * 72)
    print("BUILDING HARD-NEGATIVE SAMPLE")
    print("=" * 72)

    print("\nLoading ground truth...")

    gt = pd.read_csv(
        TRAIN_GROUND_TRUTH,
        sep="\t",
        dtype=str,
    )

    gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")

    gt_map = dict(
        zip(
            gt["source1_entity_id"],
            gt["matched_entity_ids"].apply(
                lambda x: set(
                    item.strip()
                    for item in x.split(",")
                    if item.strip()
                )
            ),
        )
    )

    print(f"Ground truth rows: {len(gt):,}")

    print("\nSampling S1...")

    s1 = pd.read_csv(
        TRAIN_SOURCE1,
        sep="\t",
        dtype=str,
    )

    s1 = s1.sample(
        n=min(S1_SAMPLE_SIZE, len(s1)),
        random_state=SEED,
    ).reset_index(drop=True)

    print(f"S1 sample: {len(s1):,}")

    print("\nLoading S2/S3...")

    source_frames = []

    for path in [TRAIN_SOURCE2, TRAIN_SOURCE3]:
        print(f"  Loading {path.name}")

        df = pd.read_csv(
            path,
            sep="\t",
            dtype=str,
        )

        df["__source"] = path.stem

        source_frames.append(df)

    candidates = pd.concat(
        source_frames,
        ignore_index=True,
    )

    print(f"Total candidates: {len(candidates):,}")

    print("\nBuilding blocking index...")

    block_index = {}

    for idx, row in candidates.iterrows():

        keys = build_block_keys(row)

        for key in keys:
            block_index.setdefault(key, []).append(idx)

        if idx > 0 and idx % 500_000 == 0:
            print(f"  Indexed {idx:,} candidates")

    print(
        f"Blocking keys: {len(block_index):,}"
    )

    print("\nGenerating hard negatives...")

    output_rows = []

    for i, s1_row in s1.iterrows():

        s1_id = s1_row["entity_id"]

        true_ids = gt_map.get(
            s1_id,
            set(),
        )

        block_keys = build_block_keys(s1_row)

        candidate_indices = set()

        for key in block_keys:
            for idx in block_index.get(key, []):
                candidate_indices.add(idx)

        hard_negative_indices = []

        for idx in candidate_indices:

            candidate_id = candidates.iloc[idx]["entity_id"]

            if candidate_id in true_ids:
                continue

            hard_negative_indices.append(idx)

        random.shuffle(hard_negative_indices)

        hard_negative_indices = hard_negative_indices[
            :MAX_HARD_NEGATIVES_PER_S1
        ]

        for idx in hard_negative_indices:

            candidate = candidates.iloc[idx]

            features = make_features(
                s1_row,
                candidate,
            )

            row = {
                "source1_entity_id": s1_id,
                "candidate_entity_id": candidate["entity_id"],
                "candidate_source": candidate["__source"],
                "label": 0,
                **features,
            }

            output_rows.append(row)

        if (i + 1) % 500 == 0:
            print(
                f"  Processed {i + 1:,} / {len(s1):,}"
            )

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

    print(f"Hard-negative rows: {len(output):,}")
    print(f"Saved: {output_path}")

    if len(output):
        print("\nHard-negative feature summary:")

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

        print(
            output[feature_columns]
            .describe(
                percentiles=[0.05, 0.50, 0.95]
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