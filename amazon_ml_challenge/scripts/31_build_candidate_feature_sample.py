from pathlib import Path
import random

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


SEED = 42
SAMPLE_SIZE = 5_000
CHUNK_SIZE = 100_000

NAME_PAIR_THRESHOLD = 500
ADDRESS_TOKEN_THRESHOLD = 100

# Keep the feature dataset manageable.
MAX_NEGATIVES_PER_S1 = 20
MAX_POSITIVES_PER_S1 = 100


COMMON_BUSINESS_TOKENS = {
    "llc", "inc", "incorporated", "corp",
    "corporation", "company", "co",
    "ltd", "limited", "private",
    "pvt", "plc", "llp",
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


def prefix_house_key(country, name, address):
    compact = compact_text(name)

    if len(compact) < 8:
        return None

    for token in normalize_address(address).split():
        if any(c.isdigit() for c in token):
            return (
                f"{country}|prefix_house|"
                f"{compact[:8]}|{token}"
            )

    return None


def token_overlap(a, b):
    a = set(token_list(a))
    b = set(token_list(b))

    if not a or not b:
        return 0.0

    return len(a & b) / min(len(a), len(b))


def containment(a, b):
    a = set(token_list(a))
    b = set(token_list(b))

    if not a or not b:
        return 0.0

    return len(a & b) / min(len(a), len(b))


def make_features(s1, source):
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
            ratio(name1, name2) / 100.0
        ),

        "name_token_set_ratio": (
            token_set_ratio(
                name1,
                name2,
            ) / 100.0
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
            ) / 100.0
        ),

        "address_token_set_ratio": (
            token_set_ratio(
                address1,
                address2,
            ) / 100.0
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


def main():

    print("=" * 72)
    print("BUILD CANDIDATE-PAIR FEATURE SAMPLE")
    print("=" * 72)

    # ============================================================
    # 1. Validation sample
    # ============================================================

    ids = [
        x.strip()
        for x in Path(
            "output/validation_s1_ids.txt"
        ).read_text().splitlines()
        if x.strip()
    ]

    rng = random.Random(SEED)
    rng.shuffle(ids)

    ids = ids[:SAMPLE_SIZE]
    id_set = set(ids)

    print(
        f"\nValidation S1 sample: {len(ids):,}"
    )

    # ============================================================
    # 2. Load S1
    # ============================================================

    s1 = pd.read_csv(
        TRAIN_SOURCE1,
        sep="\t",
        dtype=str,
    )

    s1 = s1[
        s1["entity_id"].isin(id_set)
    ]

    s1_lookup = {}

    required_exact = set()
    required_compact = set()
    required_pairs = set()
    required_prefix_house = set()
    required_address_tokens = set()

    # For recovery block:
    # rare address token -> S1 IDs
    address_key_to_s1 = {}

    for row in s1.itertuples(index=False):

        entity_id = row.entity_id
        country = country_key(row.country)
        name = normalize_name(row.business_name)
        address = normalize_address(
            row.business_address
        )

        s1_lookup[entity_id] = {
            "country": country,
            "name": name,
            "address": address,
        }

        required_exact.add(
            f"{country}|name|{name}"
        )

        compact = compact_text(name)

        if compact:
            required_compact.add(
                f"{country}|compact_name|{compact}"
            )

        required_pairs.update(
            name_pair_keys(
                country,
                name,
            )
        )

        prefix = prefix_house_key(
            country,
            name,
            address,
        )

        if prefix:
            required_prefix_house.add(prefix)

        required_address_tokens.update(
            address_token_keys(
                country,
                address,
            )
        )

    del s1

    print(
        f"S1 entities loaded: "
        f"{len(s1_lookup):,}"
    )

    # ============================================================
    # 3. Ground truth
    # ============================================================

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

    for row in gt.itertuples(index=False):

        s1_id = row.source1_entity_id

        if s1_id not in id_set:
            continue

        truth[s1_id] = {
            x.strip()
            for x in row.matched_entity_ids.split(",")
            if x.strip()
        }

    del gt

    total_true_pairs = sum(
        len(x)
        for x in truth.values()
    )

    print(
        f"True pairs: {total_true_pairs:,}"
    )

    # ============================================================
    # 4. First source pass:
    #    frequencies for name pairs + address tokens
    # ============================================================

    name_pair_frequency = {}
    address_token_frequency = {}

    print("\nPASS 1: blocking frequencies")

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

            for row in chunk.itertuples(index=False):

                country = country_key(row.country)
                name = normalize_name(row.business_name)
                address = normalize_address(
                    row.business_address
                )

                for key in (
                    name_pair_keys(
                        country,
                        name,
                    )
                    & required_pairs
                ):
                    name_pair_frequency[key] = (
                        name_pair_frequency.get(
                            key,
                            0,
                        ) + 1
                    )

                for key in (
                    address_token_keys(
                        country,
                        address,
                    )
                    & required_address_tokens
                ):
                    address_token_frequency[key] = (
                        address_token_frequency.get(
                            key,
                            0,
                        ) + 1
                    )

        print(
            f"  {source_path.name}: done"
        )

    allowed_name_pairs = {
        key
        for key, count in name_pair_frequency.items()
        if count <= NAME_PAIR_THRESHOLD
    }

    allowed_address_tokens = {
        key
        for key, count
        in address_token_frequency.items()
        if count <= ADDRESS_TOKEN_THRESHOLD
    }

    print(
        f"Allowed name-pair keys: "
        f"{len(allowed_name_pairs):,}"
    )

    print(
        f"Allowed address-token keys: "
        f"{len(allowed_address_tokens):,}"
    )

    # ============================================================
    # 5. Build address recovery map
    # ============================================================

    for s1_id, row in s1_lookup.items():

        keys = (
            address_token_keys(
                row["country"],
                row["address"],
            )
            & allowed_address_tokens
        )

        for key in keys:

            address_key_to_s1.setdefault(
                key,
                set(),
            ).add(s1_id)

    # ============================================================
    # 6. Candidate collection
    #
    # candidate_map[source_entity_id] = set(S1 IDs)
    #
    # This stores only IDs, not entire rows.
    # ============================================================

    candidate_map = {}

    def add_candidate(source_id, s1_id):
        candidate_map.setdefault(
            source_id,
            set(),
        ).add(s1_id)

    print("\nPASS 2: generating candidates")

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

            for row in chunk.itertuples(index=False):

                scanned += 1

                source_id = row.entity_id

                country = country_key(row.country)
                name = normalize_name(row.business_name)
                address = normalize_address(
                    row.business_address
                )

                # ------------------------------------------------
                # Exact name
                # ------------------------------------------------

                exact_key = (
                    f"{country}|name|{name}"
                )

                # Instead of maintaining another inverse map,
                # exact/compact/prefix/name-pair candidates are
                # reconstructed by scanning S1 keys below.
                #
                # To keep this memory-light, we create the maps
                # lazily here.
                #
                # These maps are built once outside the loop below.

                # ------------------------------------------------
                # Address recovery candidates
                # ------------------------------------------------

                addr_keys = (
                    address_token_keys(
                        country,
                        address,
                    )
                    & allowed_address_tokens
                )

                for key in addr_keys:

                    for s1_id in address_key_to_s1.get(
                        key,
                        (),
                    ):
                        add_candidate(
                            source_id,
                            s1_id,
                        )

                # ------------------------------------------------
                # Store normalized source record temporarily.
                #
                # We only need records for entities that become
                # candidates.
                # ------------------------------------------------

            print(
                f"  {source_path.name}: "
                f"{scanned:,} rows scanned"
            )

    # ============================================================
    # IMPORTANT:
    #
    # The address recovery above captures only the address-driven
    # candidates. We now need the four name-driven blocks.
    #
    # Build inverse S1 maps and scan sources again.
    # ============================================================

    print("\nPASS 3: name-driven candidates")

    exact_to_s1 = {}
    compact_to_s1 = {}
    prefix_to_s1 = {}
    pair_to_s1 = {}

    for s1_id, row in s1_lookup.items():

        country = row["country"]
        name = row["name"]
        address = row["address"]

        exact_to_s1.setdefault(
            f"{country}|name|{name}",
            set(),
        ).add(s1_id)

        compact = compact_text(name)

        if compact:
            compact_to_s1.setdefault(
                f"{country}|compact_name|{compact}",
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

        for key in (
            name_pair_keys(
                country,
                name,
            )
            & allowed_name_pairs
        ):
            pair_to_s1.setdefault(
                key,
                set(),
            ).add(s1_id)

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

            for row in chunk.itertuples(index=False):

                scanned += 1

                source_id = row.entity_id
                country = country_key(row.country)
                name = normalize_name(row.business_name)
                address = normalize_address(
                    row.business_address
                )

                exact_key = (
                    f"{country}|name|{name}"
                )

                for s1_id in exact_to_s1.get(
                    exact_key,
                    (),
                ):
                    add_candidate(
                        source_id,
                        s1_id,
                    )

                compact = compact_text(name)

                if compact:

                    compact_key = (
                        f"{country}|compact_name|{compact}"
                    )

                    for s1_id in compact_to_s1.get(
                        compact_key,
                        (),
                    ):
                        add_candidate(
                            source_id,
                            s1_id,
                        )

                prefix = prefix_house_key(
                    country,
                    name,
                    address,
                )

                if prefix:

                    for s1_id in prefix_to_s1.get(
                        prefix,
                        (),
                    ):
                        add_candidate(
                            source_id,
                            s1_id,
                        )

                for key in (
                    name_pair_keys(
                        country,
                        name,
                    )
                    & allowed_name_pairs
                ):

                    for s1_id in pair_to_s1.get(
                        key,
                        (),
                    ):
                        add_candidate(
                            source_id,
                            s1_id,
                        )

            print(
                f"  {source_path.name}: "
                f"{scanned:,} rows scanned"
            )

    # ============================================================
    # 7. Feature extraction
    #
    # Keep positives + sampled negatives per S1.
    # ============================================================

    candidate_pairs = []

    # We need source records only for candidate source IDs.
    candidate_source_ids = set(
        candidate_map.keys()
    )

    print(
        f"\nUnique candidate source entities: "
        f"{len(candidate_source_ids):,}"
    )

    # Reverse index source -> candidate S1s.
    #
    # This is already candidate_map.
    #
    # Retrieve source records in a final scan.
    # ============================================================

    rng = random.Random(SEED)

    per_s1_negatives = {
        s1_id: []
        for s1_id in s1_lookup
    }

    per_s1_positives = {
        s1_id: []
        for s1_id in s1_lookup
    }

    total_source_scanned = 0

    print("\nPASS 4: feature extraction")

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

            for row in chunk.itertuples(index=False):

                source_id = row.entity_id

                s1_ids = candidate_map.get(
                    source_id
                )

                if not s1_ids:
                    continue

                source = {
                    "country": country_key(row.country),
                    "name": normalize_name(
                        row.business_name
                    ),
                    "address": normalize_address(
                        row.business_address
                    ),
                }

                for s1_id in s1_ids:

                    s1_row = s1_lookup[s1_id]

                    label = int(
                        source_id
                        in truth.get(
                            s1_id,
                            set(),
                        )
                    )

                    feature_row = {
                        "s1_id": s1_id,
                        "source_id": source_id,
                        "label": label,
                    }

                    feature_row.update(
                        make_features(
                            s1_row,
                            source,
                        )
                    )

                    if label:

                        if len(
                            per_s1_positives[s1_id]
                        ) < MAX_POSITIVES_PER_S1:

                            per_s1_positives[
                                s1_id
                            ].append(
                                feature_row
                            )

                    else:

                        # Reservoir-style random negative sample.
                        bucket = per_s1_negatives[
                            s1_id
                        ]

                        if len(bucket) < MAX_NEGATIVES_PER_S1:

                            bucket.append(
                                feature_row
                            )

                        else:

                            j = rng.randint(
                                0,
                                len(bucket),
                            )

                            if j < MAX_NEGATIVES_PER_S1:
                                bucket[j] = feature_row

            total_source_scanned += len(chunk)

        print(
            f"  {source_path.name}: done"
        )

    # ============================================================
    # 8. Assemble dataset
    # ============================================================

    for s1_id in s1_lookup:

        candidate_pairs.extend(
            per_s1_positives[s1_id]
        )

        candidate_pairs.extend(
            per_s1_negatives[s1_id]
        )

    result = pd.DataFrame(
        candidate_pairs
    )

    output_path = Path(
        "output/candidate_feature_sample.csv"
    )

    result.to_csv(
        output_path,
        index=False,
    )

    print("\n" + "=" * 72)
    print("FINAL DATASET")
    print("=" * 72)

    print(
        f"Rows: "
        f"{len(result):,}"
    )

    print(
        f"Positive: "
        f"{int(result['label'].sum()):,}"
    )

    print(
        f"Negative: "
        f"{int((result['label'] == 0).sum()):,}"
    )

    print(
        f"Positive rate: "
        f"{result['label'].mean():.4%}"
    )

    print(
        f"\nSaved:"
        f"\n  {output_path}"
    )

    print("\nFeature summary:")

    print(
        result[
            MODEL_FEATURES
        ].describe().T[
            ["mean", "std", "min", "50%", "max"]
        ].to_string()
    )


if __name__ == "__main__":
    main()