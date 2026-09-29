from pathlib import Path
import random
import pandas as pd

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
from rapidfuzz.fuzz import (
    ratio,
    token_set_ratio,
)


SEED = 42
SAMPLE_SIZE = 5_000
CHUNK_SIZE = 100_000

NAME_PAIR_THRESHOLD = 500
ADDRESS_TOKEN_THRESHOLD = 100


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


def country_key(value):
    return str(value).strip().casefold()


def name_pair_keys(country, name):
    tokens = [
        x for x in token_list(name)
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
        x for x in token_list(address)
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


def main():

    print("=" * 72)
    print("ANALYSIS OF TRUE PAIRS MISSED BY CURRENT BLOCKING")
    print("=" * 72)

    # ------------------------------------------------------------
    # 1. Validation S1 sample
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # 2. Load S1
    # ------------------------------------------------------------

    s1 = pd.read_csv(
        TRAIN_SOURCE1,
        sep="\t",
        dtype=str,
    )

    s1 = s1[
        s1["entity_id"].isin(id_set)
    ].copy()

    s1_lookup = {}

    for row in s1.itertuples(index=False):
        s1_lookup[row.entity_id] = {
            "country": country_key(row.country),
            "name": normalize_name(
                row.business_name
            ),
            "address": normalize_address(
                row.business_address
            ),
        }

    del s1

    # ------------------------------------------------------------
    # 3. Ground truth
    # ------------------------------------------------------------

    gt = pd.read_csv(
        TRAIN_GROUND_TRUTH,
        sep="\t",
        dtype=str,
    )

    gt["matched_entity_ids"] = (
        gt["matched_entity_ids"]
        .fillna("")
    )

    true_pairs = []

    for row in gt.itertuples(index=False):

        s1_id = row.source1_entity_id

        if s1_id not in id_set:
            continue

        for entity_id in (
            x.strip()
            for x in row.matched_entity_ids.split(",")
            if x.strip()
        ):
            true_pairs.append(
                (s1_id, entity_id)
            )

    del gt

    print(
        f"True pairs: {len(true_pairs):,}"
    )

    # ------------------------------------------------------------
    # 4. Required blocking keys from S1
    # ------------------------------------------------------------

    required_exact = set()
    required_compact = set()
    required_pairs = set()
    required_prefix_house = set()
    required_address_tokens = set()

    for s1_id, row in s1_lookup.items():

        country = row["country"]
        name = row["name"]
        address = row["address"]

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

    # ------------------------------------------------------------
    # 5. Frequency pass
    # ------------------------------------------------------------

    name_pair_frequency = {}
    address_frequency = {}

    print("\nPass 1: calculating blocking frequencies...")

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

                country = country_key(
                    row.country
                )

                name = normalize_name(
                    row.business_name
                )

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
                            key, 0
                        ) + 1
                    )

                for key in (
                    address_token_keys(
                        country,
                        address,
                    )
                    & required_address_tokens
                ):
                    address_frequency[key] = (
                        address_frequency.get(
                            key, 0
                        ) + 1
                    )

        print(
            f"  {source_path.name}: done"
        )

    allowed_name_pairs = {
        k for k, v
        in name_pair_frequency.items()
        if v <= NAME_PAIR_THRESHOLD
    }

    allowed_address_tokens = {
        k for k, v
        in address_frequency.items()
        if v <= ADDRESS_TOKEN_THRESHOLD
    }

    # ------------------------------------------------------------
    # 6. Second pass: retrieve ONLY true source entities
    #
    # This lets us classify missed true pairs without storing
    # hundreds of thousands of candidate entities.
    # ------------------------------------------------------------

    needed_entities = {
        entity_id
        for _, entity_id in true_pairs
    }

    source_lookup = {}

    print("\nPass 2: retrieving true-match source records...")

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

            mask = chunk["entity_id"].isin(
                needed_entities
            )

            matched = chunk.loc[mask]

            for row in matched.itertuples(index=False):

                source_lookup[row.entity_id] = {
                    "country": country_key(row.country),
                    "name": normalize_name(
                        row.business_name
                    ),
                    "address": normalize_address(
                        row.business_address
                    ),
                }

        print(
            f"  {source_path.name}: done"
        )

    # ------------------------------------------------------------
    # 7. Classify missed pairs
    # ------------------------------------------------------------

    records = []

    for s1_id, entity_id in true_pairs:

        s1_row = s1_lookup[s1_id]
        s2_row = source_lookup.get(entity_id)

        if s2_row is None:
            continue

        country = s1_row["country"]

        # ------------------------------
        # Current blocker
        # ------------------------------

        exact_hit = (
            f"{country}|name|{s1_row['name']}"
            == f"{s2_row['country']}|name|{s2_row['name']}"
        )

        compact_hit = (
            compact_text(s1_row["name"])
            == compact_text(s2_row["name"])
            and country == s2_row["country"]
        )

        prefix_key = prefix_house_key(
            country,
            s1_row["name"],
            s1_row["address"],
        )

        source_prefix_key = prefix_house_key(
            s2_row["country"],
            s2_row["name"],
            s2_row["address"],
        )

        prefix_hit = (
            prefix_key is not None
            and prefix_key == source_prefix_key
        )

        pair_hit = bool(
            name_pair_keys(
                country,
                s1_row["name"],
            )
            & name_pair_keys(
                s2_row["country"],
                s2_row["name"],
            )
            & allowed_name_pairs
        )

        address_hit = bool(
            address_token_keys(
                country,
                s1_row["address"],
            )
            & address_token_keys(
                s2_row["country"],
                s2_row["address"],
            )
            & allowed_address_tokens
        )

        blocked = (
            exact_hit
            or compact_hit
            or prefix_hit
            or pair_hit
            or address_hit
        )

        if blocked:
            continue

        # --------------------------------------------------------
        # Similarity diagnostics
        # --------------------------------------------------------

        name_ratio = ratio(
            s1_row["name"],
            s2_row["name"],
        ) / 100.0

        name_token_ratio = token_set_ratio(
            s1_row["name"],
            s2_row["name"],
        ) / 100.0

        name_overlap = token_overlap(
            s1_row["name"],
            s2_row["name"],
        )

        name_containment = containment(
            s1_row["name"],
            s2_row["name"],
        )

        address_ratio = ratio(
            s1_row["address"],
            s2_row["address"],
        ) / 100.0

        address_token_ratio = token_set_ratio(
            s1_row["address"],
            s2_row["address"],
        ) / 100.0

        address_overlap = token_overlap(
            s1_row["address"],
            s2_row["address"],
        )

        address_containment = containment(
            s1_row["address"],
            s2_row["address"],
        )

        records.append({
            "s1_id": s1_id,
            "source_id": entity_id,

            "country": country,

            "s1_name": s1_row["name"],
            "source_name": s2_row["name"],

            "s1_address": s1_row["address"],
            "source_address": s2_row["address"],

            "name_ratio": name_ratio,
            "name_token_ratio": name_token_ratio,
            "name_overlap": name_overlap,
            "name_containment": name_containment,

            "address_ratio": address_ratio,
            "address_token_ratio": address_token_ratio,
            "address_overlap": address_overlap,
            "address_containment": address_containment,
        })

    result = pd.DataFrame(records)

    print("\n" + "=" * 72)
    print("MISSED TRUE PAIRS")
    print("=" * 72)

    print(
        f"Missed true pairs: "
        f"{len(result):,}"
    )

    if result.empty:
        print("No misses found.")
        return

    # ------------------------------------------------------------
    # 8. Pattern buckets
    # ------------------------------------------------------------

    result["strong_name"] = (
        result["name_ratio"] >= 0.75
    )

    result["very_strong_name"] = (
        result["name_ratio"] >= 0.90
    )

    result["strong_address"] = (
        result["address_ratio"] >= 0.75
    )

    result["very_strong_address"] = (
        result["address_ratio"] >= 0.90
    )

    result["strong_name_tokens"] = (
        result["name_token_ratio"] >= 0.80
    )

    result["strong_address_tokens"] = (
        result["address_token_ratio"] >= 0.80
    )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    n = len(result)

    print("\nName similarity:")
    print(
        f"  name_ratio >= .90: "
        f"{result['very_strong_name'].sum():,} "
        f"({result['very_strong_name'].mean():.2%})"
    )
    print(
        f"  name_ratio >= .75: "
        f"{result['strong_name'].sum():,} "
        f"({result['strong_name'].mean():.2%})"
    )

    print("\nAddress similarity:")
    print(
        f"  address_ratio >= .90: "
        f"{result['very_strong_address'].sum():,} "
        f"({result['very_strong_address'].mean():.2%})"
    )
    print(
        f"  address_ratio >= .75: "
        f"{result['strong_address'].sum():,} "
        f"({result['strong_address'].mean():.2%})"
    )

    print("\nCombined patterns:")

    patterns = {
        "strong_name + strong_address": (
            result["strong_name"]
            & result["strong_address"]
        ),

        "strong_name + weak_address": (
            result["strong_name"]
            & ~result["strong_address"]
        ),

        "weak_name + strong_address": (
            ~result["strong_name"]
            & result["strong_address"]
        ),

        "very_strong_name": (
            result["very_strong_name"]
        ),

        "very_strong_address": (
            result["very_strong_address"]
        ),

        "strong token-name": (
            result["strong_name_tokens"]
        ),

        "strong token-address": (
            result["strong_address_tokens"]
        ),
    }

    for label, mask in patterns.items():

        count = int(mask.sum())

        print(
            f"  {label:32s}: "
            f"{count:6,} "
            f"({count / n:.2%})"
        )

    # ------------------------------------------------------------
    # 9. Save misses for detailed inspection
    # ------------------------------------------------------------

    output_path = Path(
        "output/blocking_misses_current.csv"
    )

    result.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved detailed misses to:"
        f"\n  {output_path}"
    )

    # ------------------------------------------------------------
    # 10. Show representative difficult cases
    # ------------------------------------------------------------

    print("\n" + "=" * 72)
    print("REPRESENTATIVE MISSES")
    print("=" * 72)

    display_columns = [
        "s1_id",
        "source_id",
        "s1_name",
        "source_name",
        "name_ratio",
        "name_token_ratio",
        "s1_address",
        "source_address",
        "address_ratio",
        "address_token_ratio",
    ]

    sample = result.copy()

    # Prefer difficult cases with high address similarity
    # but low name similarity.
    sample["difficulty_score"] = (
        sample["address_ratio"]
        - sample["name_ratio"]
    )

    sample = sample.sort_values(
        "difficulty_score",
        ascending=False,
    )

    print(
        sample[
            display_columns
        ].head(30).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()