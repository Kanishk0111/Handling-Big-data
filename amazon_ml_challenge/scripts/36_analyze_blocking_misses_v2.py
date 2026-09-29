from pathlib import Path
import random
from collections import defaultdict

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


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SEED = 42
VALIDATION_SAMPLE_SIZE = 50_000
CHUNK_SIZE = 100_000

NAME_PAIR_THRESHOLD = 500
ADDRESS_TOKEN_THRESHOLD = 100

ADDRESS_RECOVERY_SET_RATIO = 0.90
ADDRESS_RECOVERY_OVERLAP = 0.80

OUTPUT_DIR = Path("output")

OUTPUT_FILE = (
    OUTPUT_DIR / "blocking_miss_analysis_50k.csv"
)


COMMON_BUSINESS_TOKENS = {
    "llc",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "company",
    "co",
    "ltd",
    "limited",
    "private",
    "pvt",
    "plc",
    "llp",
}


COMMON_ADDRESS_TOKENS = {
    "road",
    "rd",
    "street",
    "st",
    "avenue",
    "ave",
    "lane",
    "ln",
    "floor",
    "fl",
    "building",
    "bldg",
    "block",
    "plot",
    "house",
    "no",
    "number",
    "unit",
    "flat",
    "suite",
    "sector",
    "area",
    "phase",
    "village",
    "vill",
    "town",
}


# ---------------------------------------------------------------------
# Blocking helpers
# ---------------------------------------------------------------------

def country_key(value):
    if value is None:
        return ""

    return str(value).strip().casefold()


def name_pair_keys(country, name):
    tokens = [
        x
        for x in token_list(name)
        if len(x) >= 3
        and x not in COMMON_BUSINESS_TOKENS
    ]

    tokens = sorted(set(tokens))

    if not country or len(tokens) < 2:
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

    if not country:
        return set()

    return {
        f"{country}|address_token|{x}"
        for x in set(tokens)
    }


def prefix_house_key(country, name, address):
    compact_name = compact_text(name)

    if not country or len(compact_name) < 8:
        return None

    address = normalize_address(address)

    for token in address.split():
        if any(char.isdigit() for char in token):
            return (
                f"{country}|prefix_house|"
                f"{compact_name[:8]}|{token}"
            )

    return None


def token_overlap(a, b):
    a = set(token_list(a))
    b = set(token_list(b))

    if not a or not b:
        return 0.0

    return len(a & b) / min(len(a), len(b))


# ---------------------------------------------------------------------
# Load validation IDs
# ---------------------------------------------------------------------

def load_validation_ids():

    ids = [
        x.strip()
        for x in Path(
            "output/validation_s1_ids.txt"
        ).read_text().splitlines()
        if x.strip()
    ]

    rng = random.Random(SEED)
    rng.shuffle(ids)

    ids = ids[:VALIDATION_SAMPLE_SIZE]

    return ids


# ---------------------------------------------------------------------
# Load S1
# ---------------------------------------------------------------------

def load_s1(validation_ids):

    validation_set = set(validation_ids)

    print("Loading S1...")

    s1 = pd.read_csv(
        TRAIN_SOURCE1,
        sep="\t",
        dtype=str,
    )

    s1 = s1[
        s1["entity_id"].isin(validation_set)
    ]

    records = {}

    for row in s1.itertuples(index=False):

        entity_id = row.entity_id

        country = country_key(row.country)

        name = normalize_name(
            row.business_name
        )

        address = normalize_address(
            row.business_address
        )

        records[entity_id] = {
            "country": country,
            "name": name,
            "address": address,
            "name_pairs": name_pair_keys(
                country,
                name,
            ),
            "address_tokens": address_token_keys(
                country,
                address,
            ),
            "prefix_house": prefix_house_key(
                country,
                name,
                address,
            ),
        }

    print(
        f"S1 records loaded: {len(records):,}"
    )

    return records


# ---------------------------------------------------------------------
# Load ground truth
# ---------------------------------------------------------------------

def load_ground_truth(validation_ids):

    validation_set = set(validation_ids)

    print("Loading ground truth...")

    gt = pd.read_csv(
        TRAIN_GROUND_TRUTH,
        sep="\t",
        dtype=str,
    )

    gt.columns = [
        c.strip()
        for c in gt.columns
    ]

    truth = {}
    source_to_s1 = defaultdict(set)

    # These are the columns used by the existing validation scripts.
    s1_col = "source1_entity_id"
    matches_col = "matched_entity_ids"

    if s1_col not in gt.columns:
        raise ValueError(
            f"Expected {s1_col!r}; "
            f"columns are {list(gt.columns)}"
        )

    if matches_col not in gt.columns:
        raise ValueError(
            f"Expected {matches_col!r}; "
            f"columns are {list(gt.columns)}"
        )

    total_pairs = 0

    for row in gt.itertuples(index=False):

        s1_id = getattr(
            row,
            s1_col,
        )

        if s1_id not in validation_set:
            continue

        raw = getattr(
            row,
            matches_col,
        )

        if pd.isna(raw):
            matches = set()
        else:
            matches = {
                x.strip()
                for x in str(raw).split(",")
                if x.strip()
            }

        truth[s1_id] = matches

        total_pairs += len(matches)

        for source_id in matches:
            source_to_s1[source_id].add(
                s1_id
            )

    print(
        f"Validation S1s in truth: "
        f"{len(truth):,}"
    )

    print(
        f"True pairs: {total_pairs:,}"
    )

    return truth, source_to_s1


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 80)
    print("BLOCKING MISS ANALYSIS — 50K VALIDATION")
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Validation sample
    # -------------------------------------------------------------

    validation_ids = load_validation_ids()

    print(
        f"\nValidation S1 sample: "
        f"{len(validation_ids):,}"
    )

    # -------------------------------------------------------------
    # S1
    # -------------------------------------------------------------

    s1 = load_s1(
        validation_ids
    )

    # -------------------------------------------------------------
    # Ground truth
    # -------------------------------------------------------------

    truth, source_to_s1 = load_ground_truth(
        validation_ids
    )

    total_true_pairs = sum(
        len(x)
        for x in truth.values()
    )

    # -------------------------------------------------------------
    # Required blocking keys
    #
    # These are only keys belonging to the validation S1 sample.
    # -------------------------------------------------------------

    required_name_pairs = set()
    required_address_tokens = set()

    for record in s1.values():

        required_name_pairs.update(
            record["name_pairs"]
        )

        required_address_tokens.update(
            record["address_tokens"]
        )

    print(
        f"\nRequired name-pair keys: "
        f"{len(required_name_pairs):,}"
    )

    print(
        f"Required address-token keys: "
        f"{len(required_address_tokens):,}"
    )

    # -------------------------------------------------------------
    # We need source frequencies because the frozen blocker uses
    # rarity <=500 for name pairs and <=100 for address tokens.
    #
    # At the same time, collect only source rows that participate
    # in known true pairs.
    # -------------------------------------------------------------

    name_pair_frequency = defaultdict(int)
    address_token_frequency = defaultdict(int)

    true_source_rows = {}

    total_scanned = 0

    print()
    print("=" * 80)
    print("SCANNING SOURCE2 + SOURCE3")
    print("=" * 80)

    for source_path in [
        TRAIN_SOURCE2,
        TRAIN_SOURCE3,
    ]:

        print(
            f"\nScanning {source_path.name}..."
        )

        for chunk in pd.read_csv(
            source_path,
            sep="\t",
            dtype=str,
            chunksize=CHUNK_SIZE,
        ):

            for row in chunk.itertuples(
                index=False
            ):

                total_scanned += 1

                source_id = row.entity_id

                country = country_key(
                    row.country
                )

                name = normalize_name(
                    row.business_name
                )

                address = normalize_address(
                    row.business_address
                )

                # -------------------------------------------------
                # Name-pair frequencies
                # -------------------------------------------------

                pair_keys = (
                    name_pair_keys(
                        country,
                        name,
                    )
                    & required_name_pairs
                )

                for key in pair_keys:
                    name_pair_frequency[key] += 1

                # -------------------------------------------------
                # Address-token frequencies
                # -------------------------------------------------

                addr_keys = (
                    address_token_keys(
                        country,
                        address,
                    )
                    & required_address_tokens
                )

                for key in addr_keys:
                    address_token_frequency[key] += 1

                # -------------------------------------------------
                # Retain only source rows that are known true
                # matches for the validation sample.
                # -------------------------------------------------

                if source_id in source_to_s1:

                    true_source_rows[
                        source_id
                    ] = {
                        "country": country,
                        "name": name,
                        "address": address,
                        "name_pairs": name_pair_keys(
                            country,
                            name,
                        ),
                        "address_tokens": address_token_keys(
                            country,
                            address,
                        ),
                        "prefix_house": prefix_house_key(
                            country,
                            name,
                            address,
                        ),
                    }

        print(
            f"  scanned so far: "
            f"{total_scanned:,}"
        )

    print(
        f"\nTotal source rows scanned: "
        f"{total_scanned:,}"
    )

    print(
        f"True source rows retained: "
        f"{len(true_source_rows):,}"
    )

    # -------------------------------------------------------------
    # Evaluate every true pair
    # -------------------------------------------------------------

    print()
    print("=" * 80)
    print("EVALUATING TRUE PAIRS")
    print("=" * 80)

    rows = []

    baseline_recovered = 0
    address_recovery_recovered = 0
    total_recovered = 0

    for counter, s1_id in enumerate(
        validation_ids,
        start=1,
    ):

        s1_row = s1[s1_id]

        for source_id in truth.get(
            s1_id,
            set(),
        ):

            source = true_source_rows.get(
                source_id
            )

            if source is None:
                print(
                    f"WARNING: true source row "
                    f"not found: {source_id}"
                )
                continue

            same_country = (
                s1_row["country"]
                == source["country"]
            )

            # -----------------------------------------------------
            # Frozen blocker: exact name
            # -----------------------------------------------------

            name_exact = (
                same_country
                and bool(s1_row["name"])
                and s1_row["name"]
                == source["name"]
            )

            # -----------------------------------------------------
            # Frozen blocker: compact name
            # -----------------------------------------------------

            s1_compact = compact_text(
                s1_row["name"]
            )

            source_compact = compact_text(
                source["name"]
            )

            name_compact = (
                same_country
                and bool(s1_compact)
                and s1_compact
                == source_compact
            )

            # -----------------------------------------------------
            # Frozen blocker: prefix + house
            # -----------------------------------------------------

            prefix_house = (
                same_country
                and s1_row["prefix_house"]
                is not None
                and s1_row["prefix_house"]
                == source["prefix_house"]
            )

            # -----------------------------------------------------
            # Frozen blocker: name-pair <= 500
            # -----------------------------------------------------

            shared_name_pairs = (
                s1_row["name_pairs"]
                & source["name_pairs"]
            )

            allowed_name_pairs = {
                key
                for key in shared_name_pairs
                if name_pair_frequency[key]
                <= NAME_PAIR_THRESHOLD
            }

            name_pair_hit = (
                same_country
                and bool(allowed_name_pairs)
            )

            # -----------------------------------------------------
            # Frozen blocker: address-token <= 100
            # -----------------------------------------------------

            shared_address_tokens = (
                s1_row["address_tokens"]
                & source["address_tokens"]
            )

            allowed_address_tokens = {
                key
                for key in shared_address_tokens
                if address_token_frequency[key]
                <= ADDRESS_TOKEN_THRESHOLD
            }

            address_token_hit = (
                same_country
                and bool(allowed_address_tokens)
            )

            # -----------------------------------------------------
            # Address-dominant recovery
            #
            # The recovery is only possible when the pair shares
            # an allowed rare address token.
            # -----------------------------------------------------

            address_ratio = 0.0
            address_token_set_ratio = 0.0
            address_overlap = 0.0

            address_recovery = False

            if allowed_address_tokens:

                address_ratio = (
                    ratio(
                        s1_row["address"],
                        source["address"],
                    )
                    / 100.0
                )

                address_token_set_ratio = (
                    token_set_ratio(
                        s1_row["address"],
                        source["address"],
                    )
                    / 100.0
                )

                address_overlap = (
                    token_overlap(
                        s1_row["address"],
                        source["address"],
                    )
                )

                address_recovery = (
                    address_token_set_ratio
                    >= ADDRESS_RECOVERY_SET_RATIO
                    and address_overlap
                    >= ADDRESS_RECOVERY_OVERLAP
                )

            # -----------------------------------------------------
            # Final frozen blocker
            # -----------------------------------------------------

            baseline_hit = (
                name_exact
                or name_compact
                or prefix_house
                or name_pair_hit
                or address_token_hit
            )

            frozen_hit = (
                baseline_hit
                or address_recovery
            )

            if baseline_hit:
                baseline_recovered += 1

            if address_recovery:
                address_recovery_recovered += 1

            if frozen_hit:
                total_recovered += 1

            # -----------------------------------------------------
            # Diagnostic similarities
            # -----------------------------------------------------

            name_ratio = (
                ratio(
                    s1_row["name"],
                    source["name"],
                )
                / 100.0
            )

            name_token_set_ratio = (
                token_set_ratio(
                    s1_row["name"],
                    source["name"],
                )
                / 100.0
            )

            name_overlap = token_overlap(
                s1_row["name"],
                source["name"],
            )

            address_exact = (
                bool(s1_row["address"])
                and s1_row["address"]
                == source["address"]
            )

            # -----------------------------------------------------
            # Only misses are the primary analysis output.
            # -----------------------------------------------------

            if not frozen_hit:

                rows.append(
                    {
                        "source1_id": s1_id,
                        "source_id": source_id,
                        "country_same": int(
                            same_country
                        ),

                        "name_exact": int(
                            name_exact
                        ),
                        "name_compact": int(
                            name_compact
                        ),
                        "prefix_house": int(
                            prefix_house
                        ),
                        "name_pair_hit": int(
                            name_pair_hit
                        ),
                        "address_token_hit": int(
                            address_token_hit
                        ),
                        "address_recovery": int(
                            address_recovery
                        ),

                        "shared_name_pair_count": len(
                            shared_name_pairs
                        ),
                        "allowed_name_pair_count": len(
                            allowed_name_pairs
                        ),

                        "shared_address_token_count": len(
                            shared_address_tokens
                        ),
                        "allowed_address_token_count": len(
                            allowed_address_tokens
                        ),

                        "name_ratio": name_ratio,
                        "name_token_set_ratio":
                            name_token_set_ratio,
                        "name_token_overlap":
                            name_overlap,

                        "address_exact": int(
                            address_exact
                        ),
                        "address_ratio":
                            address_ratio,
                        "address_token_set_ratio":
                            address_token_set_ratio,
                        "address_token_overlap":
                            address_overlap,

                        "s1_name": s1_row["name"],
                        "source_name": source["name"],
                        "s1_address":
                            s1_row["address"],
                        "source_address":
                            source["address"],
                    }
                )

        if counter % 5_000 == 0:
            print(
                f"  processed S1: "
                f"{counter:,} / "
                f"{len(validation_ids):,}"
            )

    # -------------------------------------------------------------
    # Results
    # -------------------------------------------------------------

    missed = pd.DataFrame(rows)

    print()
    print("=" * 80)
    print("BLOCKING MISS RESULTS")
    print("=" * 80)

    print(
        f"Total true pairs: "
        f"{total_true_pairs:,}"
    )

    print(
        f"Recovered by frozen baseline: "
        f"{baseline_recovered:,}"
    )

    print(
        f"Recovered by address recovery: "
        f"{address_recovery_recovered:,}"
    )

    print(
        f"Recovered by frozen blocker: "
        f"{total_recovered:,}"
    )

    frozen_recall = (
        total_recovered
        / total_true_pairs
        if total_true_pairs
        else 0.0
    )

    print(
        f"Frozen blocking recall: "
        f"{frozen_recall:.4%}"
    )

    print(
        f"Missed true pairs: "
        f"{len(missed):,}"
    )

    if not missed.empty:

        print()
        print("-" * 80)
        print("MISSED-PAIR DIAGNOSTICS")
        print("-" * 80)

        n = len(missed)

        diagnostics = {
            "name_ratio >= 0.90":
                (missed["name_ratio"] >= 0.90).sum(),

            "name_ratio >= 0.75":
                (missed["name_ratio"] >= 0.75).sum(),

            "name_token_set_ratio >= 0.90":
                (
                    missed["name_token_set_ratio"]
                    >= 0.90
                ).sum(),

            "name_token_set_ratio >= 0.75":
                (
                    missed["name_token_set_ratio"]
                    >= 0.75
                ).sum(),

            "address_exact":
                missed["address_exact"].sum(),

            "address_ratio >= 0.90":
                (missed["address_ratio"] >= 0.90).sum(),

            "address_ratio >= 0.75":
                (missed["address_ratio"] >= 0.75).sum(),

            "address_token_set_ratio >= 0.90":
                (
                    missed["address_token_set_ratio"]
                    >= 0.90
                ).sum(),

            "address_token_set_ratio >= 0.75":
                (
                    missed["address_token_set_ratio"]
                    >= 0.75
                ).sum(),

            "address_overlap >= 0.80":
                (
                    missed["address_token_overlap"]
                    >= 0.80
                ).sum(),

            "shared address token":
                (
                    missed["shared_address_token_count"]
                    > 0
                ).sum(),

            "allowed address token":
                (
                    missed["allowed_address_token_count"]
                    > 0
                ).sum(),

            "shared name pair":
                (
                    missed["shared_name_pair_count"]
                    > 0
                ).sum(),
        }

        for label, count in diagnostics.items():

            print(
                f"{label:<42} "
                f"{count:>8,} "
                f"({count / n:.2%})"
            )

        # ---------------------------------------------------------
        # Strong combined signals
        # ---------------------------------------------------------

        strong_name = (
            missed["name_ratio"] >= 0.90
        )

        strong_address = (
            missed["address_ratio"] >= 0.90
        )

        strong_address_tokens = (
            missed["address_token_set_ratio"]
            >= 0.90
        )

        print()
        print("Combined signals:")
        print(
            "strong name + strong address: "
            f"{(strong_name & strong_address).sum():,}"
        )

        print(
            "strong name + weak address: "
            f"{(strong_name & ~strong_address).sum():,}"
        )

        print(
            "weak name + strong address: "
            f"{(~strong_name & strong_address).sum():,}"
        )

        print(
            "strong address-token similarity: "
            f"{strong_address_tokens.sum():,}"
        )

        # ---------------------------------------------------------
        # Save misses
        # ---------------------------------------------------------

        missed.to_csv(
            OUTPUT_FILE,
            index=False,
        )

        print()
        print(
            f"Saved miss details: "
            f"{OUTPUT_FILE}"
        )

        # ---------------------------------------------------------
        # Top miss examples
        # ---------------------------------------------------------

        print()
        print("-" * 80)
        print("TOP ADDRESS-DOMINANT MISSES")
        print("-" * 80)

        examples = missed.sort_values(
            [
                "address_token_set_ratio",
                "address_ratio",
            ],
            ascending=False,
        ).head(20)

        for row in examples.itertuples(
            index=False
        ):

            print()
            print(
                f"S1: {row.source1_id}"
            )
            print(
                f"S2/S3: {row.source_id}"
            )
            print(
                f"Name: "
                f"{row.s1_name!r}"
                f"  <->  "
                f"{row.source_name!r}"
            )
            print(
                f"Address: "
                f"{row.s1_address!r}"
                f"  <->  "
                f"{row.source_address!r}"
            )
            print(
                f"name_ratio="
                f"{row.name_ratio:.3f}, "
                f"address_ratio="
                f"{row.address_ratio:.3f}, "
                f"address_token_set_ratio="
                f"{row.address_token_set_ratio:.3f}, "
                f"address_overlap="
                f"{row.address_token_overlap:.3f}"
            )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()