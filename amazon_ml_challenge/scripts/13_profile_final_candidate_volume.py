"""
Fast blocking-frequency diagnostic.

This is intentionally NOT exact candidate-volume profiling.

It measures how frequently each final blocking key occurs in S2/S3
for the keys represented by a 20K S1 sample.

The sum of frequencies across blocks is an upper bound because the
same source entity may be recovered by multiple blocking keys.

Purpose:
    Identify which blocking strategy creates large candidate volumes.

This should be substantially faster than building Python candidate sets.
"""

from collections import Counter

import numpy as np
import pandas as pd

from src.config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TSV_SEPARATOR,
)

from src.blocking import (
    COMMON_BUSINESS_TOKENS,
    COMMON_ADDRESS_TOKENS,
)


RANDOM_STATE = 42
SAMPLE_SIZE = 20_000
CHUNK_SIZE = 250_000

USECOLS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]


# ---------------------------------------------------------------------
# Vectorized normalization
# ---------------------------------------------------------------------

def normalize_series(series):
    """
    Lightweight vectorized equivalent of normalize_text().

    NFKC
    casefold
    punctuation/symbol -> space
    whitespace collapse
    """

    return (
        series
        .fillna("")
        .astype(str)
        .str.normalize("NFKC")
        .str.casefold()
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def make_country(series):
    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
    )


# ---------------------------------------------------------------------
# Sample keys
# ---------------------------------------------------------------------

def sample_block_keys(sample):
    """
    Generate block keys represented by the S1 sample.

    Returns a dictionary of sets:

        {
            "name_exact": {...},
            "name_compact": {...},
            "name_prefix": {...},
            "house_number": {...},
            "address_token": {...},
            "name_token": {...},
            "name_pair": {...},
        }
    """

    country = make_country(sample["country"])
    name = normalize_series(sample["business_name"])
    address = normalize_series(sample["business_address"])

    result = {}

    # -------------------------------------------------------------
    # Exact name
    # -------------------------------------------------------------

    result["name_exact"] = set(
        (country + "|name|" + name)
        .loc[name.ne("") & country.ne("")]
    )

    # -------------------------------------------------------------
    # Compact name
    # -------------------------------------------------------------

    compact = name.str.replace(" ", "", regex=False)

    result["name_compact"] = set(
        (country + "|compact_name|" + compact)
        .loc[compact.ne("") & country.ne("")]
    )

    # -------------------------------------------------------------
    # Name prefix
    # -------------------------------------------------------------

    prefix = compact.str.slice(0, 8)

    result["name_prefix"] = set(
        (country + "|name_prefix|" + prefix)
        .loc[
            (prefix.str.len() >= 8)
            & country.ne("")
        ]
    )

    # -------------------------------------------------------------
    # House number
    # -------------------------------------------------------------

    house = address.str.extract(
        r"(\S*\d\S*)",
        expand=False,
    ).fillna("")

    result["house_number"] = set(
        (country + "|house_number|" + house)
        .loc[house.ne("") & country.ne("")]
    )

    # -------------------------------------------------------------
    # Address tokens
    # -------------------------------------------------------------

    address_tokens = address.str.split().explode()

    address_tokens = address_tokens[
        address_tokens.str.len().ge(4)
        & ~address_tokens.isin(COMMON_ADDRESS_TOKENS)
    ]

    address_tokens = address_tokens.drop_duplicates()

    # Need country aligned to exploded rows.
    address_country = country.reindex(
        address_tokens.index.get_level_values(0)
        if isinstance(address_tokens.index, pd.MultiIndex)
        else address_tokens.index
    )

    if isinstance(address_tokens.index, pd.MultiIndex):
        address_country = country.iloc[
            address_tokens.index.get_level_values(0)
        ].to_numpy()

        result["address_token"] = set(
            pd.Series(
                address_country + "|address_token|" +
                address_tokens.to_numpy()
            )
        )
    else:
        result["address_token"] = set()

    # -------------------------------------------------------------
    # Name tokens
    # -------------------------------------------------------------

    name_tokens = name.str.split().explode()

    name_tokens = name_tokens[
        name_tokens.str.len().ge(3)
        & ~name_tokens.isin(COMMON_BUSINESS_TOKENS)
    ]

    if isinstance(name_tokens.index, pd.MultiIndex):
        token_country = country.iloc[
            name_tokens.index.get_level_values(0)
        ].to_numpy()

        result["name_token"] = set(
            pd.Series(
                token_country + "|name_token|" +
                name_tokens.to_numpy()
            )
        )
    else:
        result["name_token"] = set()

    return result


# ---------------------------------------------------------------------
# Source frequency profiling
# ---------------------------------------------------------------------

def profile_source(source_path, sample_keys):
    """
    Scan one source and count frequencies of sample-relevant keys.

    Only keys that exist in the 20K S1 sample are retained.
    """

    counters = {
        block: Counter()
        for block in sample_keys
    }

    total_rows = 0

    print()
    print("=" * 72)
    print(f"Scanning {source_path.name}")
    print("=" * 72)

    for chunk_no, chunk in enumerate(
        pd.read_csv(
            source_path,
            sep=TSV_SEPARATOR,
            usecols=USECOLS,
            dtype=str,
            keep_default_na=False,
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):

        total_rows += len(chunk)

        country = make_country(chunk["country"])
        name = normalize_series(chunk["business_name"])
        address = normalize_series(chunk["business_address"])

        compact = name.str.replace(
            " ",
            "",
            regex=False,
        )

        # ---------------------------------------------------------
        # NAME EXACT
        # ---------------------------------------------------------

        if sample_keys["name_exact"]:

            keys = country + "|name|" + name

            mask = keys.isin(sample_keys["name_exact"])

            if mask.any():
                counts = keys.loc[mask].value_counts()
                counters["name_exact"].update(
                    counts.to_dict()
                )

        # ---------------------------------------------------------
        # NAME COMPACT
        # ---------------------------------------------------------

        if sample_keys["name_compact"]:

            keys = country + "|compact_name|" + compact

            mask = keys.isin(sample_keys["name_compact"])

            if mask.any():
                counts = keys.loc[mask].value_counts()
                counters["name_compact"].update(
                    counts.to_dict()
                )

        # ---------------------------------------------------------
        # NAME PREFIX
        # ---------------------------------------------------------

        if sample_keys["name_prefix"]:

            prefix = compact.str.slice(0, 8)

            keys = country + "|name_prefix|" + prefix

            mask = (
                prefix.str.len().ge(8)
                & keys.isin(sample_keys["name_prefix"])
            )

            if mask.any():
                counts = keys.loc[mask].value_counts()
                counters["name_prefix"].update(
                    counts.to_dict()
                )

        # ---------------------------------------------------------
        # HOUSE NUMBER
        # ---------------------------------------------------------

        if sample_keys["house_number"]:

            house = address.str.extract(
                r"(\S*\d\S*)",
                expand=False,
            ).fillna("")

            keys = country + "|house_number|" + house

            mask = (
                house.ne("")
                & keys.isin(sample_keys["house_number"])
            )

            if mask.any():
                counts = keys.loc[mask].value_counts()
                counters["house_number"].update(
                    counts.to_dict()
                )

        # ---------------------------------------------------------
        # ADDRESS TOKENS
        # ---------------------------------------------------------

        if sample_keys["address_token"]:

            token_df = pd.DataFrame(
                {
                    "country": country,
                    "address": address,
                }
            )

            token_df["tokens"] = token_df["address"].str.split()

            exploded = token_df.explode(
                "tokens",
                ignore_index=False,
            )

            tokens = exploded["tokens"]

            mask = (
                tokens.notna()
                & tokens.str.len().ge(4)
                & ~tokens.isin(COMMON_ADDRESS_TOKENS)
            )

            exploded = exploded.loc[mask]

            if len(exploded) > 0:

                keys = (
                    exploded["country"]
                    + "|address_token|"
                    + exploded["tokens"]
                )

                mask = keys.isin(
                    sample_keys["address_token"]
                )

                if mask.any():
                    counts = keys.loc[mask].value_counts()
                    counters["address_token"].update(
                        counts.to_dict()
                    )

        # ---------------------------------------------------------
        # NAME TOKENS
        # ---------------------------------------------------------

        if sample_keys["name_token"]:

            token_df = pd.DataFrame(
                {
                    "country": country,
                    "name": name,
                }
            )

            token_df["tokens"] = token_df["name"].str.split()

            exploded = token_df.explode(
                "tokens",
                ignore_index=False,
            )

            tokens = exploded["tokens"]

            mask = (
                tokens.notna()
                & tokens.str.len().ge(3)
                & ~tokens.isin(COMMON_BUSINESS_TOKENS)
            )

            exploded = exploded.loc[mask]

            if len(exploded) > 0:

                keys = (
                    exploded["country"]
                    + "|name_token|"
                    + exploded["tokens"]
                )

                mask = keys.isin(
                    sample_keys["name_token"]
                )

                if mask.any():
                    counts = keys.loc[mask].value_counts()
                    counters["name_token"].update(
                        counts.to_dict()
                    )

        # ---------------------------------------------------------
        # Progress
        # ---------------------------------------------------------

        print(
            f"  chunk {chunk_no:>2} | "
            f"rows processed: {total_rows:>10,}",
            flush=True,
        )

    return counters


# ---------------------------------------------------------------------
# Convert key frequencies into per-S1 statistics
# ---------------------------------------------------------------------

def frequency_stats(sample, counters):
    """
    For each S1 row calculate the frequency associated with its
    individual blocking key.

    These are NOT exact candidate counts.

    They are block-level upper-bound diagnostics.
    """

    country = make_country(sample["country"])
    name = normalize_series(sample["business_name"])
    address = normalize_series(sample["business_address"])

    compact = name.str.replace(
        " ",
        "",
        regex=False,
    )

    # -------------------------------------------------------------
    # Helper
    # -------------------------------------------------------------

    def values_for_counter(keys, counter):
        return np.asarray(
            [
                counter.get(key, 0)
                for key in keys
            ],
            dtype=np.int64,
        )

    # -------------------------------------------------------------
    # Name exact
    # -------------------------------------------------------------

    exact_keys = (
        country + "|name|" + name
    )

    exact_freq = values_for_counter(
        exact_keys,
        counters["name_exact"],
    )

    # -------------------------------------------------------------
    # Compact
    # -------------------------------------------------------------

    compact_keys = (
        country + "|compact_name|" + compact
    )

    compact_freq = values_for_counter(
        compact_keys,
        counters["name_compact"],
    )

    # -------------------------------------------------------------
    # Prefix
    # -------------------------------------------------------------

    prefix = compact.str.slice(0, 8)

    prefix_keys = (
        country + "|name_prefix|" + prefix
    )

    prefix_freq = values_for_counter(
        prefix_keys,
        counters["name_prefix"],
    )

    # -------------------------------------------------------------
    # House number
    # -------------------------------------------------------------

    house = address.str.extract(
        r"(\S*\d\S*)",
        expand=False,
    ).fillna("")

    house_keys = (
        country + "|house_number|" + house
    )

    house_freq = values_for_counter(
        house_keys,
        counters["house_number"],
    )

    # -------------------------------------------------------------
    # Address token frequency
    #
    # For each S1 row, take the SUM of frequencies of its address
    # tokens. This is deliberately an upper bound.
    # -------------------------------------------------------------

    address_token_freq = np.zeros(
        len(sample),
        dtype=np.int64,
    )

    for i, (c, addr) in enumerate(
        zip(country, address)
    ):

        tokens = {
            token
            for token in addr.split()
            if len(token) >= 4
            and token not in COMMON_ADDRESS_TOKENS
        }

        for token in tokens:
            key = f"{c}|address_token|{token}"
            address_token_freq[i] += counters[
                "address_token"
            ].get(key, 0)

    # -------------------------------------------------------------
    # Name token frequency
    # -------------------------------------------------------------

    name_token_freq = np.zeros(
        len(sample),
        dtype=np.int64,
    )

    for i, (c, nm) in enumerate(
        zip(country, name)
    ):

        tokens = {
            token
            for token in nm.split()
            if len(token) >= 3
            and token not in COMMON_BUSINESS_TOKENS
        }

        for token in tokens:
            key = f"{c}|name_token|{token}"
            name_token_freq[i] += counters[
                "name_token"
            ].get(key, 0)

    # -------------------------------------------------------------
    # Combined upper bound
    # -------------------------------------------------------------

    combined = (
        exact_freq
        + compact_freq
        + prefix_freq
        + house_freq
        + address_token_freq
        + name_token_freq
    )

    return {
        "name_exact": exact_freq,
        "name_compact": compact_freq,
        "name_prefix": prefix_freq,
        "house_number": house_freq,
        "address_token": address_token_freq,
        "name_token": name_token_freq,
        "combined_upper_bound": combined,
    }


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def report(label, values):

    values = np.asarray(values)

    print()
    print(f"{label}")
    print("-" * 60)

    print(f"mean:    {np.mean(values):,.2f}")
    print(f"median:  {np.median(values):,.0f}")
    print(f"p95:     {np.percentile(values, 95):,.0f}")
    print(f"p99:     {np.percentile(values, 99):,.0f}")
    print(f"max:     {np.max(values):,.0f}")

    print(
        f"zero:    {np.sum(values == 0):,} "
        f"({100 * np.mean(values == 0):.2f}%)"
    )


def main():

    print("=" * 72)
    print("FAST BLOCK-FREQUENCY DIAGNOSTIC")
    print("=" * 72)

    # -------------------------------------------------------------
    # S1 sample
    # -------------------------------------------------------------

    print("\nLoading S1 sample...")

    s1 = pd.read_csv(
        TRAIN_SOURCE1,
        sep=TSV_SEPARATOR,
        usecols=USECOLS,
        dtype=str,
        keep_default_na=False,
    )

    sample = (
        s1
        .sample(
            n=SAMPLE_SIZE,
            random_state=RANDOM_STATE,
        )
        .reset_index(drop=True)
    )

    del s1

    print(
        f"S1 sample: {len(sample):,}"
    )

    # -------------------------------------------------------------
    # Build sample keys
    # -------------------------------------------------------------

    print("\nBuilding sample blocking keys...")

    sample_keys = sample_block_keys(sample)

    for block, keys in sample_keys.items():
        print(
            f"  {block:<20} {len(keys):>10,} keys"
        )

    # -------------------------------------------------------------
    # S2
    # -------------------------------------------------------------

    s2_counters = profile_source(
        TRAIN_SOURCE2,
        sample_keys,
    )

    s2_stats = frequency_stats(
        sample,
        s2_counters,
    )

    print()
    print("=" * 72)
    print("S2 BLOCK FREQUENCIES")
    print("=" * 72)

    for block, values in s2_stats.items():
        report(block, values)

    # -------------------------------------------------------------
    # S3
    # -------------------------------------------------------------

    s3_counters = profile_source(
        TRAIN_SOURCE3,
        sample_keys,
    )

    s3_stats = frequency_stats(
        sample,
        s3_counters,
    )

    print()
    print("=" * 72)
    print("S3 BLOCK FREQUENCIES")
    print("=" * 72)

    for block, values in s3_stats.items():
        report(block, values)

    # -------------------------------------------------------------
    # Combined S2 + S3
    # -------------------------------------------------------------

    print()
    print("=" * 72)
    print("COMBINED S2 + S3 BLOCK UPPER BOUNDS")
    print("=" * 72)

    for block in s2_stats:

        combined = (
            s2_stats[block]
            + s3_stats[block]
        )

        report(
            block,
            combined,
        )

    # -------------------------------------------------------------
    # Save compact diagnostic
    # -------------------------------------------------------------

    output = pd.DataFrame(
        {
            "s1_entity_id": sample["entity_id"],
            "s2_name_prefix_freq":
                s2_stats["name_prefix"],
            "s2_address_token_upper":
                s2_stats["address_token"],
            "s2_name_token_upper":
                s2_stats["name_token"],
            "s2_combined_upper":
                s2_stats["combined_upper_bound"],
            "s3_name_prefix_freq":
                s3_stats["name_prefix"],
            "s3_address_token_upper":
                s3_stats["address_token"],
            "s3_name_token_upper":
                s3_stats["name_token"],
            "s3_combined_upper":
                s3_stats["combined_upper_bound"],
        }
    )

    output_path = (
        "output/block_frequency_diagnostic.csv"
    )

    output.to_csv(
        output_path,
        index=False,
    )

    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()