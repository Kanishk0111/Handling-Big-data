from pathlib import Path

import pandas as pd
from rapidfuzz.fuzz import ratio, token_set_ratio

from src.normalization import token_list


INPUT_FILE = Path(
    "output/blocking_miss_analysis_50k.csv"
)


def token_overlap(a, b):
    a = set(token_list(a))
    b = set(token_list(b))

    if not a or not b:
        return 0.0

    return len(a & b) / min(len(a), len(b))


df = pd.read_csv(
    INPUT_FILE,
    dtype=str,
)


df["name_ratio_actual"] = [
    ratio(
        a or "",
        b or "",
    ) / 100.0
    for a, b in zip(
        df["s1_name"],
        df["source_name"],
    )
]


df["address_exact_actual"] = (
    df["s1_address"].fillna("")
    == df["source_address"].fillna("")
).astype(int)


df["address_ratio_actual"] = [
    ratio(
        a or "",
        b or "",
    ) / 100.0
    for a, b in zip(
        df["s1_address"],
        df["source_address"],
    )
]


df["address_token_set_ratio_actual"] = [
    token_set_ratio(
        a or "",
        b or "",
    ) / 100.0
    for a, b in zip(
        df["s1_address"],
        df["source_address"],
    )
]


df["address_token_overlap_actual"] = [
    token_overlap(
        a or "",
        b or "",
    )
    for a, b in zip(
        df["s1_address"],
        df["source_address"],
    )
]


n = len(df)

print("=" * 80)
print("CORRECTED ADDRESS ANALYSIS")
print("=" * 80)

print(f"Missed true pairs: {n:,}")

tests = {
    "address exact":
        df["address_exact_actual"] == 1,

    "address ratio >= .90":
        df["address_ratio_actual"] >= 0.90,

    "address ratio >= .80":
        df["address_ratio_actual"] >= 0.80,

    "address ratio >= .70":
        df["address_ratio_actual"] >= 0.70,

    "address token-set >= .95":
        df["address_token_set_ratio_actual"] >= 0.95,

    "address token-set >= .90":
        df["address_token_set_ratio_actual"] >= 0.90,

    "address token-set >= .80":
        df["address_token_set_ratio_actual"] >= 0.80,

    "address overlap >= .80":
        df["address_token_overlap_actual"] >= 0.80,

    "address overlap >= .60":
        df["address_token_overlap_actual"] >= 0.60,
}


for label, mask in tests.items():

    count = int(mask.sum())

    print(
        f"{label:<32}"
        f"{count:>8,}"
        f" ({count / n:.2%})"
    )


print()
print("-" * 80)
print("COMBINED RECOVERY SIGNALS")
print("-" * 80)


signals = {
    "exact address + name ratio >= .50":
        (
            (df["address_exact_actual"] == 1)
            & (df["name_ratio_actual"] >= 0.50)
        ),

    "exact address + name ratio >= .75":
        (
            (df["address_exact_actual"] == 1)
            & (df["name_ratio_actual"] >= 0.75)
        ),

    "addr token-set >= .90 + overlap >= .80":
        (
            (df["address_token_set_ratio_actual"] >= 0.90)
            & (df["address_token_overlap_actual"] >= 0.80)
        ),

    "addr token-set >= .95 + overlap >= .80":
        (
            (df["address_token_set_ratio_actual"] >= 0.95)
            & (df["address_token_overlap_actual"] >= 0.80)
        ),

    "addr token-set >= .90 + name ratio >= .50":
        (
            (df["address_token_set_ratio_actual"] >= 0.90)
            & (df["name_ratio_actual"] >= 0.50)
        ),
}


for label, mask in signals.items():

    count = int(mask.sum())

    print(
        f"{label:<48}"
        f"{count:>8,}"
        f" ({count / n:.2%})"
    )


print()
print("=" * 80)
print("DONE")
print("=" * 80)