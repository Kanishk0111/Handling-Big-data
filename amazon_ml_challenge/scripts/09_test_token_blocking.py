from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.blocking import (
    name_tokens_key,
    name_token_pair_keys,
)


examples = [
    (
        "US",
        "Orelee's Barbershop",
    ),
    (
        "India",
        "Consulting Nyasa Nursing Private Limited",
    ),
    (
        "India",
        "राम मार्केटिंग प्राइवेट लिमिटेड",
    ),
    (
        "US",
        "LLC Hernandez Colonial Redwood",
    ),
]


for country, name in examples:

    print("\n" + "=" * 80)
    print(name)

    print("\nToken keys:")

    for key in name_tokens_key(
        country,
        name,
    ):
        print(" ", key)

    print("\nToken-pair keys:")

    for key in name_token_pair_keys(
        country,
        name,
    ):
        print(" ", key)