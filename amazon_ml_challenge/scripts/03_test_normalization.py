from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.normalization import (
    normalize_text,
    normalize_name,
    normalize_address,
    compact_text,
    token_set,
    token_list,
)


TEST_VALUES = [
    "Pvt. EFS Print Ventures Ltd.",
    "SCI Ptit Àmicale",
    "Fractales Amis Groupe S.A.S",
    "  Prime   Money  ",
    "B+ Retail Inc",
    "Orelee's Barbershop",
    "राम मार्केटिंग प्राइवेट लिमिटेड",
    "आदित्य प्रॉपर्टीज एलएलपी",
    "मॉडर्न फाइनेंस",
    "Shri Supreme Consulting Private  (Limited)",
    "G.t. Karnal Road, Industrial Area, New Delhi, null, A-68, दिल्ली",
    "",
    None,
]


def main():
    print("=" * 80)
    print("NORMALIZATION TEST")
    print("=" * 80)

    for value in TEST_VALUES:

        print("\n" + "-" * 80)
        print("Original:")
        print(repr(value))

        normalized = normalize_text(value)

        print("\nNormalized:")
        print(repr(normalized))

        print("\nCompact:")
        print(repr(compact_text(value)))

        print("\nTokens:")
        print(token_list(value))

        print("\nToken set:")
        print(token_set(value))


if __name__ == "__main__":
    main()