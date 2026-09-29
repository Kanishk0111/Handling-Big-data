from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import TRAIN_SOURCE1, TRAIN_GROUND_TRUTH


SEED = 42
VALIDATION_FRACTION = 0.20

OUTPUT_DIR = Path("output")

TRAIN_S1_IDS = OUTPUT_DIR / "validation_train_s1_ids.txt"
VALIDATION_S1_IDS = OUTPUT_DIR / "validation_s1_ids.txt"


def main():

    print("=" * 72)
    print("CREATING ENTITY-LEVEL VALIDATION SPLIT")
    print("=" * 72)

    # ------------------------------------------------------------
    # Load S1 IDs
    # ------------------------------------------------------------

    s1 = pd.read_csv(
        TRAIN_SOURCE1,
        sep="\t",
        dtype=str,
        usecols=["entity_id"],
    )

    s1_ids = s1["entity_id"].tolist()

    print(
        f"Total training S1 entities: "
        f"{len(s1_ids):,}"
    )

    # ------------------------------------------------------------
    # Split by S1 entity
    # ------------------------------------------------------------

    train_ids, validation_ids = train_test_split(
        s1_ids,
        test_size=VALIDATION_FRACTION,
        random_state=SEED,
    )

    train_ids = sorted(train_ids)
    validation_ids = sorted(validation_ids)

    print(
        f"Development S1 entities: "
        f"{len(train_ids):,}"
    )

    print(
        f"Validation S1 entities: "
        f"{len(validation_ids):,}"
    )

    # ------------------------------------------------------------
    # Sanity check
    # ------------------------------------------------------------

    assert not (
        set(train_ids) & set(validation_ids)
    )

    assert (
        len(train_ids) + len(validation_ids)
        == len(s1_ids)
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    OUTPUT_DIR.mkdir(exist_ok=True)

    TRAIN_S1_IDS.write_text(
        "\n".join(train_ids) + "\n"
    )

    VALIDATION_S1_IDS.write_text(
        "\n".join(validation_ids) + "\n"
    )

    print("\nSaved:")

    print(
        f"  {TRAIN_S1_IDS}"
    )

    print(
        f"  {VALIDATION_S1_IDS}"
    )

    print("\n" + "=" * 72)
    print("SPLIT COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()