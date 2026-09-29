"""
Build the disk-backed selective blocking index.

Indexes S2 and S3 in chunks.

Only these blocks are included initially:

    name_exact
    name_compact
    name_token_pair

The source files are NEVER loaded completely into memory.
"""

import sqlite3
from pathlib import Path

import pandas as pd

from src.config import (
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TSV_SEPARATOR,
)

from src.blocking import (
    name_exact_key,
    name_compact_key,
    name_token_pair_keys,
)

from src.disk_index import (
    create_index,
    insert_rows,
    count_rows,
    count_unique_entities,
)


CHUNK_SIZE = 100_000

INDEX_PATH = Path(
    "output/selective_blocking_index.sqlite"
)


USECOLS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]


def process_source(
    connection: sqlite3.Connection,
    source_path: Path,
):
    print()
    print("=" * 72)
    print(f"Processing {source_path.name}")
    print("=" * 72)

    total_rows = 0
    indexed_rows = 0

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

        rows = []

        for row in chunk.itertuples(index=False):

            entity_id = row.entity_id
            country = row.country
            name = row.business_name

            # ----------------------------------------------------
            # Exact name
            # ----------------------------------------------------

            key = name_exact_key(
                country,
                name,
            )

            if key:
                rows.append(
                    (
                        "name_exact",
                        key,
                        entity_id,
                    )
                )

            # ----------------------------------------------------
            # Compact name
            # ----------------------------------------------------

            key = name_compact_key(
                country,
                name,
            )

            if key:
                rows.append(
                    (
                        "name_compact",
                        key,
                        entity_id,
                    )
                )

            # ----------------------------------------------------
            # Name token pairs
            # ----------------------------------------------------

            for key in name_token_pair_keys(
                country,
                name,
            ):
                rows.append(
                    (
                        "name_token_pair",
                        key,
                        entity_id,
                    )
                )

        if rows:
            insert_rows(
                connection,
                rows,
            )

        connection.commit()

        total_rows += len(chunk)
        indexed_rows += len(rows)

        print(
            f"  chunk {chunk_no:>3} | "
            f"source rows: {total_rows:>10,} | "
            f"index rows: {indexed_rows:>10,}",
            flush=True,
        )


def main():

    print("=" * 72)
    print("BUILDING SELECTIVE DISK INDEX")
    print("=" * 72)

    # Start clean.
    if INDEX_PATH.exists():
        print(
            f"\nRemoving existing index: {INDEX_PATH}"
        )
        INDEX_PATH.unlink()

    # Remove SQLite WAL/SHM leftovers if present.
    for suffix in ("-wal", "-shm"):
        path = Path(
            str(INDEX_PATH) + suffix
        )
        if path.exists():
            path.unlink()

    connection = create_index(
        INDEX_PATH
    )

    try:

        process_source(
            connection,
            TRAIN_SOURCE2,
        )

        process_source(
            connection,
            TRAIN_SOURCE3,
        )

        connection.commit()

        print()
        print("=" * 72)
        print("INDEX COMPLETE")
        print("=" * 72)

        print(
            f"Index rows:       "
            f"{count_rows(connection):,}"
        )

        print(
            f"Unique entities:   "
            f"{count_unique_entities(connection):,}"
        )

        print(
            f"Database size:     "
            f"{INDEX_PATH.stat().st_size / (1024 ** 3):.2f} GB"
        )

    finally:
        connection.close()


if __name__ == "__main__":
    main()