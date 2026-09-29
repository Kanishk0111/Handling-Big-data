"""
Disk-backed blocking index.

The index is stored in SQLite so that we never need to keep millions
of S2/S3 records or Python dictionaries in RAM.

Initial index contains the selective blocking keys:

    - exact normalized name
    - compact normalized name
    - name token pairs

Each source entity is stored as a globally unique entity_id
(S2-... or S3-...).
"""

import sqlite3
from pathlib import Path
from typing import Iterable

from .blocking import (
    name_exact_key,
    name_compact_key,
    name_token_pair_keys,
)


SCHEMA = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS blocking_index (
    block_type TEXT NOT NULL,
    block_key TEXT NOT NULL,
    entity_id TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_blocking_lookup
ON blocking_index(block_type, block_key);

CREATE INDEX IF NOT EXISTS idx_entity
ON blocking_index(entity_id);
"""


def create_index(db_path: Path) -> sqlite3.Connection:
    """
    Create/open a SQLite blocking index.
    """

    db_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        db_path,
        timeout=120,
    )

    connection.executescript(SCHEMA)

    return connection


def insert_rows(
    connection: sqlite3.Connection,
    rows: Iterable[tuple[str, str, str]],
) -> None:
    """
    Batch insert:

        (block_type, block_key, entity_id)
    """

    connection.executemany(
        """
        INSERT INTO blocking_index (
            block_type,
            block_key,
            entity_id
        )
        VALUES (?, ?, ?)
        """,
        rows,
    )


def lookup(
    connection: sqlite3.Connection,
    block_type: str,
    block_key: str,
) -> list[str]:
    """
    Look up entity IDs for one blocking key.
    """

    if not block_key:
        return []

    rows = connection.execute(
        """
        SELECT entity_id
        FROM blocking_index
        WHERE block_type = ?
          AND block_key = ?
        """,
        (block_type, block_key),
    ).fetchall()

    return [row[0] for row in rows]


def lookup_many(
    connection: sqlite3.Connection,
    block_type: str,
    block_keys: Iterable[str],
) -> set[str]:
    """
    Look up several blocking keys and return a deduplicated
    entity-id set.
    """

    candidates: set[str] = set()

    for key in block_keys:
        if not key:
            continue

        rows = connection.execute(
            """
            SELECT entity_id
            FROM blocking_index
            WHERE block_type = ?
              AND block_key = ?
            """,
            (block_type, key),
        )

        for row in rows:
            candidates.add(row[0])

    return candidates


def count_rows(connection: sqlite3.Connection) -> int:
    result = connection.execute(
        """
        SELECT COUNT(*)
        FROM blocking_index
        """
    ).fetchone()

    return int(result[0])


def count_unique_entities(connection: sqlite3.Connection) -> int:
    result = connection.execute(
        """
        SELECT COUNT(DISTINCT entity_id)
        FROM blocking_index
        """
    ).fetchone()

    return int(result[0])