"""
Two-stage candidate generation for business entity resolution.

Stage 1:
    Generate candidates using precise / moderately selective blocks.

Stage 2:
    Add broader candidates only when there is supporting evidence.

The output is a deduplicated set of source entity IDs.
"""

from typing import Optional

from .blocking import (
    name_exact_key,
    name_compact_key,
    name_prefix_key,
    house_number_key,
    address_token_key,
    address_tokens_key,
    name_tokens_key,
    name_token_pair_keys,
)


def _add_key_matches(
    index: dict[str, set[str]],
    key: str,
    candidates: set[str],
) -> None:
    """Add candidates associated with one blocking key."""

    if not key:
        return

    matches = index.get(key)

    if matches:
        candidates.update(matches)


def generate_stage1_keys(
    country: Optional[str],
    business_name: Optional[str],
    business_address: Optional[str],
) -> set[str]:
    """
    Selective blocking keys.

    These are intended to produce the initial candidate pool.
    """

    keys: set[str] = set()

    for key in (
        name_exact_key(country, business_name),
        name_compact_key(country, business_name),
    ):
        if key:
            keys.add(key)

    # Token-pair blocking is considerably more selective than
    # individual-token blocking.
    keys.update(
        name_token_pair_keys(
            country,
            business_name,
        )
    )

    return keys


def generate_stage2_keys(
    country: Optional[str],
    business_name: Optional[str],
    business_address: Optional[str],
) -> set[str]:
    """
    Broader blocking keys.

    These should NOT necessarily be used blindly for every record.
    They are intended to recover difficult matches.
    """

    keys: set[str] = set()

    # Name prefix
    key = name_prefix_key(
        country,
        business_name,
    )

    if key:
        keys.add(key)

    # House number
    key = house_number_key(
        country,
        business_address,
    )

    if key:
        keys.add(key)

    # Address token blocks
    keys.update(
        address_token_key(
            country,
            business_address,
        ) and [
            address_token_key(
                country,
                business_address,
            )
        ] or []
    )

    keys.update(
        address_tokens_key(
            country,
            business_address,
        )
    )

    return keys


def generate_all_keys(
    country: Optional[str],
    business_name: Optional[str],
    business_address: Optional[str],
) -> set[str]:
    """
    Generate every currently available blocking key.
    """

    keys = generate_stage1_keys(
        country,
        business_name,
        business_address,
    )

    keys.update(
        generate_stage2_keys(
            country,
            business_name,
            business_address,
        )
    )

    keys.update(
        name_tokens_key(
            country,
            business_name,
        )
    )

    return keys


def stage1_candidates(
    country: Optional[str],
    business_name: Optional[str],
    business_address: Optional[str],
    indexes: dict[str, dict[str, set[str]]],
) -> set[str]:
    """
    Generate selective initial candidates.

    indexes should contain:

        {
            "name_exact": {...},
            "name_compact": {...},
            "name_token_pair": {...},
        }
    """

    candidates: set[str] = set()

    key = name_exact_key(
        country,
        business_name,
    )

    _add_key_matches(
        indexes.get("name_exact", {}),
        key,
        candidates,
    )

    key = name_compact_key(
        country,
        business_name,
    )

    _add_key_matches(
        indexes.get("name_compact", {}),
        key,
        candidates,
    )

    for key in name_token_pair_keys(
        country,
        business_name,
    ):
        _add_key_matches(
            indexes.get("name_token_pair", {}),
            key,
            candidates,
        )

    return candidates


def stage2_candidates(
    country: Optional[str],
    business_name: Optional[str],
    business_address: Optional[str],
    indexes: dict[str, dict[str, set[str]]],
) -> set[str]:
    """
    Generate broader candidates.

    These candidates should subsequently be filtered/ranked by
    similarity scoring.
    """

    candidates: set[str] = set()

    # ------------------------------------------------------------
    # Name prefix
    # ------------------------------------------------------------

    key = name_prefix_key(
        country,
        business_name,
    )

    _add_key_matches(
        indexes.get("name_prefix", {}),
        key,
        candidates,
    )

    # ------------------------------------------------------------
    # House number
    # ------------------------------------------------------------

    key = house_number_key(
        country,
        business_address,
    )

    _add_key_matches(
        indexes.get("house_number", {}),
        key,
        candidates,
    )

    # ------------------------------------------------------------
    # Address prefix
    # ------------------------------------------------------------

    key = address_token_key(
        country,
        business_address,
    )

    _add_key_matches(
        indexes.get("address_prefix", {}),
        key,
        candidates,
    )

    # ------------------------------------------------------------
    # Reordered address tokens
    # ------------------------------------------------------------

    for key in address_tokens_key(
        country,
        business_address,
    ):
        _add_key_matches(
            indexes.get("address_token", {}),
            key,
            candidates,
        )

    return candidates