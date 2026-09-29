"""
Blocking-key utilities.

This module generates lightweight deterministic keys.
It does NOT perform fuzzy matching.
"""

from typing import Optional

from .normalization import (
    normalize_name,
    normalize_address,
    compact_text,
    token_list,
)


def country_key(country: Optional[str]) -> str:
    if country is None:
        return ""

    return str(country).strip().casefold()


def name_exact_key(
    country: Optional[str],
    business_name: Optional[str],
) -> str:

    country = country_key(country)
    name = normalize_name(business_name)

    if not country or not name:
        return ""

    return f"{country}|name|{name}"


def name_compact_key(
    country: Optional[str],
    business_name: Optional[str],
) -> str:

    country = country_key(country)
    name = compact_text(business_name)

    if not country or not name:
        return ""

    return f"{country}|compact_name|{name}"


def name_prefix_key(
    country: Optional[str],
    business_name: Optional[str],
    prefix_length: int = 8,
) -> str:

    country = country_key(country)
    name = compact_text(business_name)

    if not country or not name:
        return ""

    if len(name) < prefix_length:
        return ""

    return f"{country}|name_prefix|{name[:prefix_length]}"


def address_token_key(
    country: Optional[str],
    business_address: Optional[str],
    token_count: int = 3,
) -> str:

    country = country_key(country)
    tokens = token_list(business_address)

    if not country or len(tokens) < token_count:
        return ""

    key_tokens = tokens[:token_count]

    return (
        f"{country}|address_prefix|"
        + "|".join(key_tokens)
    )


def house_number_key(
    country: Optional[str],
    business_address: Optional[str],
) -> str:

    country = country_key(country)
    address = normalize_address(business_address)

    if not country or not address:
        return ""

    tokens = address.split()

    for token in tokens:
        # Look for a token containing at least one digit.
        if any(char.isdigit() for char in token):
            return f"{country}|house_number|{token}"

    return ""


def generate_blocking_keys(
    country: Optional[str],
    business_name: Optional[str],
    business_address: Optional[str],
) -> dict[str, str]:

    return {
        "name_exact": name_exact_key(
            country,
            business_name,
        ),
        "name_compact": name_compact_key(
            country,
            business_name,
        ),
        "name_prefix": name_prefix_key(
            country,
            business_name,
        ),
        "address_prefix": address_token_key(
            country,
            business_address,
        ),
        "house_number": house_number_key(
            country,
            business_address,
        ),
    }

def name_tokens_key(
    country: Optional[str],
    business_name: Optional[str],
) -> list[str]:
    """
    Return country-qualified individual name-token keys.

    Common legal/entity-type tokens are excluded because
    they are usually too frequent to be useful for blocking.
    """
    country = country_key(country)
    tokens = token_list(business_name)

    if not country:
        return []

    keys = []

    for token in tokens:
        if len(token) < 3:
            continue

        if token in COMMON_BUSINESS_TOKENS:
            continue

        keys.append(
            f"{country}|name_token|{token}"
        )

    return keys


def name_token_pair_keys(
    country: Optional[str],
    business_name: Optional[str],
) -> list[str]:
    """
    Generate country-qualified pairs of informative name tokens.

    Common legal/entity-type tokens are excluded and token
    order is ignored.
    """
    country = country_key(country)

    tokens = [
        token
        for token in token_list(business_name)
        if len(token) >= 3
        and token not in COMMON_BUSINESS_TOKENS
    ]

    if not country or len(tokens) < 2:
        return []

    unique_tokens = sorted(set(tokens))

    keys = []

    for i in range(len(unique_tokens)):
        for j in range(i + 1, len(unique_tokens)):
            keys.append(
                f"{country}|name_pair|"
                f"{unique_tokens[i]}|"
                f"{unique_tokens[j]}"
            )

    return keys

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


def address_tokens_key(
    country: Optional[str],
    business_address: Optional[str],
) -> list[str]:
    """
    Generate country-qualified keys for informative address tokens.

    Token order is ignored.
    Common structural address words are excluded.
    """
    country = country_key(country)

    tokens = [
        token
        for token in token_list(business_address)
        if len(token) >= 4
        and token not in COMMON_ADDRESS_TOKENS
    ]

    if not country:
        return []

    unique_tokens = sorted(set(tokens))

    return [
        f"{country}|address_token|{token}"
        for token in unique_tokens
    ]