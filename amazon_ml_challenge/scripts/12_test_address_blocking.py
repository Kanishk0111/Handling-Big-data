from src.blocking import address_tokens_key


examples = [
    (
        "India",
        "Property No- 4/06 (Hig), Sector-4 Vaishali, Ghaziabad, Uttar Pradesh",
    ),
    (
        "India",
        "PLOT 13 PROPERTY NO- 4/06 (HIG), SECTOR-4 VAISHALI, GHAZIABAD, Uttar Pradesh",
    ),
    (
        "India",
        "Hotel Mansingh, Sansar Chandra Road, Jaipur, Rajasthan",
    ),
]


for country, address in examples:
    print("=" * 80)
    print(address)

    for key in address_tokens_key(country, address):
        print(" ", key)