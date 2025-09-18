import ijson

path = "/projectnb/sparkgrp/ml-bpl-rag-data/extraneous/metadata/bpl_data.json"

with open(path, "r") as f:
    parser = ijson.parse(f)
    for prefix, event, value in parser:
        print(f"{prefix} — {event} — {value}")
        if prefix.count('.') > 2:  # stop after we get into nested elements
            break

with open(path, "r") as f:
    items = ijson.items(f, "Data.item")
    for i, item in enumerate(items):
        print(f"\nItem {i + 1}:\n", item)
        if i >= 2:  # only show first 3 items
            break