import json
from collections import OrderedDict
from pathlib import Path

from datasets import load_dataset


def main():
    print("Loading WTQ dataset (stanfordnlp/wikitablequestions, random-split-1)...")
    wtq = load_dataset("stanfordnlp/wikitablequestions", "random-split-1")

    table_schemas = OrderedDict()

    for split_name, split_ds in wtq.items():
        print(f"Scanning split: {split_name} (size={len(split_ds)})")
        for ex in split_ds:
            table = ex["table"]
            table_name = table["name"]          
            header = table["header"]            

            if table_name not in table_schemas:
                table_schemas[table_name] = {
                    "table_name": table_name,
                    "columns": header,
                }

    schema_entries = list(table_schemas.values())
    print(f"Total unique tables found: {len(schema_entries)}")

    out_path = Path("schema_basic.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(schema_entries, f, indent=2, ensure_ascii=False)

    print(f"Wrote basic schema JSON to {out_path}")


if __name__ == "__main__":
    main()
