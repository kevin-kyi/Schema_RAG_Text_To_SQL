import json
import sqlite3
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import shutil

SPIDER_FILE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"

MAX_ROWS_PER_TABLE: Optional[int] = None  

PRUNE_NON_DEV_DATABASES = True


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_spider_downloaded(spider_raw_dir: Path) -> Path:

    spider_raw_dir.mkdir(parents=True, exist_ok=True)

    existing_tables = list(spider_raw_dir.rglob("tables.json"))
    if existing_tables:
        spider_root = existing_tables[0].parent
        print(f"Found existing tables.json at {existing_tables[0]}")
        print(f"Treating {spider_root} as spider_root.")
        return spider_root

    zip_path = spider_raw_dir / "spider_data.zip"
    if not zip_path.exists():
        from shutil import which
        if which("gdown") is None:
            raise RuntimeError(
                "gdown not found.\n"

            )
        subprocess.check_call(
            ["gdown", "--id", SPIDER_FILE_ID, "-O", str(zip_path)]
        )
        print(f"Downloaded spider_data.zip to {zip_path}")
    else:
        print(f"Using existing zip at {zip_path}")

    print(f"Extracting {zip_path} into {spider_raw_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(spider_raw_dir)

    existing_tables = list(spider_raw_dir.rglob("tables.json"))
    if not existing_tables:
        raise FileNotFoundError(
            f"After extracting, could not find tables.json under {spider_raw_dir}"
        )
    spider_root = existing_tables[0].parent
    print(f"Spider root detected at: {spider_root}")
    return spider_root


def get_dev_db_ids(spider_root: Path) -> List[str]:
    dev_path = spider_root / "dev.json"
    if not dev_path.exists():
        raise FileNotFoundError(f"Cannot find dev.json at {dev_path}")
    dev_data = load_json(dev_path)
    db_ids = sorted({ex["db_id"] for ex in dev_data})
    print(f"Dev split has {len(db_ids)} databases: {db_ids}")
    return db_ids


def extract_tables_for_db(
    db_meta: Dict[str, Any],
    db_file: Path,
    out_tables_dir: Path,
) -> List[Dict[str, Any]]:
    db_id = db_meta["db_id"]
    table_names = db_meta["table_names"]
    table_names_original = db_meta["table_names_original"]
    column_names = db_meta["column_names"]
    column_types = db_meta["column_types"]

    cols_by_table = {i: [] for i in range(len(table_names))}
    for col_idx, (t_idx, col_name) in enumerate(column_names):
        if t_idx < 0:  
            continue
        col_type = column_types[col_idx]
        cols_by_table[t_idx].append(
            {
                "col_idx": col_idx,
                "name": col_name,
                "type": col_type,
            }
        )

    if not db_file.exists():
        raise FileNotFoundError(f"SQLite file not found for db_id={db_id}: {db_file}")

    conn = sqlite3.connect(str(db_file))
    schema_entries: List[Dict[str, Any]] = []

    try:
        for t_idx, table_name in enumerate(table_names):
            table_name_original = table_names_original[t_idx]

            try:
                query = f'SELECT * FROM "{table_name_original}"'
                if MAX_ROWS_PER_TABLE is not None:
                    query += f" LIMIT {int(MAX_ROWS_PER_TABLE)}"

                df = pd.read_sql_query(query, conn)
            except Exception as e:
                print(
                    f"failed to read table {table_name_original} "
                    f"from {db_file}: {e}"
                )
                continue

            out_tables_dir.mkdir(parents=True, exist_ok=True)
            csv_name = f"{db_id}__{table_name}.csv"
            csv_path = out_tables_dir / csv_name
            df.to_csv(csv_path, index=False)

            cols = cols_by_table.get(t_idx, [])
            cols_desc = "; ".join(
                f"{c['name']} ({c['type']})" for c in cols
            ) or "no columns"

            schema_text = (
                f"Database {db_id}, table {table_name}. "
                f"Columns: {cols_desc}."
            )

            schema_entries.append(
                {
                    "db_id": db_id,
                    "table_name": table_name,
                    "table_name_original": table_name_original,
                    "csv_path": str(csv_path.resolve()),
                    "columns": cols,
                    "schema_text": schema_text,
                }
            )

            print(
                f"Exported {db_id}.{table_name} "
                f"({len(df)} rows, {len(df.columns)} cols) -> {csv_path}"
            )
    finally:
        conn.close()

    return schema_entries


def main():
    script_dir = Path(__file__).parent
    project_root = (script_dir / "..").resolve()

    data_root = project_root / "data"
    spider_raw_dir = data_root / "spider_raw"

    print(f"Project root: {project_root}")
    print(f"Data root:    {data_root}")

    spider_root = ensure_spider_downloaded(spider_raw_dir)

    tables_json_path = spider_root / "tables.json"
    db_root = spider_root / "database"

    if not tables_json_path.exists():
        raise FileNotFoundError(f"Cannot find tables.json at {tables_json_path}")

    print(f"Reading schema from: {tables_json_path}")
    tables_data = load_json(tables_json_path)

    dev_db_ids = set(get_dev_db_ids(spider_root))

    if PRUNE_NON_DEV_DATABASES:
        print("Pruning non-dev databases from spider_root/database/ ...")
        for db_dir in db_root.iterdir():
            if not db_dir.is_dir():
                continue
            db_id = db_dir.name
            if db_id not in dev_db_ids:
                print(f"  - Removing non-dev database folder: {db_dir}")
                shutil.rmtree(db_dir)
        print("Pruning complete.")

    out_tables_dir = data_root / "tables"
    schema_json_path = data_root / "schema.json"

    all_schema_entries: List[Dict[str, Any]] = []

    for db_meta in tables_data:
        db_id = db_meta["db_id"]
        if db_id not in dev_db_ids:
            continue

        print(f"\n=== Processing database: {db_id} ===")
        db_file = db_root / db_id / f"{db_id}.sqlite"
        schema_entries = extract_tables_for_db(db_meta, db_file, out_tables_dir)
        all_schema_entries.extend(schema_entries)

    print(
        f"\nWriting schema metadata for {len(all_schema_entries)} tables "
        f"to {schema_json_path}"
    )
    schema_json_path.parent.mkdir(parents=True, exist_ok=True)
    with schema_json_path.open("w", encoding="utf-8") as f:
        json.dump(all_schema_entries, f, indent=2, ensure_ascii=False)

    print("FINISHED! 20 dev databases exported and schema.json created.")

if __name__ == "__main__":
    main()