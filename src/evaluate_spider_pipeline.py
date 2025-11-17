from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from schema_rag import SchemaRAGPipeline


@dataclass
class SpiderExample:
    question: str
    db_id: str
    sql_query: str
    gold_table_name: str  # e.g. "singer"
    gold_answer: Any      # scalar or list of rows

def load_tables_metadata(tables_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load tables.json and build a dict: db_id -> schema_entry.
    """
    with tables_path.open("r", encoding="utf-8") as f:
        entries = json.load(f)
    by_db = {entry["db_id"]: entry for entry in entries}
    return by_db

def get_single_table_examples(
    dev_path: Path,
    tables_meta: Dict[str, Dict[str, Any]],
    db_root: Path,
    max_examples: Optional[int] = None,
) -> List[SpiderExample]:
    """
    Load Spider dev.json and extract single-table examples with gold answers.

    Returns a list of SpiderExample objects.
    """
    with dev_path.open("r", encoding="utf-8") as f:
        dev_data = json.load(f)

    examples: List[SpiderExample] = []

    for ex in dev_data:
        db_id = ex["db_id"]
        question = ex["question"]
        sql_struct = ex["sql"]
        sql_query = ex["query"]

        # Only keep single-table examples: exactly one table_unit in FROM
        table_units = sql_struct["from"]["table_units"]
        if len(table_units) != 1:
            continue

        unit_type, table_idx = table_units[0]
        if unit_type != "table_unit":
            continue

        # Map table_idx -> table name
        schema_entry = tables_meta.get(db_id)
        if schema_entry is None:
            continue

        table_names = schema_entry.get("table_names_original") or schema_entry["table_names"]
        if not (0 <= table_idx < len(table_names)):
            continue
        gold_table_name = table_names[table_idx]

        # Execute the SQL to get gold answer
        db_path = db_root / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            continue

        try:
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()
            cur.execute(sql_query)
            rows = cur.fetchall()
            conn.close()
        except Exception as e:
            print(f"[WARN] Failed to execute query for {db_id}: {sql_query!r} - {e}")
            continue

        # Simplify gold answer representation
        if len(rows) == 1 and len(rows[0]) == 1:
            gold_answer: Any = rows[0][0]  # scalar
        else:
            gold_answer = rows  # list of tuples

        examples.append(
            SpiderExample(
                question=question,
                db_id=db_id,
                sql_query=sql_query,
                gold_table_name=gold_table_name,
                gold_answer=gold_answer,
            )
        )

        if max_examples is not None and len(examples) >= max_examples:
            break

    print(f"[INFO] Collected {len(examples)} single-table examples from {dev_path}")
    return examples


def parse_doc_db_and_table(doc_name: str) -> Tuple[Optional[str], str]:
    """
    Parse db_id and table_name from a TableDocument.name.

    For Spider documents created by load_spider_table_documents, the name is:
        "<db_id>.<table_name>"

    If there is no dot, db_id is None and full name is treated as table_name.
    """
    if "." in doc_name:
        db_id, table_name = doc_name.split(".", 1)
        return db_id, table_name
    else:
        return None, doc_name


def normalize_answer_for_compare(ans: Any) -> str:
    """
    Convert an answer (scalar or list of rows) into a normalized string
    for simple equality comparison.
    """
    if isinstance(ans, (int, float)):
        # Use a stable float formatting for numeric answers
        return str(float(ans)).strip()
    if isinstance(ans, str):
        return ans.strip()

    # List/tuple of rows or other structures
    return str(ans).strip()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SchemaRAGPipeline on Spider dev.json single-table questions."
    )
    parser.add_argument(
        "--spider-dir",
        type=Path,
        required=True,
        help="Path to Spider root (directory containing tables.json, dev.json, database/).",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=100,
        help="Maximum number of dev examples to evaluate (default: 100).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Top-k tables to retrieve for evaluation (default: 1; we use top-1 for TAPAS).",
    )
    parser.add_argument(
        "--max-table-rows",
        type=int,
        default=64,
        help="Max rows per table passed to TAPAS.",
    )
    args = parser.parse_args()

    spider_root = args.spider_dir
    tables_path = spider_root / "tables.json"
    dev_path = spider_root / "dev.json"
    db_root = spider_root / "database"

    if not tables_path.exists():
        raise FileNotFoundError(f"tables.json not found at {tables_path}")
    if not dev_path.exists():
        raise FileNotFoundError(f"dev.json not found at {dev_path}")
    if not db_root.exists():
        raise FileNotFoundError(f"database directory not found at {db_root}")

    print(f"[INFO] Using Spider root: {spider_root}")

    tables_meta = load_tables_metadata(tables_path)
    examples = get_single_table_examples(
        dev_path=dev_path,
        tables_meta=tables_meta,
        db_root=db_root,
        max_examples=args.max_examples,
    )

    if not examples:
        print("[WARN] No single-table examples found. Nothing to evaluate.")
        return

    # Build pipeline with ONLY Spider tables (no CSV data_dir)
    pipeline = SchemaRAGPipeline(
        data_dir=None,
        spider_dir=spider_root,
        max_table_rows=args.max_table_rows,
    )

    total = 0
    correct_db = 0
    correct_table = 0
    correct_answer = 0

    for ex in examples:
        total += 1
        gold_db = ex.db_id
        gold_table = ex.gold_table_name
        gold_ans_norm = normalize_answer_for_compare(ex.gold_answer)

        # Retrieval
        retrieved = pipeline.vector_store.search(ex.question, top_k=args.top_k)
        if not retrieved:
            print(f"[MISS] No retrieval for question: {ex.question!r}")
            continue

        # Look at top-1 for TAPAS evaluation
        top_doc, score = retrieved[0]
        ret_db, ret_table = parse_doc_db_and_table(top_doc.name)

        db_match = (ret_db == gold_db)
        table_match = (ret_db == gold_db) and (ret_table == gold_table)

        if db_match:
            correct_db += 1
        if table_match:
            correct_table += 1

        # Run TAPAS only on the top-1 retrieved table
        df = top_doc.load_dataframe()
        if args.max_table_rows:
            df = df.head(args.max_table_rows)
        tapas_out = pipeline.answerer.answer(df, ex.question)
        tapas_ans_norm = normalize_answer_for_compare(tapas_out["answer"])

        ans_match = (tapas_ans_norm == gold_ans_norm)
        if ans_match:
            correct_answer += 1

        # Optional: print a few debug lines for the first few examples
        if total <= 5:
            print("=" * 80)
            print(f"Q{total}: {ex.question}")
            print(f"  Gold: db={gold_db}, table={gold_table}")
            print(f"  Gold SQL: {ex.sql_query}")
            print(f"  Gold answer (norm): {gold_ans_norm}")
            print(f"  Retrieved: name={top_doc.name}, score={score:.3f}")
            print(f"  TAPAS answer: {tapas_out['answer']} (norm={tapas_ans_norm})")
            print(f"  DB match: {db_match}, Table match: {table_match}, Answer match: {ans_match}")

    if total == 0:
        print("[WARN] No examples evaluated.")
        return

    print("\n" + "=" * 80)
    print(f"Evaluated {total} single-table Spider dev examples.")
    print(f"Top-1 DB accuracy:      {correct_db / total:.3f}")
    print(f"Top-1 table accuracy:   {correct_table / total:.3f}")
    print(f"TAPAS answer accuracy:  {correct_answer / total:.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()