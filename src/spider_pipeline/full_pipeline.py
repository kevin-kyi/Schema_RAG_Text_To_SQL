from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

from schema_rag import EMBED_MODEL_NAME, TAPAS_MODEL_NAME, SchemaRAGPipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve the most relevant table schema with FAISS and answer the query via TAPAS."
        )
    )
    default_data_dir = (Path(__file__).parent / ".." / "data").resolve()

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir,
        help=f"Directory containing CSV tables (default: {default_data_dir})",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Natural language question to answer.",
    )
    parser.add_argument(
        "--spider-dir",
        type=Path,
        default=None,
        help=(
            "Optional Spider dataset directory (expects tables.json + database/). "
            "When provided, every Spider table is indexed alongside local CSVs."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of tables to retrieve and run through TAPAS.",
    )
    parser.add_argument(
        "--max-table-rows",
        type=int,
        default=64,
        help="Limit table rows passed to TAPAS to control sequence length.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=EMBED_MODEL_NAME,
        help=f"Hugging Face model used to encode schemas (default: {EMBED_MODEL_NAME})",
    )
    parser.add_argument(
        "--tapas-model",
        type=str,
        default=TAPAS_MODEL_NAME,
        help=f"TAPAS checkpoint to use for QA (default: {TAPAS_MODEL_NAME})",
    )

    return parser


def render_result_block(result: dict, idx: int) -> str:
    wrapper = textwrap.TextWrapper(
        width=100,
        initial_indent="    ",
        subsequent_indent="    ",
    )
    schema_wrapped = wrapper.fill(f"Schema: {result['schema']}")

    answer = result["answer"]
    coords = ", ".join(str(coord) for coord in answer["coordinates"])
    selected_cells = ", ".join(answer["selected_cells"]) or "<none>"

    block = [
        f"[{idx}] Table: {result['table_name']} (source={result['source']}, "
        f"score={result['retrieval_score']:.3f})",
        f"    Path: {result['table_path']}",
        schema_wrapped,
        f"    TAPAS answer: {answer['answer']}",
        f"    Aggregation: {answer['aggregation']}",
        f"    Selected coordinates: {coords or '<none>'}",
        f"    Selected cells: {selected_cells}",
    ]

    return "\n".join(block)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    pipeline = SchemaRAGPipeline(
        data_dir=args.data_dir,
        spider_dir=args.spider_dir,
        embedding_model=args.embedding_model,
        tapas_model=args.tapas_model,
        max_table_rows=args.max_table_rows,
    )

    results = pipeline.answer(args.query, top_k=args.top_k)

    if not results:
        print("No tables were retrieved for the provided query.")
        return

    print("=" * 80)
    print(f"Query: {args.query}")
    print("=" * 80)

    for idx, result in enumerate(results, start=1):
        print(render_result_block(result, idx))
        if idx < len(results):
            print("-" * 80)


if __name__ == "__main__":
    main()
