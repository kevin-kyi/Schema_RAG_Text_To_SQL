from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import faiss
import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    TapasForQuestionAnswering,
    TapasTokenizer,
)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TAPAS_MODEL_NAME = "google/tapas-large-finetuned-wtq"


@dataclass
class TableDocument:
    name: str
    path: Path | None
    schema_text: str
    columns: List[str]
    n_rows: int
    loader: Callable[[], pd.DataFrame] | None = None
    source: str = "csv"

    def load_dataframe(self) -> pd.DataFrame:
        if self.loader:
            return self.loader()
        if self.path and self.path.suffix.lower() == ".csv":
            return pd.read_csv(self.path)
        raise ValueError(f"No loader defined for document: {self.name}")


def describe_table_schema(
    name: str,
    df: pd.DataFrame,
    sample_rows: int = 2,
    total_rows: int | None = None,
    column_type_hints: Dict[str, str] | None = None,
) -> str:
    dtype_items = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        if column_type_hints and col in column_type_hints:
            dtype = f"{column_type_hints[col]}|{dtype}"
        dtype_items.append(f"{col} ({dtype})")

    preview_rows = df.head(sample_rows).astype(str)
    if preview_rows.empty:
        sample_text = "Table has no rows."
    else:
        rendered_rows = []
        for record in preview_rows.to_dict(orient="records"):
            rendered = ", ".join(f"{k}={v}" for k, v in record.items())
            rendered_rows.append(rendered)
        sample_text = " | ".join(rendered_rows)

    row_count = total_rows if total_rows is not None else len(df)
    return (
        f"Table {name} with {len(df.columns)} columns and {row_count} rows. "
        f"Columns: {', '.join(dtype_items)}. Sample rows: {sample_text}"
    )


def load_table_documents(data_dir: Path | None, sample_rows: int = 2) -> List[TableDocument]:
    if data_dir is None:
        return []

    data_dir = Path(data_dir)
    csv_paths = sorted(data_dir.glob("*.csv"))
    documents: List[TableDocument] = []

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        schema_text = describe_table_schema(csv_path.stem, df, sample_rows=sample_rows)
        documents.append(
            TableDocument(
                name=csv_path.stem,
                path=csv_path.resolve(),
                schema_text=schema_text,
                columns=list(df.columns),
                n_rows=len(df),
                source="csv",
            )
        )

    if not documents:
        raise ValueError(f"No CSV tables found in directory: {data_dir}")

    return documents

def _escape_identifier(identifier: str) -> str:
    return identifier.replace('"', '""')

def _make_sqlite_table_loader(db_path: Path, table_name: str) -> Callable[[], pd.DataFrame]:
    escaped_name = _escape_identifier(table_name)

    def loader() -> pd.DataFrame:
        with sqlite3.connect(db_path) as conn:
            query = f'SELECT * FROM "{escaped_name}"'
            return pd.read_sql_query(query, conn)

    return loader


def load_spider_table_documents(
    spider_dir: Path | None,
    sample_rows: int = 2,
) -> List[TableDocument]:
    if spider_dir is None:
        return []

    spider_dir = Path(spider_dir)
    tables_path = spider_dir / "tables.json"
    database_dir = spider_dir / "database"

    if not tables_path.exists():
        raise FileNotFoundError(f"Could not locate Spider tables.json at {tables_path}")

    with tables_path.open() as f:
        schema_entries = json.load(f)

    documents: List[TableDocument] = []

    for entry in schema_entries:
        db_id = entry["db_id"]
        table_names = entry.get("table_names_original") or entry["table_names"]
        column_names = entry.get("column_names_original") or entry["column_names"]
        column_types = entry["column_types"]
        db_path = database_dir / db_id / f"{db_id}.sqlite"

        if not db_path.exists():
            continue

        columns_by_table: Dict[int, List[Tuple[str, str]]] = {idx: [] for idx in range(len(table_names))}
        for (table_idx, column_name), column_type in zip(column_names, column_types):
            if table_idx < 0:
                continue
            columns_by_table.setdefault(table_idx, []).append((column_name, column_type))

        with sqlite3.connect(db_path) as conn:
            for table_idx, table_name in enumerate(table_names):
                column_items = columns_by_table.get(table_idx, [])
                column_names_only = [name for name, _ in column_items]
                column_type_hints = {name: dtype for name, dtype in column_items}
                escaped_name = _escape_identifier(table_name)

                try:
                    preview_df = pd.read_sql_query(
                        f'SELECT * FROM "{escaped_name}" LIMIT {max(sample_rows, 1)}',
                        conn,
                    )
                except Exception:
                    preview_df = pd.DataFrame(columns=column_names_only)

                try:
                    total_rows = conn.execute(
                        f'SELECT COUNT(*) FROM "{escaped_name}"'
                    ).fetchone()[0]
                except Exception:
                    total_rows = 0

                schema_text = describe_table_schema(
                    f"{db_id}.{table_name}",
                    preview_df,
                    sample_rows=sample_rows,
                    total_rows=total_rows,
                    column_type_hints=column_type_hints,
                )

                documents.append(
                    TableDocument(
                        name=f"{db_id}.{table_name}",
                        path=db_path,
                        schema_text=schema_text,
                        columns=column_names_only,
                        n_rows=total_rows,
                        loader=_make_sqlite_table_loader(db_path, table_name),
                        source="spider",
                    )
                )

    if not documents:
        raise ValueError(f"No Spider tables found inside directory: {spider_dir}")

    return documents


class SchemaEncoder:
    def __init__(self, model_name: str = EMBED_MODEL_NAME, device: str | None = None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: Sequence[str], batch_size: int = 16) -> np.ndarray:
        vectors: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            outputs = self.model(**encoded)

            token_embeddings = outputs.last_hidden_state
            input_mask = encoded["attention_mask"].unsqueeze(-1)
            summed = torch.sum(token_embeddings * input_mask, dim=1)
            counts = torch.clamp(input_mask.sum(dim=1), min=1e-9)
            mean_pooled = summed / counts
            normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)

            vectors.append(normalized.cpu().numpy().astype("float32"))

        return np.vstack(vectors)


class SchemaVectorStore:
    def __init__(self, encoder: SchemaEncoder, documents: Sequence[TableDocument]):
        self.encoder = encoder
        self.documents = list(documents)
        self.index: faiss.Index | None = None
        self.dim: int | None = None
        self._build_index()

    def _build_index(self) -> None:
        texts = [doc.schema_text for doc in self.documents]
        embeddings = self.encoder.encode(texts)
        dim = embeddings.shape[1]

        index = faiss.IndexFlatIP(dim)
        index.add(np.ascontiguousarray(embeddings))

        self.index = index
        self.dim = dim

    def search(self, query: str, top_k: int = 3) -> List[Tuple[TableDocument, float]]:
        if not self.index:
            raise RuntimeError("Vector index has not been built yet.")

        k = min(top_k, len(self.documents))
        query_vec = self.encoder.encode([query])
        scores, indices = self.index.search(query_vec, k)

        results: List[Tuple[TableDocument, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc = self.documents[idx]
            results.append((doc, float(score)))

        return results


class TapasAnswerer:
    def __init__(self, model_name: str = TAPAS_MODEL_NAME, device: str | None = None):
        self.tokenizer = TapasTokenizer.from_pretrained(model_name)
        self.model = TapasForQuestionAnswering.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def answer(self, table: pd.DataFrame, question: str) -> dict:
        table = table.astype(str)
        inputs = self.tokenizer(
            table=table,
            queries=[question],
            padding="max_length",
            return_tensors="pt",
        )

        inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs_on_device)

        inputs_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        logits = outputs.logits.detach().cpu()
        agg_logits = (
            outputs.logits_aggregation.detach().cpu()
            if outputs.logits_aggregation is not None
            else None
        )

        if agg_logits is None:
            predicted_coords = [[]]
            predicted_aggs = [0]
        else:
            predicted_coords, predicted_aggs = self.tokenizer.convert_logits_to_predictions(
                inputs_cpu,
                logits,
                agg_logits,
            )

        coords = predicted_coords[0] if predicted_coords else []
        agg_idx = predicted_aggs[0] if predicted_aggs else 0

        if self.model.config.aggregation_labels:
            id2agg = self.model.config.aggregation_labels
        else:
            id2agg = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}

        aggregation = id2agg.get(agg_idx, "UNKNOWN")

        selected_cells = []
        for row_idx, col_idx in coords:
            value = table.iat[row_idx, col_idx]
            selected_cells.append(str(value))

        if not coords:
            answer_value = "<no cells selected>"
        elif aggregation == "NONE":
            answer_value = ", ".join(selected_cells)
        else:
            try:
                numeric_cells = [float(v) for v in selected_cells]
            except ValueError:
                numeric_cells = None

            if numeric_cells is None:
                answer_value = ", ".join(selected_cells)
            else:
                if aggregation == "SUM":
                    answer_value = str(sum(numeric_cells))
                elif aggregation == "AVERAGE":
                    answer_value = str(sum(numeric_cells) / len(numeric_cells))
                elif aggregation == "COUNT":
                    answer_value = str(len(numeric_cells))
                else:
                    answer_value = ", ".join(selected_cells)

        return {
            "question": question,
            "answer": answer_value,
            "aggregation": aggregation,
            "coordinates": coords,
            "selected_cells": selected_cells,
        }


class SchemaRAGPipeline:
    def __init__(
        self,
        data_dir: Path | None = None,
        spider_dir: Path | None = None,
        embedding_model: str = EMBED_MODEL_NAME,
        tapas_model: str = TAPAS_MODEL_NAME,
        max_table_rows: int = 64,
    ):
        self.data_dir = Path(data_dir) if data_dir else None
        self.spider_dir = Path(spider_dir) if spider_dir else None
        self.max_table_rows = max_table_rows

        documents: List[TableDocument] = []
        documents.extend(load_table_documents(self.data_dir))
        documents.extend(load_spider_table_documents(self.spider_dir))

        if not documents:
            raise ValueError("No tables available for retrieval.")

        self.encoder = SchemaEncoder(model_name=embedding_model)
        self.vector_store = SchemaVectorStore(self.encoder, documents)
        self.answerer = TapasAnswerer(model_name=tapas_model)

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[TableDocument, float]]:
        return self.vector_store.search(query, top_k=top_k)

    def answer(self, query: str, top_k: int = 1) -> List[dict]:
        results = []
        for doc, score in self.retrieve(query, top_k=top_k):
            df = doc.load_dataframe()
            if self.max_table_rows:
                df = df.head(self.max_table_rows)

            answer_payload = self.answerer.answer(df, query)
            results.append(
                {
                    "table_name": doc.name,
                    "table_path": str(doc.path) if doc.path else "<in-memory>",
                    "source": doc.source,
                    "retrieval_score": score,
                    "schema": doc.schema_text,
                    "answer": answer_payload,
                }
            )

        return results


__all__ = [
    "SchemaEncoder",
    "SchemaRAGPipeline",
    "SchemaVectorStore",
    "TableDocument",
    "TapasAnswerer",
    "load_table_documents",
    "load_spider_table_documents",
]
