# Schema_RAG_Text_To_SQL

## Schema-aware RAG pipeline

`src/schema_rag.py` a small retrieval stack that:

- loads every CSV inside `data/` and writes a short schema description (table name, column + dtype pairs and a couple of sample rows),
- encodes those schema strings with the `sentence-transformers/all-MiniLM-L6-v2` encoder,
- stores the vectors inside a FAISS `IndexFlatIP` so we can run fast cosine-similarity lookups at inference time,
- and feeds the highest scoring table(s) to a TAPAS question-answering model.

`src/full_pipeline.py` is the CLI entry point:

```bash
pip install -r requirements.txt
python src/full_pipeline.py \
    --query "Which actors have won the most Oscars?" \
    --top-k 2
```

### Spider dataset integration

download spider dataset: https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view unzip and place under data/
```bash
python src/full_pipeline.py \
    --query "List colleges with the most students." \
    --spider-dir data/spider_data/spider_data \
    --top-k 3
```

The pipeline will parse every Spider table, encode its schema description, and feed the retrieved table directly to TAPAS—skipping the original SQL ground-truth step and keeping everything in the TAPAS QA domain.

Flags:

- `--data-dir`: where to scan for CSV tables (defaults to `data/`).
- `--spider-dir`: optional Spider dataset root (indexes every SQLite table inside).
- `--embedding-model`: Hugging Face checkpoint used to encode schemas.
- `--tapas-model`: TAPAS checkpoint used for QA (`google/tapas-large-finetuned-wtq` by default).
- `--max-table-rows`: limit the number of rows sent to TAPAS (keeps sequences within limits).

The script prints the retrieved tables, the schema summary that was embedded, cosine similarity scores, and the TAPAS predictions (aggregation, selected cells, etc.).
