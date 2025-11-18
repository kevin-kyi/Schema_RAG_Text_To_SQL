from pathlib import Path

import torch
import pandas as pd
from transformers import TapasTokenizer, TapasForQuestionAnswering


MODEL_NAME = "google/tapas-large-finetuned-wtq"


def load_model():
    print("=" * 80)
    print(f"Loading TAPAS model: {MODEL_NAME}")
    print("=" * 80)

    tokenizer = TapasTokenizer.from_pretrained(MODEL_NAME)
    model = TapasForQuestionAnswering.from_pretrained(MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print(f"Model loaded on device: {device}")
    print("Note: TAPAS-large is ~300M parameters; CPU is fine for these tiny tables.\n")

    return model, tokenizer, device


def load_table():
    script_dir = Path(__file__).parent
    data_path = (script_dir / ".." / "data" / "actors.csv").resolve()

    if not data_path.exists():
        raise FileNotFoundError(f"Could not find actors.csv at: {data_path}")

    print(f"Loading table from: {data_path}")
    df = pd.read_csv(data_path)
    df_str = df.astype(str)

    print(f"Loaded table with {len(df_str)} rows and {len(df_str.columns)} columns.")
    print("Columns:", list(df_str.columns))
    print()
    return df_str


def answer_question(model, tokenizer, device, table, question):

    inputs = tokenizer(
        table=table,
        queries=[question],
        padding="max_length",
        return_tensors="pt",
    )

    inputs_on_device = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs_on_device)

    predicted_coords, predicted_agg_indices = tokenizer.convert_logits_to_predictions(
        inputs,
        outputs.logits.detach().cpu(),
        outputs.logits_aggregation.detach().cpu(),
    )

    coords = predicted_coords[0] if predicted_coords else []
    agg_idx = predicted_agg_indices[0] if len(predicted_agg_indices) > 0 else 0

    if model.config.aggregation_labels:
        id2agg = model.config.aggregation_labels
        aggregation = id2agg.get(agg_idx, "UNKNOWN")
    else:
        id2agg_fallback = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
        aggregation = id2agg_fallback.get(agg_idx, "UNKNOWN")

    selected_cells = []
    for (row_idx, col_idx) in coords:
        value = table.iat[row_idx, col_idx]
        selected_cells.append(str(value))

    answer_str = ""
    if not coords:
        answer_str = "<no cells selected>"
    else:
        if aggregation == "NONE":
            answer_str = ", ".join(selected_cells)
        else:
            try:
                nums = [float(v) for v in selected_cells]
                if aggregation == "SUM":
                    value = sum(nums)
                elif aggregation == "AVERAGE":
                    value = sum(nums) / len(nums)
                elif aggregation == "COUNT":
                    value = len(nums)
                else:
                    value = ", ".join(selected_cells)
                answer_str = str(value)
            except ValueError:
                answer_str = ", ".join(selected_cells)

    return {
        "question": question,
        "answer": answer_str,
        "aggregation": aggregation,
        "coordinates": coords,
        "selected_cells": selected_cells,
    }


def main():
    model, tokenizer, device = load_model()
    table = load_table()

    questions = [
        "How many movies has George Clooney appeared in?",
        "When was Natalie Portman born?",
        "Which actor has the most movies?",
        "What is the total number of Oscars won by all actors?",
        "Which action-genre actor has the highest number of movies?",
    ]

    print("=" * 80)
    print("Running TAPAS on actors table")
    print("=" * 80)

    for q in questions:
        result = answer_question(model, tokenizer, device, table, q)

        print(f"\nQuestion: {result['question']}")
        print(f"Predicted answer: {result['answer']}")
        print(f"Aggregation: {result['aggregation']}")
        print(f"Selected cell coordinates: {result['coordinates']}")
        print(f"Selected cell values: {result['selected_cells']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
