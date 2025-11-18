from transformers import TapasTokenizer, TapasForQuestionAnswering
import torch
import pandas as pd

def load_model(use_finetuned=False):
    if use_finetuned:
        model_name = "google/tapas-base-finetuned-wikisql-supervised"
        print(f"Loading fine-tuned model: {model_name}...")
    else:
        model_name = "google/tapas-large"
        print(f"Loading base model: {model_name}...")
        print("   Note: Base model requires fine-tuning for question answering tasks.")
    
    tokenizer = TapasTokenizer.from_pretrained(model_name)
    model = TapasForQuestionAnswering.from_pretrained(model_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    return model, tokenizer, device


def answer_question(model, tokenizer, device, table, question):

    table = table.astype(str)
    
    inputs = tokenizer(
        table=table,
        queries=question,
        padding="max_length",
        return_tensors="pt"
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    try:
        with torch.no_grad():
            outputs = model(**inputs)
    except (ValueError, RuntimeError) as e:
        if "nan" in str(e).lower() or "inf" in str(e).lower() or "constraint" in str(e).lower():
            print("   ⚠ Error: Model produced invalid predictions (NaN/inf values).")
            print("   This indicates the model needs fine-tuning. Using fine-tuned checkpoint recommended.")
            return {
                "answer": None,
                "aggregation": "NONE (Model needs fine-tuning)",
                "coordinates": []
            }
        else:
            raise
    
    if outputs.logits_aggregation is not None:
        inputs_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
            inputs_cpu, outputs.logits.detach().cpu(), outputs.logits_aggregation.detach().cpu()
        )
        id2aggregation = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
        aggregation = id2aggregation[predicted_aggregation_indices[0]]
    else:
        print("   ⚠ Warning: Base model detected. Fine-tuning required for accurate predictions.")
        print("   Consider using a fine-tuned checkpoint like 'google/tapas-base-finetuned-wikisql-supervised'")
        predicted_answer_coordinates = [[]]
        aggregation = "NONE (Model needs fine-tuning)"
    
    answers = []
    for coordinates in predicted_answer_coordinates:
        if len(coordinates) == 1:
            row_idx, col_idx = coordinates[0]
            answers.append(str(table.iloc[row_idx, col_idx]))
        else:
            cell_values = []
            for coord in coordinates:
                row_idx, col_idx = coord
                cell_values.append(str(table.iloc[row_idx, col_idx]))
            answers.append(", ".join(cell_values))
    
    return {
        "answer": answers[0] if answers else None,
        "aggregation": aggregation,
        "coordinates": predicted_answer_coordinates[0] if predicted_answer_coordinates else []
    }


def main():

    use_finetuned = True
    if use_finetuned:
        print("="*60)
        print("Using fine-tuned model for question answering.")
        print("="*60)
    else:
        print("="*60)
        print("Note: Using base model. For actual question answering,")
        print("      set use_finetuned=True in load_model() call.")
        print("="*60)
    model, tokenizer, device = load_model(use_finetuned=use_finetuned)
    
    table = pd.DataFrame({
        "Actor": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney", "Tom Hanks"],
        "Movies": [87, 53, 69, 95],
        "Birth Year": [1963, 1974, 1961, 1956],
        "Oscars": [2, 1, 2, 2]
    })
    table = table.astype(str)
    
    questions = [
        "How many movies has George Clooney played in?",
        "When was Brad Pitt born?",
        "Who has the most movies?",
        "What is the total number of Oscars won by all actors?"
    ]
    
    print("\n" + "="*60)
    print("Table Question Answering Examples")
    print("="*60)
    
    for question in questions:
        print(f"\nQuestion: {question}")
        result = answer_question(model, tokenizer, device, table, question)
        print(f"Answer: {result['answer']}")
        print(f"Aggregation: {result['aggregation']}")


if __name__ == "__main__":
    main()

