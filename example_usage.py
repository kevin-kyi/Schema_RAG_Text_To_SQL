#!/usr/bin/env python3
"""
Example usage of the TAPAS large model for table question answering.
"""

from transformers import TapasTokenizer, TapasForQuestionAnswering
import torch
import pandas as pd

def load_model(use_finetuned=False):
    """
    Load the TAPAS model and tokenizer.
    
    Args:
        use_finetuned: If True, use a fine-tuned checkpoint (smaller but trained for QA).
                      If False, use the base large model (needs fine-tuning for QA).
    """
    if use_finetuned:
        # Use a fine-tuned model (smaller but ready for question answering)
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
    """
    Answer a question about a table using the TAPAS model.
    
    Args:
        model: The TAPAS model
        tokenizer: The TAPAS tokenizer
        device: Device to run inference on
        table: pandas DataFrame with table data (all values must be strings)
        question: String question about the table
    
    Returns:
        Dictionary with answer and metadata
    """
    # Ensure all values are strings (required by TAPAS tokenizer)
    table = table.astype(str)
    
    # Tokenize inputs
    inputs = tokenizer(
        table=table,
        queries=question,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get predictions
    try:
        with torch.no_grad():
            outputs = model(**inputs)
    except (ValueError, RuntimeError) as e:
        # Handle cases where model produces invalid logits (NaN/inf) due to uninitialized weights
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
    
    # Handle case where aggregation logits might be None (base model without fine-tuning)
    if outputs.logits_aggregation is not None:
        # Convert logits to predictions (for fine-tuned models)
        # Move logits and inputs to CPU as convert_logits_to_predictions expects CPU tensors
        inputs_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
            inputs_cpu, outputs.logits.detach().cpu(), outputs.logits_aggregation.detach().cpu()
        )
        # Map aggregation indices to names
        id2aggregation = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
        aggregation = id2aggregation[predicted_aggregation_indices[0]]
    else:
        # Base model without fine-tuning: logits_aggregation is None
        # The base model needs fine-tuning for question answering tasks
        # For now, we'll return a message indicating fine-tuning is needed
        print("   ⚠ Warning: Base model detected. Fine-tuning required for accurate predictions.")
        print("   Consider using a fine-tuned checkpoint like 'google/tapas-base-finetuned-wikisql-supervised'")
        predicted_answer_coordinates = [[]]
        aggregation = "NONE (Model needs fine-tuning)"
    
    # Extract answer from table
    answers = []
    for coordinates in predicted_answer_coordinates:
        if len(coordinates) == 1:
            # Single cell answer
            row_idx, col_idx = coordinates[0]
            answers.append(str(table.iloc[row_idx, col_idx]))
        else:
            # Multiple cells
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
    """Example usage of the TAPAS model."""
    # Load model - set use_finetuned=True to use a fine-tuned checkpoint
    # The fine-tuned model is smaller but ready for question answering
    # Note: Base model (use_finetuned=False) requires fine-tuning and may produce errors
    use_finetuned = True  # Set to False to use base model (requires fine-tuning)
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
    
    # Example table (must be pandas DataFrame with all string values)
    table = pd.DataFrame({
        "Actor": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney", "Tom Hanks"],
        "Movies": [87, 53, 69, 95],
        "Birth Year": [1963, 1974, 1961, 1956],
        "Oscars": [2, 1, 2, 2]
    })
    # Convert all values to strings (required by TAPAS tokenizer)
    table = table.astype(str)
    
    # Example questions
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

