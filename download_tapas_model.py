#!/usr/bin/env python3
"""
Script to download and test the TAPAS large model from Hugging Face.
This model is designed for table question answering tasks.
"""

import os
from transformers import TapasTokenizer, TapasForQuestionAnswering
import torch
import pandas as pd

def download_tapas_model():
    """
    Download the TAPAS large model and tokenizer from Hugging Face.
    The model will be cached in ~/.cache/huggingface/transformers/
    """
    print("Downloading TAPAS large model and tokenizer...")
    print("This may take a few minutes depending on your internet connection...")
    
    model_name = "google/tapas-large"
    
    try:
        # Download tokenizer
        print(f"\n1. Downloading tokenizer: {model_name}")
        tokenizer = TapasTokenizer.from_pretrained(model_name)
        print("   ✓ Tokenizer downloaded successfully")
        
        # Download model
        print(f"\n2. Downloading model: {model_name}")
        model = TapasForQuestionAnswering.from_pretrained(model_name)
        print("   ✓ Model downloaded successfully")
        
        # Print model info
        print(f"\n3. Model Information:")
        print(f"   - Model name: {model_name}")
        print(f"   - Model type: {type(model).__name__}")
        print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   - Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   - Device: {device}")
        
        return model, tokenizer, device
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("\nMake sure you have:")
        print("  1. Installed transformers: pip install transformers")
        print("  2. Installed torch: pip install torch")
        print("  3. Internet connection for downloading the model")
        raise


def test_model(model, tokenizer, device="cpu"):
    """
    Test the downloaded model with a simple example.
    """
    print("\n" + "="*60)
    print("Testing the model with a sample table and question...")
    print("="*60)
    
    # Sample table data (must be pandas DataFrame with all string values)
    table = pd.DataFrame({
        "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
        "Number of movies": [87, 53, 69],
        "Date of birth": ["1963-12-18", "1974-11-11", "1961-05-06"]
    })
    # Convert all values to strings (required by TAPAS tokenizer)
    table = table.astype(str)
    
    # Sample question
    queries = ["How many movies has George Clooney played in?", "When was Brad Pitt born?"]
    
    try:
        for query in queries:
            print(f"\nQuestion: {query}")
            
            # Tokenize inputs
            inputs = tokenizer(
                table=table,
                queries=query,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            model = model.to(device)
            
            # Get predictions
            outputs = model(**inputs)
            
            # Process predictions
            predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
                inputs, outputs.logits.detach(), outputs.logits_aggregation.detach()
            )
            
            # Get the answer
            id2aggregation = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
            aggregation_predictions_string = [id2aggregation[x] for x in predicted_aggregation_indices]
            
            answers = []
            for coordinates in predicted_answer_coordinates:
                if len(coordinates) == 1:
                    # Only one cell
                    row_idx, col_idx = coordinates[0]
                    col_name = table.columns[col_idx]
                    answers.append(str(table.iloc[row_idx, col_idx]))
                else:
                    # Multiple cells
                    cell_values = []
                    for coord in coordinates:
                        row_idx, col_idx = coord
                        col_name = table.columns[col_idx]
                        cell_values.append(str(table.iloc[row_idx, col_idx]))
                    answers.append(", ".join(cell_values))
            
            print(f"   Answer: {answers[0]}")
            print(f"   Aggregation: {aggregation_predictions_string[0]}")
            
    except Exception as e:
        print(f"\n❌ Error testing model: {e}")
        print("Note: This is a basic test. For full functionality, you may need to fine-tune the model.")
        raise


def main():
    """
    Main function to download and test the TAPAS model.
    """
    print("="*60)
    print("TAPAS Large Model Downloader")
    print("="*60)
    
    # Download model
    model, tokenizer, device = download_tapas_model()
    
    # Test model
    try:
        test_model(model, tokenizer, device)
        print("\n" + "="*60)
        print("✓ Model downloaded and tested successfully!")
        print("="*60)
        print("\nThe model is now cached locally and ready to use.")
        print(f"Cache location: ~/.cache/huggingface/transformers/")
    except Exception as e:
        print(f"\n⚠ Warning: Model downloaded but test failed: {e}")
        print("The model is still available for use, but may need fine-tuning.")


if __name__ == "__main__":
    main()

