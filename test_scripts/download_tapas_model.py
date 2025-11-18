import os
from transformers import TapasTokenizer, TapasForQuestionAnswering
import torch
import pandas as pd

def download_tapas_model():
    model_name = "google/tapas-large"
    
    try:
        tokenizer = TapasTokenizer.from_pretrained(model_name)
        
        model = TapasForQuestionAnswering.from_pretrained(model_name)

        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   - Device: {device}")
        
        return model, tokenizer, device
        
    except Exception as e:
        print(f"Error downloading model")
        raise


def test_model(model, tokenizer, device="cpu"):
    
    table = pd.DataFrame({
        "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
        "Number of movies": [87, 53, 69],
        "Date of birth": ["1963-12-18", "1974-11-11", "1961-05-06"]
    })
    table = table.astype(str)
    
    queries = ["How many movies has George Clooney played in?", "When was Brad Pitt born?"]
    
    try:
        for query in queries:
            print(f"\nQuestion: {query}")
            
            inputs = tokenizer(
                table=table,
                queries=query,
                padding="max_length",
                return_tensors="pt"
            )
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            model = model.to(device)
            
            outputs = model(**inputs)
            
            predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
                inputs, outputs.logits.detach(), outputs.logits_aggregation.detach()
            )
            
            id2aggregation = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
            aggregation_predictions_string = [id2aggregation[x] for x in predicted_aggregation_indices]
            
            answers = []
            for coordinates in predicted_answer_coordinates:
                if len(coordinates) == 1:
                    row_idx, col_idx = coordinates[0]
                    col_name = table.columns[col_idx]
                    answers.append(str(table.iloc[row_idx, col_idx]))
                else:
                    cell_values = []
                    for coord in coordinates:
                        row_idx, col_idx = coord
                        col_name = table.columns[col_idx]
                        cell_values.append(str(table.iloc[row_idx, col_idx]))
                    answers.append(", ".join(cell_values))
            
            print(f"   Answer: {answers[0]}")
            print(f"   Aggregation: {aggregation_predictions_string[0]}")
            
    except Exception as e:
        print(f"Error testing model")
        raise

def main():

    model, tokenizer, device = download_tapas_model()
    
    try:
        test_model(model, tokenizer, device)
        print("\n" + "="*60)
        print("Model downloaded and tested successfully!")
        print("="*60)
        print("\nThe model is now cached locally and ready to use.")
        print(f"Cache location: ~/.cache/huggingface/transformers/")
    except Exception as e:
        print("Model Error")


if __name__ == "__main__":
    main()

