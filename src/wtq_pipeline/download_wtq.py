# Ensure datasets==2.20.0 installed in environment
from pathlib import Path
from datasets import load_dataset

DATA_DIR = Path("data/wtq_hf")

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading WTQ dataset (stanfordnlp/wikitablequestions, random-split-1)...")
    wtq = load_dataset("stanfordnlp/wikitablequestions", "random-split-1")

    for split in ["train", "validation", "test"]:
        print(f"Saving {split} split to disk...")
        wtq[split].to_json(DATA_DIR / f"wtq_{split}.jsonl")

    print("Done. Saved WTQ splits under", DATA_DIR)

if __name__ == "__main__":
    main()