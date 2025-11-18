import seaborn as sns
import pandas as pd
from pathlib import Path

TABLE_SPECS = {
    "tips": {
        "columns": ["total_bill", "tip", "sex", "smoker", "day", "time", "size"],
        "n_rows": 80,
    },
    "titanic": {
        "columns": ["pclass", "sex", "age", "sibsp", "parch", "fare", "survived"],
        "n_rows": 200,
    },
    "penguins": {
        "columns": [
            "species",
            "island",
            "sex",
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
        ],
        "n_rows": 200,
    },
    "flights": {
        "columns": ["year", "month", "passengers"],
        "n_rows": 200,
    },
    "iris": {
        "columns": [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
            "species",
        ],
        "n_rows": 150,
    },
}


def main():
    script_dir = Path(__file__).parent

    out_dir = (script_dir / ".." / "data").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving tables to: {out_dir}")

    for name, spec in TABLE_SPECS.items():
        print(f"\n=== Preparing table: {name} ===")

        df = sns.load_dataset(name)

        cols = [c for c in spec["columns"] if c in df.columns]
        df = df[cols]

        df = df.dropna()

        n_rows = spec.get("n_rows", len(df))
        df = df.head(n_rows)

        out_path = out_dir / f"{name}.csv"
        df.to_csv(out_path, index=False)

        print(f"   Saved {len(df)} rows × {len(df.columns)} columns → {out_path}")


if __name__ == "__main__":
    main()