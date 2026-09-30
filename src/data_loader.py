from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "cs-training.csv.zip"
)

TARGET_COLUMN = "SeriousDlqin2yrs"


def load_training_data() -> pd.DataFrame:
    """Load and perform basic cleaning of the training dataset."""

    df = pd.read_csv(DATA_PATH)

    # Kaggle includes an unnecessary index column.
    unnamed_columns = [
        column for column in df.columns
        if column.startswith("Unnamed")
    ]

    if unnamed_columns:
        df = df.drop(columns=unnamed_columns)

    return df


def main() -> None:
    df = load_training_data()

    print("Dataset loaded successfully.")
    print("=" * 60)

    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    print("\nColumn names:")
    for column in df.columns:
        print(f"- {column}")

    print("\nTarget distribution:")
    print(df[TARGET_COLUMN].value_counts())

    print("\nTarget distribution (%):")
    print(
        df[TARGET_COLUMN]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()