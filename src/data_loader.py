
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


def load_training_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the raw training dataset.

    Remove CSV export-index columns and validate
    the basic structure of the dataset.
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"Training dataset not found: {path}\n"
            "Check that cs-training.csv.zip is available "
            "in data/raw/."
        )

    df = pd.read_csv(path)

    # Remove index columns generated during CSV export.
    unnamed_columns = [
        column
        for column in df.columns
        if column.startswith("Unnamed:")
    ]

    if unnamed_columns:
        df = df.drop(columns=unnamed_columns)

    if df.empty:
        raise ValueError(
            "The training dataset is empty."
        )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            "not found in the dataset."
        )

    return df


def main() -> None:
    df = load_training_data()

    print("\nDATASET OVERVIEW")
    print("=" * 60)

    print(f"Rows:    {df.shape[0]:,}")
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
