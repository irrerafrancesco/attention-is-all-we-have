from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import load_training_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_clean.csv"
)

DELINQUENCY_COLUMNS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw training dataset."""

    df = df.copy()

    # Remove impossible age values.
    df = df[df["age"] > 0].copy()

    # Replace sentinel delinquency values with missing values.
    for column in DELINQUENCY_COLUMNS:
        df.loc[df[column] >= 90, column] = np.nan

    # Median imputation for missing values.
    median_columns = [
        "MonthlyIncome",
        "NumberOfDependents",
        *DELINQUENCY_COLUMNS,
    ]

    for column in median_columns:
        df[column] = df[column].fillna(df[column].median())

    return df


def main() -> None:
    df = load_training_data()
    cleaned_df = preprocess_data(df)

    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleaned_df.to_csv(
        PROCESSED_DATA_PATH,
        index=False,
    )

    print("Preprocessing completed.")
    print("=" * 60)

    print(f"Rows before: {len(df):,}")
    print(f"Rows after:  {len(cleaned_df):,}")

    print("\nMissing values after preprocessing:")
    print(cleaned_df.isna().sum())

    print(f"\nSaved to:")
    print(PROCESSED_DATA_PATH)


if __name__ == "__main__":
    main()