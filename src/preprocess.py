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

DELINQUENCY_SENTINELS = [96, 98]


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply structural cleaning to the raw training dataset.

    Statistical imputation is intentionally not performed here.
    Missing values are handled inside the modelling pipelines to
    prevent information leakage from the test set.
    """

    df = df.copy()

    # Remove observations with impossible age values.
    df = df.loc[df["age"] > 0].copy()

    # Treat anomalous delinquency codes as missing values.
    for column in DELINQUENCY_COLUMNS:
        df.loc[
            df[column].isin(DELINQUENCY_SENTINELS),
            column,
        ] = np.nan

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

    print("Structural preprocessing completed.")
    print("=" * 60)

    print(f"Rows before: {len(df):,}")
    print(f"Rows after:  {len(cleaned_df):,}")

    print("\nMissing values after structural cleaning:")
    missing_values = cleaned_df.isna().sum()
    print(missing_values[missing_values > 0])

    print("\nNo statistical imputation was performed.")
    print("Missing values will be handled inside the modelling pipelines.")

    print(f"\nSaved to:\n{PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()
