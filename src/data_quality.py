from data_loader import load_training_data


DELINQUENCY_COLUMNS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]


def main() -> None:
    df = load_training_data()

    print("\nAGE CHECK")
    print("=" * 60)
    print(df["age"].value_counts().sort_index().head(10))

    print("\nDELINQUENCY EXTREME VALUES")
    print("=" * 60)

    for column in DELINQUENCY_COLUMNS:
        print(f"\n{column}")

        print(
            df[column]
            .value_counts()
            .sort_index()
            .tail(10)
        )

    print("\nREVOLVING UTILIZATION EXTREMES")
    print("=" * 60)

    print(
        df["RevolvingUtilizationOfUnsecuredLines"]
        .quantile([0.90, 0.95, 0.99, 0.999, 1.0])
    )

    print("\nDEBT RATIO EXTREMES")
    print("=" * 60)

    print(
        df["DebtRatio"]
        .quantile([0.90, 0.95, 0.99, 0.999, 1.0])
    )

    print("\nMONTHLY INCOME EXTREMES")
    print("=" * 60)

    print(
        df["MonthlyIncome"]
        .quantile([0.90, 0.95, 0.99, 0.999, 1.0])
    )

    print("\nSUSPICIOUS DELINQUENCY VALUES")
    print("=" * 60)

    suspicious = df[
        (df["NumberOfTime30-59DaysPastDueNotWorse"] >= 90)
        | (df["NumberOfTimes90DaysLate"] >= 90)
        | (df["NumberOfTime60-89DaysPastDueNotWorse"] >= 90)
    ]

    print(f"Rows with delinquency values >= 90: {len(suspicious):,}")

    print("\nTarget distribution among suspicious rows:")
    print(
        suspicious["SeriousDlqin2yrs"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )


if __name__ == "__main__":
    main()