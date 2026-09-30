from data_loader import load_training_data


def main() -> None:
    df = load_training_data()

    print("\nDATASET OVERVIEW")
    print("=" * 70)
    df.info()

    print("\nSUMMARY STATISTICS")
    print("=" * 70)
    print(df.describe().T)

    print("\nTARGET DISTRIBUTION")
    print("=" * 70)
    print(df["SeriousDlqin2yrs"].value_counts())

    print("\nTARGET DISTRIBUTION (%)")
    print("=" * 70)
    print(
        df["SeriousDlqin2yrs"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\nMISSING VALUES")
    print("=" * 70)

    missing = df.isna().sum().to_frame("missing_count")

    missing["missing_percent"] = (
        missing["missing_count"] / len(df) * 100
    ).round(2)

    print(
        missing[
            missing["missing_count"] > 0
        ].sort_values(
            "missing_percent",
            ascending=False,
        )
    )

    print("\nUNIQUE VALUES")
    print("=" * 70)

    for column in df.columns:
        print(
            f"{column:<45} "
            f"{df[column].nunique(dropna=False):>10,}"
        )

    print("\nMINIMUM AND MAXIMUM VALUES")
    print("=" * 70)

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:
        print(
            f"{column:<45} "
            f"min={df[column].min():>12.4f}   "
            f"max={df[column].max():>12.4f}"
        )


if __name__ == "__main__":
    main()