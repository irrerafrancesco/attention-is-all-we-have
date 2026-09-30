import math

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_loader import load_training_data


TARGET = "SeriousDlqin2yrs"

DELINQUENCY_COLUMNS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]


def prepare_data():
    """Prepare features and target without leaking test information."""

    df = load_training_data().copy()

    # Remove the single impossible age value.
    df = df[df["age"] > 0].copy()

    # Convert anomalous delinquency codes into missing values.
    for column in DELINQUENCY_COLUMNS:
        df.loc[df[column] >= 90, column] = np.nan

    X = df.drop(columns=TARGET)
    y = df[TARGET]

    return X, y


def evaluate_review_capacity(y_true, probabilities, capacities):
    """
    Measure how many future problem cases are captured when analysts
    can review only a limited percentage of customers.
    """

    results = []

    ranked = (
        y_true
        .to_frame("actual")
        .assign(probability=probabilities)
        .sort_values("probability", ascending=False)
    )

    total_problem_cases = ranked["actual"].sum()

    for capacity in capacities:
        review_count = math.ceil(len(ranked) * capacity)

        review_queue = ranked.head(review_count)

        captured_cases = review_queue["actual"].sum()

        capture_rate = captured_cases / total_problem_cases
        queue_precision = captured_cases / review_count

        results.append(
            (
                capacity,
                review_count,
                captured_cases,
                capture_rate,
                queue_precision,
            )
        )

    return results


def main():
    X, y = prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    numeric_features = X.columns.tolist()

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    print("Training Logistic Regression baseline...")

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)

    print("\nBASELINE MODEL")
    print("=" * 60)

    print(f"Training customers: {len(X_train):,}")
    print(f"Test customers:     {len(X_test):,}")

    print(f"\nROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")

    print("\nATTENTION BUDGET")
    print("=" * 60)

    results = evaluate_review_capacity(
        y_test,
        probabilities,
        capacities=[0.05, 0.10, 0.20, 0.30],
    )

    print(
        f"{'Capacity':<12}"
        f"{'Reviews':>10}"
        f"{'Problems':>12}"
        f"{'Captured':>12}"
        f"{'Queue risk':>12}"
    )

    print("-" * 58)

    for (
        capacity,
        review_count,
        captured_cases,
        capture_rate,
        queue_precision,
    ) in results:

        print(
            f"{capacity:>10.0%}"
            f"{review_count:>10,}"
            f"{captured_cases:>12,}"
            f"{capture_rate:>12.2%}"
            f"{queue_precision:>12.2%}"
        )


if __name__ == "__main__":
    main()