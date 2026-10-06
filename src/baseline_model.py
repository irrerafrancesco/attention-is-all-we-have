import math

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
from preprocess import preprocess_data


TARGET = "SeriousDlqin2yrs"

RANDOM_STATE = 42
TEST_SIZE = 0.20

REVIEW_CAPACITIES = (
    0.05,
    0.10,
    0.20,
    0.30,
)


def prepare_data():
    """
    Load and structurally clean the dataset.

    Missing values are intentionally preserved here because statistical
    imputation is performed inside the modelling pipeline.
    """

    df = load_training_data()
    df = preprocess_data(df)

    X = df.drop(columns=TARGET)
    y = df[TARGET]

    return X, y


def split_data(X, y):
    """Create a reproducible stratified train/test split."""

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def build_logistic_model():
    """
    Build the Logistic Regression baseline.

    Imputation and scaling are fitted only on the training data,
    preventing information leakage from the test set.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_review_capacity(
    y_true,
    probabilities,
    capacities,
):
    """
    Evaluate how many future problem cases are captured when only a
    limited percentage of customers can be reviewed.

    Customers are ranked from highest to lowest predicted probability.
    """

    if len(y_true) != len(probabilities):
        raise ValueError(
            "y_true and probabilities must have the same length."
        )

    ranked = (
        y_true
        .to_frame("actual")
        .assign(probability=probabilities)
        .sort_values(
            "probability",
            ascending=False,
        )
    )

    total_problem_cases = int(ranked["actual"].sum())

    if total_problem_cases == 0:
        raise ValueError(
            "Review-capacity evaluation requires at least one positive case."
        )

    results = []

    for capacity in capacities:

        if not 0 < capacity <= 1:
            raise ValueError(
                "Each review capacity must be between 0 and 1."
            )

        review_count = math.ceil(
            len(ranked) * capacity
        )

        review_queue = ranked.head(review_count)

        captured_cases = int(
            review_queue["actual"].sum()
        )

        capture_rate = (
            captured_cases / total_problem_cases
        )

        queue_precision = (
            captured_cases / review_count
        )

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

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
    )

    model = build_logistic_model()

    print("Training Logistic Regression baseline...")

    model.fit(
        X_train,
        y_train,
    )

    probabilities = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

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
        capacities=REVIEW_CAPACITIES,
    )

    print(
        f"{'Capacity':<12}"
        f"{'Reviews':>10}"
        f"{'Problems':>12}"
        f"{'Captured':>12}"
        f"{'Precision':>12}"
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
