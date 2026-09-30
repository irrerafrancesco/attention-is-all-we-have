from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from baseline_model import (
    evaluate_review_capacity,
    prepare_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "model_comparison.csv"
)

CAPACITIES = [0.05, 0.10, 0.20, 0.30]


def build_logistic_model(features):
    """Build the Logistic Regression baseline pipeline."""

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
                features,
            )
        ]
    )

    return Pipeline(
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


def build_boosted_model():
    """Build the Gradient Boosting pipeline."""

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    learning_rate=0.08,
                    max_iter=200,
                    max_leaf_nodes=31,
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Train and evaluate a model using both ML and attention-budget metrics."""

    print(f"Training {name}...")

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)

    attention_results = evaluate_review_capacity(
        y_test,
        probabilities,
        CAPACITIES,
    )

    result = {
        "model": name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }

    for (
        capacity,
        review_count,
        captured_cases,
        capture_rate,
        queue_precision,
    ) in attention_results:

        percentage = int(capacity * 100)

        result[f"capture_at_{percentage}pct"] = capture_rate
        result[f"queue_risk_at_{percentage}pct"] = queue_precision

    return result


def main():
    X, y = prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = [
        (
            "Logistic Regression",
            build_logistic_model(X.columns.tolist()),
        ),
        (
            "Gradient Boosting",
            build_boosted_model(),
        ),
    ]

    results = []

    for name, model in models:
        result = evaluate_model(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        results.append(result)

    comparison = pd.DataFrame(results)

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        REPORT_PATH,
        index=False,
    )

    print("\nMODEL COMPARISON")
    print("=" * 90)

    display_columns = [
        "model",
        "roc_auc",
        "pr_auc",
        "capture_at_5pct",
        "capture_at_10pct",
        "capture_at_20pct",
        "capture_at_30pct",
    ]

    printable = comparison[display_columns].copy()

    percentage_columns = [
        "capture_at_5pct",
        "capture_at_10pct",
        "capture_at_20pct",
        "capture_at_30pct",
    ]

    for column in percentage_columns:
        printable[column] = (
            printable[column]
            .mul(100)
            .round(2)
            .astype(str)
            + "%"
        )

    print(printable.to_string(index=False))

    print(f"\nResults saved to:")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()