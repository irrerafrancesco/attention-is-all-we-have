from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from baseline_model import (
    REVIEW_CAPACITIES,
    build_logistic_model,
    evaluate_review_capacity,
    prepare_data,
    split_data,
)
from boosted_model import build_boosted_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "model_comparison.csv"
)


def evaluate_model(
    name,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
):
    """
    Train and evaluate a model using both predictive and
    attention-budget metrics.
    """

    print(f"Training {name}...")

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

    attention_results = evaluate_review_capacity(
        y_test,
        probabilities,
        REVIEW_CAPACITIES,
    )

    result = {
        "model": name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }

    for (
        capacity,
        _review_count,
        _captured_cases,
        capture_rate,
        queue_precision,
    ) in attention_results:

        percentage = int(capacity * 100)

        result[
            f"capture_at_{percentage}pct"
        ] = capture_rate

        result[
            f"precision_at_{percentage}pct"
        ] = queue_precision

    return result


def main():
    X, y = prepare_data()

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
    )

    models = [
        (
            "Logistic Regression",
            build_logistic_model(),
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

    capture_columns = [
        f"capture_at_{int(capacity * 100)}pct"
        for capacity in REVIEW_CAPACITIES
    ]

    display_columns = [
        "model",
        "roc_auc",
        "pr_auc",
        *capture_columns,
    ]

    printable = comparison[
        display_columns
    ].copy()

    for column in capture_columns:
        printable[column] = (
            printable[column]
            .mul(100)
            .round(2)
            .astype(str)
            + "%"
        )

    print(
        printable.to_string(
            index=False,
        )
    )

    print(f"\nResults saved to:\n{REPORT_PATH}")


if __name__ == "__main__":
    main()
