from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from baseline_model import (
    evaluate_review_capacity,
    prepare_data,
)


def main():
    X, y = prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
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

    print("Training Gradient Boosting model...")

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)

    print("\nBOOSTED MODEL")
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