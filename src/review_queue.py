import math
from pathlib import Path

import pandas as pd

from baseline_model import (
    evaluate_review_capacity,
    prepare_data,
    split_data,
)
from boosted_model import build_boosted_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "review_queue.csv"
)

REVIEW_CAPACITY = 0.20


def build_review_queue(
    X_test,
    y_test,
    probabilities,
    capacity=REVIEW_CAPACITY,
):
    """
    Build a priority queue from predicted probabilities.

    Customers are ranked from highest to lowest predicted risk.
    Only the number of customers allowed by the review capacity
    is returned.
    """

    if len(X_test) != len(y_test):
        raise ValueError(
            "X_test and y_test must have the same length."
        )

    if len(X_test) != len(probabilities):
        raise ValueError(
            "X_test and probabilities must have the same length."
        )

    if not 0 < capacity <= 1:
        raise ValueError(
            "Review capacity must be between 0 and 1."
        )

    scored_customers = X_test.copy()

    # Preserve a stable identifier derived from the original dataset index.
    scored_customers.insert(
        0,
        "customer_id",
        X_test.index + 1,
    )

    scored_customers["predicted_risk"] = probabilities

    # The true outcome is available only because this is historical
    # validation data. It would not be known in a live scoring setting.
    scored_customers["actual_outcome"] = y_test.values

    scored_customers = (
        scored_customers
        .sort_values(
            "predicted_risk",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    scored_customers["review_rank"] = (
        scored_customers.index + 1
    )

    review_count = math.ceil(
        len(scored_customers) * capacity
    )

    review_queue = (
        scored_customers
        .head(review_count)
        .copy()
    )

    # Kept for compatibility with downstream reporting/database steps.
    review_queue["selected_for_review"] = True

    return review_queue


def main():
    X, y = prepare_data()

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
    )

    print("Training Gradient Boosting model...")

    model = build_boosted_model()

    model.fit(
        X_train,
        y_train,
    )

    probabilities = model.predict_proba(X_test)[:, 1]

    review_queue = build_review_queue(
        X_test=X_test,
        y_test=y_test,
        probabilities=probabilities,
        capacity=REVIEW_CAPACITY,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    review_queue.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    (
        _capacity,
        review_count,
        captured_problem_cases,
        capture_rate,
        queue_precision,
    ) = evaluate_review_capacity(
        y_test,
        probabilities,
        capacities=[REVIEW_CAPACITY],
    )[0]

    total_problem_cases = int(
        y_test.sum()
    )

    print("\nREVIEW QUEUE")
    print("=" * 60)

    print(f"Test population:        {len(X_test):,}")
    print(f"Review capacity:        {REVIEW_CAPACITY:.0%}")
    print(f"Available reviews:      {review_count:,}")

    print(
        "Problem cases captured: "
        f"{captured_problem_cases:,} / "
        f"{total_problem_cases:,}"
    )

    print(f"Capture rate:           {capture_rate:.2%}")
    print(f"Queue precision:        {queue_precision:.2%}")

    print("\nTop 10 priority customers:")

    print(
        review_queue[
            [
                "customer_id",
                "review_rank",
                "predicted_risk",
                "actual_outcome",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print(f"\nQueue saved to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
