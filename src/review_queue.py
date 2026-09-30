from pathlib import Path
import math

import pandas as pd
from sklearn.model_selection import train_test_split

from baseline_model import prepare_data
from model_comparison import build_boosted_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "review_queue.csv"
)

REVIEW_CAPACITY = 0.20


def main():
    # Load the same cleaned feature set used for model evaluation.
    X, y = prepare_data()

    # Use exactly the same train/test split as the previous experiments.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("Training Gradient Boosting model...")

    model = build_boosted_model()
    model.fit(X_train, y_train)

    # Estimate the probability of future serious financial distress.
    probabilities = model.predict_proba(X_test)[:, 1]

    # Build a scored population.
    scored_customers = X_test.copy()

    scored_customers.insert(
        0,
        "customer_id",
        X_test.index + 1,
    )

    scored_customers["predicted_risk"] = probabilities

    # The true outcome is included only because this is a historical
    # validation dataset. It would not be available in production.
    scored_customers["actual_outcome"] = y_test.values

    # Highest predicted risk receives the highest review priority.
    scored_customers = scored_customers.sort_values(
        "predicted_risk",
        ascending=False,
    ).reset_index(drop=True)

    scored_customers["review_rank"] = (
        scored_customers.index + 1
    )

    review_count = math.ceil(
        len(scored_customers) * REVIEW_CAPACITY
    )

    # Select only the customers that analysts have capacity to review.
    review_queue = scored_customers.head(
        review_count
    ).copy()

    review_queue["selected_for_review"] = True

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    review_queue.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    total_problem_cases = y_test.sum()

    captured_problem_cases = (
        review_queue["actual_outcome"].sum()
    )

    capture_rate = (
        captured_problem_cases
        / total_problem_cases
    )

    queue_risk = (
        captured_problem_cases
        / len(review_queue)
    )

    print("\nREVIEW QUEUE")
    print("=" * 60)

    print(f"Test population:        {len(X_test):,}")
    print(f"Review capacity:        {REVIEW_CAPACITY:.0%}")
    print(f"Available reviews:      {review_count:,}")

    print(
        f"Problem cases captured: "
        f"{captured_problem_cases:,} / "
        f"{total_problem_cases:,}"
    )

    print(f"Capture rate:           {capture_rate:.2%}")
    print(f"Risk inside queue:      {queue_risk:.2%}")

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

    print(f"\nQueue saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()