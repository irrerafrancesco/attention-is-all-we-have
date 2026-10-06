import math
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_model import (
    RANDOM_STATE,
    REVIEW_CAPACITIES,
    prepare_data,
    split_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "simple_rule_baseline.csv"
)

FEATURE = "NumberOfTimes90DaysLate"
N_REPEATS = 100


def evaluate_simple_rule(
    y_true,
    scores,
    capacities,
    n_repeats=N_REPEATS,
):
    """
    Evaluate a simple ranking rule based on a single feature.

    Many customers share the same delinquency count, so ties are broken
    randomly across multiple repetitions. The reported capture rate is
    therefore the average across repeated random tie-breaks.
    """

    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    if len(y_true) != len(scores):
        raise ValueError(
            "y_true and scores must have the same length."
        )

    if n_repeats <= 0:
        raise ValueError(
            "n_repeats must be greater than zero."
        )

    total_positive_cases = int(y_true.sum())

    if total_positive_cases == 0:
        raise ValueError(
            "Simple-rule evaluation requires at least one positive case."
        )

    results = []

    for capacity in capacities:

        if not 0 < capacity <= 1:
            raise ValueError(
                "Each review capacity must be between 0 and 1."
            )

        review_count = math.ceil(
            len(y_true) * capacity
        )

        capture_rates = []

        for repeat in range(n_repeats):

            rng = np.random.default_rng(
                RANDOM_STATE + repeat
            )

            evaluation = pd.DataFrame(
                {
                    "target": y_true,
                    "score": scores,
                    "tie_breaker": rng.random(len(y_true)),
                }
            )

            evaluation = evaluation.sort_values(
                by=[
                    "score",
                    "tie_breaker",
                ],
                ascending=[
                    False,
                    False,
                ],
            )

            selected = evaluation.head(
                review_count
            )

            captured_cases = int(
                selected["target"].sum()
            )

            capture_rate = (
                captured_cases
                / total_positive_cases
            )

            capture_rates.append(
                capture_rate
            )

        results.append(
            {
                "capacity": capacity,
                "review_count": review_count,
                "capture_rate_mean": np.mean(
                    capture_rates
                ),
                "capture_rate_std": np.std(
                    capture_rates
                ),
            }
        )

    return pd.DataFrame(results)


def main():
    X, y = prepare_data()

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
    )

    # Estimate the imputation value using training data only.
    training_median = X_train[
        FEATURE
    ].median()

    simple_rule_scores = X_test[
        FEATURE
    ].fillna(
        training_median
    )

    results = evaluate_simple_rule(
        y_true=y_test,
        scores=simple_rule_scores,
        capacities=REVIEW_CAPACITIES,
        n_repeats=N_REPEATS,
    )

    print("\nSIMPLE RULE BASELINE")
    print("=" * 60)

    print(f"Ranking feature: {FEATURE}")
    print(
        "Training-set median used for missing values: "
        f"{training_median}"
    )
    print(
        f"Random tie-breaking repetitions: {N_REPEATS}"
    )

    print("\nATTENTION BUDGET RESULTS")
    print("=" * 60)

    for _, row in results.iterrows():

        print(
            f"{row['capacity']:.0%} review capacity | "
            f"{int(row['review_count']):,} reviews | "
            f"{row['capture_rate_mean']:.2%} captured "
            f"(± {row['capture_rate_std']:.2%})"
        )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        REPORT_PATH,
        index=False,
    )

    print(f"\nResults saved to:\n{REPORT_PATH}")


if __name__ == "__main__":
    main()
