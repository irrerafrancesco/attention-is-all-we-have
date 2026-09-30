from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from baseline_model import prepare_data


FEATURE = "NumberOfTimes90DaysLate"
CAPACITIES = [0.05, 0.10, 0.20, 0.30]
RANDOM_STATE = 42
N_REPEATS = 100


def evaluate_simple_rule(y_true, scores, capacities, n_repeats=100):
    """
    Evaluate a simple ranking rule based on a single feature.

    Because many customers have the same delinquency count, ties are broken
    randomly several times. We report the average capture rate across repeats.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    total_positive_cases = y_true.sum()

    results = []

    for capacity in capacities:
        review_count = int(len(y_true) * capacity)
        capture_rates = []

        for repeat in range(n_repeats):
            rng = np.random.default_rng(RANDOM_STATE + repeat)

            evaluation = pd.DataFrame(
                {
                    "target": y_true,
                    "score": scores,
                    "tie_breaker": rng.random(len(y_true)),
                }
            )

            evaluation = evaluation.sort_values(
                by=["score", "tie_breaker"],
                ascending=[False, False],
            )

            selected = evaluation.head(review_count)

            captured_cases = selected["target"].sum()
            capture_rate = captured_cases / total_positive_cases

            capture_rates.append(capture_rate)

        results.append(
            {
                "capacity": capacity,
                "review_count": review_count,
                "capture_rate_mean": np.mean(capture_rates),
                "capture_rate_std": np.std(capture_rates),
            }
        )

    return pd.DataFrame(results)


def main():
    X, y = prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Median calculated only from the training set to avoid data leakage.
    training_median = X_train[FEATURE].median()

    simple_rule_scores = X_test[FEATURE].fillna(training_median)

    results = evaluate_simple_rule(
        y_true=y_test,
        scores=simple_rule_scores,
        capacities=CAPACITIES,
        n_repeats=N_REPEATS,
    )

    print("\nSIMPLE RULE BASELINE")
    print(f"Ranking feature: {FEATURE}")
    print(f"Training-set median used for missing values: {training_median}")
    print(f"Random tie-breaking repetitions: {N_REPEATS}")

    print("\nATTENTION BUDGET RESULTS")

    for _, row in results.iterrows():
        print(
            f"{row['capacity']:.0%} review capacity | "
            f"{int(row['review_count']):,} reviews | "
            f"{row['capture_rate_mean']:.2%} captured "
            f"(± {row['capture_rate_std']:.2%})"
        )

    output_path = Path("reports/simple_rule_baseline.csv")
    results.to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()