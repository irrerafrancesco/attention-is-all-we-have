from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from baseline_model import prepare_data
from model_comparison import build_boosted_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORTS_DIR = PROJECT_ROOT / "reports"

IMPORTANCE_CSV_PATH = (
    REPORTS_DIR / "feature_importance.csv"
)

IMPORTANCE_PLOT_PATH = (
    REPORTS_DIR / "feature_importance.png"
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

    print("Training Gradient Boosting model...")

    model = build_boosted_model()
    model.fit(X_train, y_train)

    print("Calculating permutation importance...")

    importance = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )

    results = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    )

    results = results.sort_values(
        "importance_mean",
        ascending=False,
    ).reset_index(drop=True)

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        IMPORTANCE_CSV_PATH,
        index=False,
    )

    print("\nFEATURE IMPORTANCE")
    print("=" * 70)

    print(
        results[
            [
                "feature",
                "importance_mean",
                "importance_std",
            ]
        ].to_string(index=False)
    )

    # Plot from least important to most important
    plot_data = results.sort_values(
        "importance_mean",
        ascending=True,
    )

    plt.figure(figsize=(9, 6))

    plt.barh(
        plot_data["feature"],
        plot_data["importance_mean"],
        xerr=plot_data["importance_std"],
    )

    plt.xlabel("Decrease in Average Precision")
    plt.ylabel("Feature")
    plt.title("Permutation Feature Importance")

    plt.tight_layout()

    plt.savefig(
        IMPORTANCE_PLOT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("\nResults saved to:")
    print(IMPORTANCE_CSV_PATH)
    print(IMPORTANCE_PLOT_PATH)


if __name__ == "__main__":
    main()