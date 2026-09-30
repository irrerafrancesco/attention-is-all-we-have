from pathlib import Path
import sqlite3

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_COMPARISON_PATH = (
    PROJECT_ROOT
    / "reports"
    / "model_comparison.csv"
)

DATABASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "attention.db"
)

REPORTS_DIR = PROJECT_ROOT / "reports"

ATTENTION_CURVE_PATH = (
    REPORTS_DIR / "attention_curve.png"
)

RISK_BANDS_PATH = (
    REPORTS_DIR / "risk_bands.png"
)


def create_attention_curve():
    """Compare how efficiently each model uses limited review capacity."""

    comparison = pd.read_csv(MODEL_COMPARISON_PATH)

    capacities = [5, 10, 20, 30]

    plt.figure(figsize=(8, 5))

    for _, row in comparison.iterrows():
        capture_rates = [
            row["capture_at_5pct"] * 100,
            row["capture_at_10pct"] * 100,
            row["capture_at_20pct"] * 100,
            row["capture_at_30pct"] * 100,
        ]

        plt.plot(
            capacities,
            capture_rates,
            marker="o",
            label=row["model"],
        )

    # Random selection baseline:
    # reviewing 10% of customers would capture about 10% of problem cases.
    plt.plot(
        capacities,
        capacities,
        linestyle="--",
        label="Random Review",
    )

    plt.xlabel("Customers Reviewed (%)")
    plt.ylabel("Future Problem Cases Captured (%)")
    plt.title("Attention Efficiency Curve")

    plt.xticks(capacities)
    plt.ylim(0, 100)

    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        ATTENTION_CURVE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def create_risk_bands_chart():
    """Compare predicted and observed risk across review-queue segments."""

    query = """
        SELECT
            CASE
                WHEN predicted_risk >= 0.70 THEN 'Very High'
                WHEN predicted_risk >= 0.50 THEN 'High'
                WHEN predicted_risk >= 0.30 THEN 'Medium'
                ELSE 'Lower'
            END AS risk_band,

            AVG(predicted_risk) * 100
                AS predicted_risk_pct,

            100.0 * SUM(actual_outcome) / COUNT(*)
                AS observed_problem_rate_pct

        FROM review_queue

        GROUP BY risk_band

        ORDER BY AVG(predicted_risk)
    """

    with sqlite3.connect(DATABASE_PATH) as connection:
        risk_bands = pd.read_sql_query(
            query,
            connection,
        )

    x = range(len(risk_bands))

    width = 0.35

    plt.figure(figsize=(8, 5))

    plt.bar(
        [value - width / 2 for value in x],
        risk_bands["predicted_risk_pct"],
        width=width,
        label="Predicted Risk",
    )

    plt.bar(
        [value + width / 2 for value in x],
        risk_bands["observed_problem_rate_pct"],
        width=width,
        label="Observed Problem Rate",
    )

    plt.xlabel("Risk Band")
    plt.ylabel("Percentage (%)")
    plt.title("Predicted Risk vs Observed Outcomes")

    plt.xticks(
        list(x),
        risk_bands["risk_band"],
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RISK_BANDS_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def main():
    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Creating attention efficiency curve...")
    create_attention_curve()

    print("Creating risk-band comparison...")
    create_risk_bands_chart()

    print("\nReports created successfully.")
    print(f"- {ATTENTION_CURVE_PATH}")
    print(f"- {RISK_BANDS_PATH}")


if __name__ == "__main__":
    main()