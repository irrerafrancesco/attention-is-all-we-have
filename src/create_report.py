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

SIMPLE_RULE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "simple_rule_baseline.csv"
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
    """
    Compare how efficiently different prioritization strategies
    use limited human review capacity.
    """

    comparison = pd.read_csv(MODEL_COMPARISON_PATH)
    simple_rule = pd.read_csv(SIMPLE_RULE_PATH)

    capacities = [5, 10, 20, 30]

    plt.figure(figsize=(8, 5))

    # Machine-learning models.
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

    # Simple domain-based rule:
    # prioritize customers only by previous 90+ day delinquencies.
    simple_rule_capacities = (
        simple_rule["capacity"] * 100
    )

    simple_rule_capture = (
        simple_rule["capture_rate_mean"] * 100
    )

    plt.plot(
        simple_rule_capacities,
        simple_rule_capture,
        marker="o",
        label="90+ Days Late Rule",
    )

    # Random selection baseline:
    # reviewing x% of customers would capture approximately
    # x% of problem cases on average.
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
    """
    Compare predicted and observed risk across
    review-queue segments.
    """

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