from pathlib import Path
import sqlite3
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "attention.db"
)


def run_query(connection, title, query):
    print(f"\n{title}")
    print("=" * 70)

    result = pd.read_sql_query(query, connection)

    print(result.to_string(index=False))


def main():
    print("Opening database...")

    with sqlite3.connect(DATABASE_PATH) as connection:

        run_query(
            connection,
            "QUEUE SUMMARY",
            """
            SELECT
                COUNT(*) AS customers,
                ROUND(AVG(predicted_risk), 4) AS avg_predicted_risk,
                SUM(actual_outcome) AS actual_problem_cases
            FROM review_queue
            """
        )

        run_query(
            connection,
            "HIGH RISK CUSTOMERS",
            """
            SELECT
                COUNT(*) AS customers_above_50pct_risk
            FROM review_queue
            WHERE predicted_risk >= 0.50
            """
        )

        run_query(
            connection,
            "RISK BANDS",
            """
            SELECT
                CASE
                    WHEN predicted_risk >= 0.70 THEN 'Very High'
                    WHEN predicted_risk >= 0.50 THEN 'High'
                    WHEN predicted_risk >= 0.30 THEN 'Medium'
                    ELSE 'Lower'
                END AS risk_band,

                COUNT(*) AS customers,

                ROUND(
                    AVG(predicted_risk) * 100,
                    2
                ) AS avg_predicted_risk_pct,

                SUM(actual_outcome) AS actual_problem_cases,

                ROUND(
                    100.0 * SUM(actual_outcome) / COUNT(*),
                    2
                ) AS observed_problem_rate_pct

            FROM review_queue

            GROUP BY risk_band

            ORDER BY AVG(predicted_risk) DESC
            """
        )

        run_query(
            connection,
            "TOP 10 PRIORITY CASES",
            """
            SELECT
                customer_id,
                review_rank,
                ROUND(predicted_risk * 100, 2) AS predicted_risk_pct,
                actual_outcome
            FROM review_queue
            ORDER BY review_rank
            LIMIT 10
            """
        )


if __name__ == "__main__":
    main()