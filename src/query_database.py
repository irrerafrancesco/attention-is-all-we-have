
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "attention.db"
)

TABLE_NAME = "review_queue"

REQUIRED_COLUMNS = {
    "customer_id",
    "review_rank",
    "predicted_risk",
    "actual_outcome",
}


def validate_database(connection):
    """Check that the expected table and columns exist."""

    table_exists = connection.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table' AND name = ?
        """,
        (TABLE_NAME,),
    ).fetchone()

    if table_exists is None:
        raise ValueError(
            f"Table '{TABLE_NAME}' not found. "
            "Run build_database.py first."
        )

    columns = {
        row[1]
        for row in connection.execute(
            "PRAGMA table_info(review_queue)"
        )
    }

    missing_columns = REQUIRED_COLUMNS - columns

    if missing_columns:
        raise ValueError(
            f"Missing database columns: {sorted(missing_columns)}"
        )


def run_query(connection, title, query):
    """Execute a SQL query and display its results."""

    result = pd.read_sql_query(query, connection)

    print(f"\n{title}")
    print("=" * 70)

    if result.empty:
        print("No results found.")
    else:
        print(result.to_string(index=False))

    return result


def main():
    if not DATABASE_PATH.is_file():
        raise FileNotFoundError(
            f"Database not found: {DATABASE_PATH}\n"
            "Run build_database.py first."
        )

    print("Opening database...")

    with closing(sqlite3.connect(DATABASE_PATH)) as connection:

        validate_database(connection)

        run_query(
            connection,
            "QUEUE SUMMARY",
            """
            SELECT
                COUNT(*) AS customers,
                ROUND(AVG(predicted_risk), 4)
                    AS avg_predicted_risk,
                COALESCE(SUM(actual_outcome), 0)
                    AS actual_problem_cases
            FROM review_queue
            """,
        )

        run_query(
            connection,
            "CUSTOMERS ABOVE 50% PREDICTED RISK",
            """
            SELECT
                COUNT(*) AS customers_above_50pct_risk
            FROM review_queue
            WHERE predicted_risk >= 0.50
            """,
        )

        run_query(
            connection,
            "PREDICTED RISK BANDS",
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
            """,
        )

        run_query(
            connection,
            "TOP 10 PRIORITY CASES",
            """
            SELECT
                customer_id,
                review_rank,
                ROUND(predicted_risk * 100, 2)
                    AS predicted_risk_pct,
                actual_outcome

            FROM review_queue

            ORDER BY review_rank ASC

            LIMIT 10
            """,
        )

    print("\nDatabase queries completed successfully.")


if __name__ == "__main__":
    main()
