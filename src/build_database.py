
from pathlib import Path
import sqlite3
from contextlib import closing

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

QUEUE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "review_queue.csv"
)

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
}


def build_database(
    queue_path=QUEUE_PATH,
    database_path=DATABASE_PATH,
):
    """Load the review queue into a SQLite database."""

    if not queue_path.is_file():
        raise FileNotFoundError(
            f"Review queue not found: {queue_path}\n"
            "Run review_queue.py first."
        )

    review_queue = pd.read_csv(queue_path)

    if review_queue.empty:
        raise ValueError("The review queue is empty.")

    missing_columns = REQUIRED_COLUMNS - set(review_queue.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    database_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            review_queue.to_sql(
                TABLE_NAME,
                connection,
                if_exists="replace",
                index=False,
            )

            inserted_rows = connection.execute(
                f"SELECT COUNT(*) FROM {TABLE_NAME}"
            ).fetchone()[0]

    if inserted_rows != len(review_queue):
        raise RuntimeError(
            "Database row count does not match the source CSV."
        )

    return inserted_rows


def main():
    inserted_rows = build_database()

    print("\nDATABASE CREATED SUCCESSFULLY")
    print("=" * 60)

    print(f"Table:         {TABLE_NAME}")
    print(f"Rows inserted: {inserted_rows:,}")
    print(f"Database:      {DATABASE_PATH}")


if __name__ == "__main__":
    main()
