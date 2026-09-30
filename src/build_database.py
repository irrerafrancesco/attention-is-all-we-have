from pathlib import Path
import sqlite3

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


def main():
    review_queue = pd.read_csv(QUEUE_PATH)

    with sqlite3.connect(DATABASE_PATH) as connection:
        review_queue.to_sql(
            "review_queue",
            connection,
            if_exists="replace",
            index=False,
        )

    print("Database created successfully.")
    print(f"Rows inserted: {len(review_queue):,}")
    print(f"Database: {DATABASE_PATH}")


if __name__ == "__main__":
    main()