import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from preprocess import preprocess_data
from baseline_model import evaluate_review_capacity
from review_queue import build_review_queue
from build_database import build_database
from query_database import validate_database
import sqlite3


def test_preprocessing_preserves_missing_values_and_removes_bad_ages():
    df = pd.DataFrame({
        'age': [30, 0, 40],
        'NumberOfTime30-59DaysPastDueNotWorse': [96, 0, 2],
        'NumberOfTimes90DaysLate': [0, 0, 98],
        'NumberOfTime60-89DaysPastDueNotWorse': [0, 0, 1],
        'MonthlyIncome': [np.nan, 1000, 2000],
    })
    cleaned = preprocess_data(df)
    assert len(cleaned) == 2
    assert pd.isna(cleaned.loc[0, 'NumberOfTime30-59DaysPastDueNotWorse'])
    assert pd.isna(cleaned.loc[2, 'NumberOfTimes90DaysLate'])
    assert pd.isna(cleaned.loc[0, 'MonthlyIncome'])


def test_attention_metrics():
    y_true = pd.Series([0, 1, 0, 1])
    predictions = np.array([0.1, 0.9, 0.3, 0.8])
    assert evaluate_review_capacity(y_true, predictions, [0.5]) == [(.5, 2, 2, 1.0, 1.0)]


def test_attention_invalid_inputs():
    with pytest.raises(ValueError):
        evaluate_review_capacity(pd.Series([0, 0]), [0.1, 0.2], [0.5])
    with pytest.raises(ValueError):
        evaluate_review_capacity(pd.Series([0, 1]), [0.1], [0.5])
    with pytest.raises(ValueError):
        evaluate_review_capacity(pd.Series([0, 1]), [0.1, 0.2], [1.1])


def test_review_queue_rank_and_capacity():
    X = pd.DataFrame({'age': [30, 31, 32, 33]}, index=[3, 4, 5, 6])
    y = pd.Series([0, 1, 0, 1], index=X.index)
    scores = np.array([0.1, 0.9, 0.3, 0.8])
    queue = build_review_queue(X, y, scores, capacity=.5)
    assert queue['customer_id'].tolist() == [5, 7]
    assert queue['review_rank'].tolist() == [1, 2]
    assert queue['actual_outcome'].tolist() == [1, 1]


def test_sqlite_database_roundtrip(tmp_path):
    csv_path = tmp_path / 'queue.csv'
    db_path = tmp_path / 'db' / 'attention.db'
    pd.DataFrame({'customer_id': [1, 2], 'review_rank': [1, 2],
                  'predicted_risk': [.8, .4], 'actual_outcome': [1, 0]}).to_csv(csv_path, index=False)
    assert build_database(csv_path, db_path) == 2
    with sqlite3.connect(db_path) as conn:
        validate_database(conn)
        assert conn.execute('SELECT COUNT(*) FROM review_queue').fetchone()[0] == 2


def test_pipeline_runs_scripts_in_order_without_executing_them():
    from unittest.mock import patch
    import run_pipeline
    with patch.object(run_pipeline.subprocess, 'run') as subprocess_run:
        run_pipeline.main()
    called_paths = [call.args[0][1] for call in subprocess_run.call_args_list]
    assert len(called_paths) == len(run_pipeline.SCRIPTS)
    assert called_paths[0].endswith('data_loader.py')
    assert called_paths[-1].endswith('explain_model.py')
    assert all(call.kwargs['check'] is True for call in subprocess_run.call_args_list)
