# Validation record — 8 October 2026

Dataset: Kaggle *Give Me Some Credit*, `cs-training.csv.zip` (150,000 rows). An observation with age 0 is removed (149,999 remain). Internal stratified split: 119,999 train; 30,000 test (`random_state=42`).

All 13 original stages executed successfully on the training data in the validation environment, and six targeted pytest unit/integration tests passed. Reports and plots in `reports/` were regenerated from these data.

| Strategy | ROC-AUC | PR-AUC | Capture @ 20% |
| --- | ---: | ---: | ---: |
| Single-feature domain rule | — | — | 42.83% (mean over 100 tie-break repeats) |
| Logistic Regression | 0.8145 | 0.3465 | 65.49% |
| Histogram Gradient Boosting | 0.8682 | 0.3974 | 72.97% |

Gradient Boosting review queue: 6,000 customers, 1,463 of 2,005 positive test cases captured, observed queue precision 24.38%.

Tested with Python 3.13.5, NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0, Matplotlib 3.10.8, SciPy 1.17.0. Original archived reports contained a 73.27% capture at 20%; running the supplied code and data under the tested environment produced 72.97%. This is not necessarily a modeling improvement or degradation; rerun results under the stated environment before making a claim.

The separately supplied Kaggle `cs-test.csv.zip` lacks outcome labels and was not used for supervised evaluation. It is not needed to reproduce the proof-of-concept metrics.

## Local commands

Place the original `cs-training.csv.zip` in `data/raw/`, then run:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python src/run_pipeline.py
```

Raw data and generated SQLite database are not committed to version control. This is a historical demonstration and should not be used for automated real-world lending decisions.
