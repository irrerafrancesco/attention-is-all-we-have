# Attention Is All We Have

**Human-in-the-loop risk prioritization under limited review capacity.**

> We cannot review everything.  
> The real question is not only *who is at risk?* — but *where should limited human attention be spent first?*

---

## Overview

Most credit-risk projects focus on a standard machine-learning question:

> Can we predict which customers will experience financial distress?

This project starts from a different problem.

In a real organization, analysts have limited time and cannot manually investigate every customer with the same level of attention.

So the question becomes:

> **If only a fraction of customers can be manually reviewed, which customers should be reviewed first?**

**Attention Is All We Have** is a human-in-the-loop risk prioritization system designed around this constraint.

The objective is not to replace analysts.

The objective is to use machine learning to help them focus their attention where it is most likely to matter.

---

## Core Idea

The system:

1. estimates each customer's probability of serious financial distress;
2. ranks customers according to predicted risk;
3. creates a priority review queue;
4. adapts the queue to the available human review capacity;
5. measures how much future financial distress can be captured within that limited capacity.

The project therefore treats human attention as a scarce resource.

Instead of asking only:

> *How accurate is the model?*

it asks:

> **How much future risk can we capture with a limited review budget?**

---

## Dataset

The project uses the **Give Me Some Credit** dataset.

The training dataset contains approximately **150,000 customers** and information including:

- revolving credit utilization;
- age;
- debt ratio;
- monthly income;
- number of open credit lines and loans;
- previous 30–59 day delinquencies;
- previous 60–89 day delinquencies;
- previous 90+ day delinquencies;
- real-estate loans or credit lines;
- number of dependents.

The target variable is:

`SeriousDlqin2yrs`

which indicates whether a customer experienced serious financial distress within the following two years.

Only **6.68%** of observations belong to the positive class, making the problem strongly imbalanced.

The raw dataset is not included in this repository and must be obtained separately from the original Kaggle competition.

---

## Data Quality

Before modelling, the dataset was inspected for missing values, implausible observations and anomalous values.

Main issues identified:

- approximately 19.8% missing values in `MonthlyIncome`;
- approximately 2.6% missing values in `NumberOfDependents`;
- one observation with `age = 0`;
- anomalous values such as `96` and `98` in delinquency-count variables;
- extreme values in credit utilization, debt ratio and monthly income.

For model training:

- the impossible age observation was removed;
- anomalous delinquency codes were treated as missing values;
- missing values were imputed using statistics calculated only from the training data.

This avoids leaking information from the test set into the model.

---

## Models

Two models were compared using the same stratified train/test split:

- **Logistic Regression** as an interpretable baseline;
- **Histogram Gradient Boosting** as a nonlinear model.

### Predictive Performance

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Logistic Regression | 0.8145 | 0.3465 |
| Gradient Boosting | **0.8682** | **0.3979** |

Because the positive class represents only 6.68% of customers, accuracy alone would be misleading.

For this reason, the project focuses on ranking quality, PR-AUC and attention-budget metrics.

---

## The Attention Budget

Customers are ranked by predicted probability of serious financial distress.

The system then simulates different levels of available human review capacity.

### Future Problem Cases Captured

| Review Capacity | Logistic Regression | Gradient Boosting |
|---|---:|---:|
| 5% | 33.37% | **36.06%** |
| 10% | 50.72% | **54.76%** |
| 20% | 65.49% | **73.27%** |
| 30% | 74.11% | **83.94%** |

The central result is:

> **By reviewing only 20% of customers, the Gradient Boosting model concentrated 73.27% of all future serious financial-distress cases inside the review queue.**

For the 30,000-customer test population:

- total future problem cases: **2,005**
- available reviews at 20% capacity: **6,000**
- problem cases captured: **1,469**
- capture rate: **73.27%**
- observed problem rate inside the selected queue: **24.48%**

This converts the model from a prediction tool into a **resource-allocation system**.

---

## Priority Review Queue

The Gradient Boosting model produces a ranked queue of customers.

Instead of presenting analysts with thousands of isolated risk probabilities, the system produces an operational output:

```text
Customer 1  → highest priority
Customer 2  → second priority
Customer 3  → third priority
...
```

At 20% review capacity, the system selects the 6,000 highest-risk customers from the 30,000-customer test population.

The resulting queue is stored as structured data and can be queried by downstream systems or analysts.

---

## Risk Bands

The selected queue was divided into risk bands.

| Risk Band | Customers | Avg. Predicted Risk | Observed Problem Rate |
|---|---:|---:|---:|
| Very High | 81 | 74.45% | 70.37% |
| High | 541 | 58.78% | 57.49% |
| Medium | 1,024 | 38.70% | 38.96% |
| Lower | 4,354 | 15.55% | 16.12% |

The close relationship between predicted risk and observed outcomes provides an additional validation that the ranking meaningfully separates different risk levels.

---

## Explainability

Permutation importance was used to estimate which variables contributed most strongly to model performance.

The most influential features were:

| Feature | Permutation Importance |
|---|---:|
| NumberOfTimes90DaysLate | 0.0993 |
| RevolvingUtilizationOfUnsecuredLines | 0.0791 |
| NumberOfTime30-59DaysPastDueNotWorse | 0.0393 |
| NumberOfTime60-89DaysPastDueNotWorse | 0.0341 |
| age | 0.0141 |

The model therefore relies primarily on previous delinquency behaviour and credit utilization when prioritizing customers.

---

## SQL Layer

The generated review queue is stored in a **SQLite database**.

SQL is used as part of the operational layer rather than simply as an additional technology.

Queries can answer questions such as:

- Which customers have the highest priority?
- How many customers have predicted risk above 50%?
- How many customers belong to each risk band?
- What is the observed problem rate within each risk segment?

This creates a separation between:

**Machine Learning**

which generates the risk ranking,

and

**SQL / Operational Analytics**

which allows analysts or downstream applications to work with the resulting queue.

---

## Project Structure

```text
attention-is-all-we-have/
│
├── reports/
│   ├── attention_curve.png
│   ├── risk_bands.png
│   ├── feature_importance.png
│   ├── feature_importance.csv
│   └── model_comparison.csv
│
├── sql/
│   └── review_queue.sql
│
├── src/
│   ├── data_loader.py
│   ├── data_explore.py
│   ├── data_quality.py
│   ├── preprocess.py
│   ├── baseline_model.py
│   ├── boosted_model.py
│   ├── model_comparison.py
│   ├── review_queue.py
│   ├── build_database.py
│   ├── query_database.py
│   ├── create_report.py
│   └── explain_model.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

The `data/` directory is intentionally excluded from version control.

---

## Workflow

```text
Raw Credit Data
       ↓
Data Exploration
       ↓
Data Quality Analysis
       ↓
Preprocessing
       ↓
Logistic Regression Baseline
       ↓
Gradient Boosting
       ↓
Model Comparison
       ↓
Risk Ranking
       ↓
Attention Budget
       ↓
Priority Review Queue
       ↓
SQLite / SQL
       ↓
Operational Analysis
       ↓
Reporting & Explainability
```

---

## Visual Results

### Attention Efficiency

The attention-efficiency curve compares the percentage of customers reviewed with the percentage of future serious financial-distress cases captured.

![Attention Efficiency Curve](reports/attention_curve.png)

### Risk Bands

Predicted risk is compared with the observed problem rate across the generated risk bands.

![Risk Bands](reports/risk_bands.png)

### Feature Importance

Permutation importance shows which customer characteristics contribute most strongly to the model's ranking performance.

![Feature Importance](reports/feature_importance.png)

---

## Installation

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Data Setup

Download the `cs-training.csv` dataset from the original **Give Me Some Credit** Kaggle competition.

Place the compressed training file at:

```text
data/raw/cs-training.csv.zip
```

The data directory is excluded from Git.

---

## Running the Project

Run the data inspection stages:

```bash
python src/data_loader.py
python src/data_explore.py
python src/data_quality.py
```

Run preprocessing:

```bash
python src/preprocess.py
```

Train and compare the models:

```bash
python src/baseline_model.py
python src/boosted_model.py
python src/model_comparison.py
```

Generate the operational review queue:

```bash
python src/review_queue.py
```

Build and query the SQLite database:

```bash
python src/build_database.py
python src/query_database.py
```

Generate visual reports and explainability outputs:

```bash
python src/create_report.py
python src/explain_model.py
```

---

## Why This Project?

Machine-learning models are often evaluated