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

Two machine-learning models were compared using the same stratified train/test split:

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

## Do We Actually Need Machine Learning?

Before assuming that a multivariate model was necessary, the system was compared with a much simpler domain-based rule:

> **Review first the customers with the highest number of previous 90+ day delinquencies.**

The feature used was:

`NumberOfTimes90DaysLate`

Because many customers share the same delinquency count, ties were broken randomly across **100 repetitions**, and the mean capture rate was reported.

This provides a deliberately simple benchmark against which the machine-learning models can be evaluated.

---

## The Attention Budget

Customers are ranked according to each prioritization strategy.

The system then simulates different levels of available human review capacity.

### Future Problem Cases Captured

| Review Capacity | Random Review | 90+ Days Late Rule | Logistic Regression | Gradient Boosting |
|---|---:|---:|---:|---:|
| 5% | 5.00% | 30.81% | 33.37% | **36.06%** |
| 10% | 10.00% | 35.65% | 50.72% | **54.76%** |
| 20% | 20.00% | 42.83% | 65.49% | **73.27%** |
| 30% | 30.00% | 49.99% | 74.11% | **83.94%** |

The comparison shows that previous severe delinquency is already a useful signal.

However, as review capacity increases, the simple single-feature rule captures substantially less risk than the multivariate models.

At **20% review capacity**:

- random review would capture approximately **20%** of future problem cases;
- the 90+ days late rule captures **42.83%**;
- Logistic Regression captures **65.49%**;
- Gradient Boosting captures **73.27%**.

This suggests that combining multiple customer characteristics adds meaningful prioritization value beyond obvious prior delinquency history.

---

## Main Result

The central result of the project is:

> **By reviewing only 20% of customers, the Gradient Boosting model concentrated 73.27% of all future serious financial-distress cases inside the review queue.**

For the 30,000-customer test population:

- total future problem cases: **2,005**
- available reviews at 20% capacity: **6,000**
- problem cases captured: **1,469**
- capture rate: **73.27%**
- observed problem rate inside the selected queue: **24.48%**

The simple 90+ day delinquency rule captures approximately **42.83%** of future problem cases at the same review capacity.

The comparison therefore tests whether model complexity actually adds value rather than assuming that it does.

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

The selected review queue was divided into risk bands.

| Risk Band | Customers | Avg. Predicted Risk | Observed Problem Rate |
|---|---:|---:|---:|
| Very High | 81 | 74.45% | 70.37% |
| High | 541 | 58.78% | 57.49% |
| Medium | 1,024 | 38.70% | 38.96% |
| Lower | 4,354 | 15.55% | 16.12% |

The observed rates are descriptively close to the average predicted probabilities across these broad segments.

This is useful as a diagnostic, but it should not be interpreted as a formal probability-calibration analysis.

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

Previous delinquency behaviour and revolving credit utilization are therefore among the strongest contributors to the model's ranking performance.

The fact that `NumberOfTimes90DaysLate` is the most important feature also motivated the single-feature baseline used to test whether a simpler rule could achieve comparable prioritization performance.

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
│   ├── model_comparison.csv
│   └── simple_rule_baseline.csv
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
│   ├── simple_rule_baseline.py
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
Simple Domain-Based Rule
       ↓
Logistic Regression
       ↓
Gradient Boosting
       ↓
Strategy Comparison
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

It includes:

- random review;
- a single-feature 90+ day delinquency rule;
- Logistic Regression;
- Gradient Boosting.

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

Train the machine-learning models:

```bash
python src/baseline_model.py
python src/boosted_model.py
```

Evaluate the simple domain-based baseline:

```bash
python src/simple_rule_baseline.py
```

Compare the machine-learning models:

```bash
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

Machine-learning models are often evaluated as if prediction were the final objective.

Real organizations operate under constraints.

Analysts have limited time.

Investigations have a cost.

Not every customer can receive the same level of attention.

**Attention Is All We Have** reframes risk modelling as an allocation problem:

> **How can machine learning help humans focus limited attention where it is most likely to matter?**

The project also asks a second question:

> **Do we actually need a machine-learning model, or would a simple domain rule be enough?**

By comparing random review, a single-feature delinquency rule, Logistic Regression and Gradient Boosting under the same review capacities, the project measures the additional value provided by increasingly sophisticated prioritization strategies.

The model does not replace the analyst.

It helps decide **where the analyst should look first**.

---

## Limitations

This project is a historical machine-learning proof of concept and not a production credit-decision system.

Important limitations include:

- the dataset represents a specific historical lending population;
- the system should not be used to make real lending decisions;
- fairness and protected-attribute analysis would be required before any real-world credit application;
- the review capacity is represented through simplified percentage-based scenarios;
- the observed target is available only for retrospective validation;
- operational costs are not explicitly modelled;
- the models were evaluated on a single held-out test split rather than through a full cross-validation study;
- hyperparameter tuning was intentionally limited;
- no formal probability-calibration procedure was performed;
- a production implementation would require monitoring for data drift, performance drift and changes in review behaviour.

These limitations are intentional boundaries of the current proof of concept rather than claims of production readiness.

---

## Key Takeaway

> **20% of human review capacity captured 73.27% of future serious financial-distress cases.**

A simple delinquency-based rule captured **42.83%** under the same constraint.

When attention is limited, the problem is not only predicting risk.

It is deciding **where to look first — and whether additional model complexity actually helps.**