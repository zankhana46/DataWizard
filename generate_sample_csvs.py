"""Run once to generate sample CSV files in data/."""
import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
OUT = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT, exist_ok=True)

# ── Regression CSV ────────────────────────────────────────────────────────────
n = 300
age = rng.integers(22, 65, n)
experience = (age - 22 + rng.integers(0, 5, n)).clip(0, 40)
hours_per_week = rng.integers(30, 60, n)
education_years = rng.integers(12, 22, n)
salary = (
    30000
    + experience * 1500
    + education_years * 800
    + hours_per_week * 200
    + rng.normal(0, 5000, n)
).round(2)

reg_df = pd.DataFrame({
    "age": age,
    "experience_years": experience,
    "hours_per_week": hours_per_week,
    "education_years": education_years,
    "salary": salary,
})
reg_df.to_csv(os.path.join(OUT, "sample_regression.csv"), index=False)
print("Created sample_regression.csv")

# ── Classification CSV ────────────────────────────────────────────────────────
n = 300
credit_score = rng.integers(300, 850, n)
income = rng.integers(20000, 150000, n)
loan_amount = rng.integers(5000, 50000, n)
loan_term = rng.choice([12, 24, 36, 48, 60], n)
debt_to_income = (loan_amount / income * 100).round(2)

prob_default = 1 / (1 + np.exp(
    0.005 * credit_score + 0.00001 * income - 0.02 * debt_to_income - 3.5
))
default = (rng.random(n) < prob_default).astype(int)

clf_df = pd.DataFrame({
    "credit_score": credit_score,
    "income": income,
    "loan_amount": loan_amount,
    "loan_term_months": loan_term,
    "debt_to_income_pct": debt_to_income,
    "defaulted": default,
})
clf_df.to_csv(os.path.join(OUT, "sample_classification.csv"), index=False)
print("Created sample_classification.csv")
