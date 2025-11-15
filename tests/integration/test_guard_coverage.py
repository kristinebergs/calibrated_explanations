import pandas as pd
import numpy as np


def test_guard_coverage_within_tolerance():
    df = pd.read_csv('guard_coverage_results.csv')
    tol = 0.08  # allow some sampling variability for small datasets
    # Group by alpha and check mean empirical coverage
    for alpha, sub in df.groupby('alpha'):
        emp_mean = sub['empirical'].mean()
        assert abs(emp_mean - (1 - alpha)) <= tol, (
            f"alpha={alpha} empirical={emp_mean:.3f} target={1-alpha:.3f} tol={tol}"
        )
