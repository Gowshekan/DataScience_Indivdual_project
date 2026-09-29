import pandas as pd

import src.banking_income_spending_analysis as analysis


def test_dataset_has_expected_columns_and_rows():
    df = analysis.build_dataset(n_rows=200)

    required = {
        "Customer_ID",
        "Age_Group",
        "Occupation",
        "Monthly_Income",
        "Monthly_Expenses",
        "Savings",
        "Credit_Card_Spending",
        "Loan_Amount",
        "Employment_Type",
    }

    assert isinstance(df, pd.DataFrame)
    assert required.issubset(set(df.columns))
    assert len(df) == 200
    assert (df["Monthly_Income"] > 0).all()


def test_key_business_metrics_are_computed():
    df = analysis.build_dataset(n_rows=120)
    summary = analysis.compute_segment_summary(df)

    assert "Savings_Ratio" in summary.columns
    assert "Monthly_Expenses" in summary.columns
    assert "Monthly_Income" in summary.columns
    assert not summary.empty
