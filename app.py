from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src.banking_income_spending_analysis import (
    build_dataset,
    compare_employment_patterns,
    compute_segment_summary,
    detect_unusual_spending,
    train_linear_regression,
)

DATA_PATH = Path(__file__).resolve().parent / "data" / "banking_dataset.csv"
st.set_page_config(page_title="Banking Insights Dashboard", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df = build_dataset(n_rows=600)
        df.to_csv(DATA_PATH, index=False)

    if "Savings_Ratio" not in df.columns:
        df["Savings_Ratio"] = df["Savings"] / df["Monthly_Income"]
    if "Expense_to_Income_Ratio" not in df.columns:
        df["Expense_to_Income_Ratio"] = df["Monthly_Expenses"] / df["Monthly_Income"]
    return df


def main() -> None:
    st.title("Banking Analysis Dashboard")
    st.caption("Does income really determine spending?")

    df = load_data()
    summary = compute_segment_summary(df)
    comparison = compare_employment_patterns(df)
    unusual = detect_unusual_spending(df)
    model, metrics, _ = train_linear_regression(df)

    st.subheader("Executive Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Records", f"{len(df):,}")
    col2.metric("Avg Income", f"₹{df['Monthly_Income'].mean():,.0f}")
    col3.metric("Avg Savings Ratio", f"{df['Savings_Ratio'].mean():.1%}")
    col4.metric("Regression R²", f"{metrics['R2']:.3f}")

    st.markdown("---")

    left, right = st.columns(2)
    with left:
        st.subheader("Income vs. Monthly Expenses")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=df, x="Monthly_Income", y="Monthly_Expenses", hue="Employment_Type", alpha=0.7, ax=ax)
        ax.set_title("Income vs. Monthly Expenses")
        ax.set_xlabel("Monthly Income (₹)")
        ax.set_ylabel("Monthly Expenses (₹)")
        st.pyplot(fig)

    with right:
        st.subheader("Savings Ratio by Segment")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=summary, x="Age_Group", y="Savings_Ratio", hue="Employment_Type", ax=ax)
        ax.set_title("Savings Ratio by Age Group and Employment Type")
        ax.set_xlabel("Age Group")
        ax.set_ylabel("Savings Ratio")
        st.pyplot(fig)

    st.subheader("Employment Type Comparison")
    st.dataframe(comparison.round(2), use_container_width=True)

    left2, right2 = st.columns(2)
    with left2:
        st.subheader("Expense-to-Income Ratio by Employment Type")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.boxplot(data=df, x="Employment_Type", y="Expense_to_Income_Ratio", ax=ax)
        ax.set_title("Expense-to-Income Ratio")
        ax.set_xlabel("Employment Type")
        ax.set_ylabel("Expense / Income")
        st.pyplot(fig)

    with right2:
        st.subheader("Unusual Spending Outliers")
        unusual_top = unusual.head(25)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(data=unusual_top, x="Monthly_Income", y="Monthly_Expenses", size="Spending_ZScore", sizes=(20, 200), legend=False, alpha=0.8, ax=ax)
        ax.set_title("High-Spending Outliers Relative to Income")
        ax.set_xlabel("Monthly Income (₹)")
        ax.set_ylabel("Monthly Expenses (₹)")
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Key Business Insights")
    corr = df["Monthly_Income"].corr(df["Monthly_Expenses"])
    avg_savings_ratio = df["Savings_Ratio"].mean()
    outlier_share = unusual["Is_Unusual"].mean() * 100
    self_emp_ratio = comparison.loc[comparison["Employment_Type"] == "Self-Employed", "Expense_to_Income_Ratio"].iloc[0]
    salaried_ratio = comparison.loc[comparison["Employment_Type"] == "Salaried", "Expense_to_Income_Ratio"].iloc[0]

    st.write(f"- Income and spending have a positive correlation of {corr:.2f}.")
    st.write(f"- Average savings ratio is {avg_savings_ratio:.2%}, which shows customers do not automatically spend all additional income.")
    st.write(f"- Self-employed customers spend {self_emp_ratio:.2%} of income, compared with {salaried_ratio:.2%} for salaried customers.")
    st.write(f"- Approximately {outlier_share:.1f}% of customers show unusually high spending relative to income.")
    st.write(f"- The linear regression model explains {metrics['R2']:.0%} of variation in monthly expenses based on the selected variables.")

    st.markdown("---")
    st.subheader("Regression Evaluation")
    st.write({
        "R2": round(metrics["R2"], 4),
        "RMSE": round(metrics["RMSE"], 2),
        "MAE": round(metrics["MAE"], 2),
    })

    st.subheader("Practical Action Plan")
    st.markdown(
        "1. Segment customers separately for salaried and self-employed groups.\n"
        "2. Create spending alerts for income-to-expense outliers.\n"
        "3. Promote savings products for age groups with weaker savings ratios.\n"
        "4. Use income plus credit-card and loan signals together in risk decisions."
    )


if __name__ == "__main__":
    main()
