from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"


def build_dataset(n_rows: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age_groups = ["18-25", "26-35", "36-45", "46-55", "56+"]
    occupations = [
        "Engineer",
        "Teacher",
        "Doctor",
        "Marketing",
        "Business Owner",
        "Manager",
        "Artist",
        "IT Professional",
        "Consultant",
        "Nurse",
    ]
    employment_types = ["Salaried", "Self-Employed"]

    data = []
    for idx in range(n_rows):
        age_group = rng.choice(age_groups, p=[0.12, 0.28, 0.24, 0.22, 0.14])
        occupation = rng.choice(occupations)
        employment_type = rng.choice(employment_types, p=[0.68, 0.32])

        if occupation in {"Engineer", "IT Professional", "Manager", "Consultant"}:
            base_income = 9000
        elif occupation in {"Doctor", "Business Owner"}:
            base_income = 14000
        elif occupation in {"Teacher", "Nurse"}:
            base_income = 6500
        elif occupation == "Artist":
            base_income = 5500
        else:
            base_income = 7800

        age_factor = {"18-25": -0.18, "26-35": 0.00, "36-45": 0.16, "46-55": 0.22, "56+": 0.14}[age_group]
        employment_factor = 0.12 if employment_type == "Self-Employed" else -0.05
        income_noise = rng.normal(0, 1800)
        monthly_income = max(2500, base_income * (1 + age_factor + employment_factor) + income_noise)

        spend_ratio = 0.52 + rng.normal(0, 0.08)
        if employment_type == "Self-Employed":
            spend_ratio += 0.08
        if occupation in {"Business Owner", "Consultant"}:
            spend_ratio += 0.04

        if age_group == "18-25":
            spend_ratio += 0.04
        monthly_expenses = monthly_income * max(0.35, min(0.9, spend_ratio))

        credit_card_spending = monthly_income * rng.uniform(0.08, 0.26)
        if employment_type == "Self-Employed":
            credit_card_spending *= 1.12

        loan_amount = max(0, monthly_income * rng.uniform(0.2, 1.1))
        savings = monthly_income - monthly_expenses

        data.append(
            {
                "Customer_ID": f"CUST-{idx + 1:05d}",
                "Age_Group": age_group,
                "Occupation": occupation,
                "Employment_Type": employment_type,
                "Monthly_Income": round(monthly_income, 2),
                "Monthly_Expenses": round(monthly_expenses, 2),
                "Savings": round(savings, 2),
                "Credit_Card_Spending": round(credit_card_spending, 2),
                "Loan_Amount": round(loan_amount, 2),
            }
        )

    df = pd.DataFrame(data)
    df = clean_dataset(df)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates(subset="Customer_ID").reset_index(drop=True)
    cleaned["Age_Group"] = cleaned["Age_Group"].astype(str).str.strip()
    cleaned["Occupation"] = cleaned["Occupation"].astype(str).str.strip()
    cleaned["Employment_Type"] = cleaned["Employment_Type"].astype(str).str.strip()

    numeric_columns = [
        "Monthly_Income",
        "Monthly_Expenses",
        "Savings",
        "Credit_Card_Spending",
        "Loan_Amount",
    ]
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.dropna(subset=numeric_columns + ["Age_Group", "Occupation", "Employment_Type"]).copy()
    cleaned = cleaned[(cleaned["Monthly_Income"] > 0) & (cleaned["Monthly_Expenses"] >= 0)].copy()

    cleaned["Savings_Ratio"] = cleaned["Savings"] / cleaned["Monthly_Income"]
    cleaned["Expense_to_Income_Ratio"] = cleaned["Monthly_Expenses"] / cleaned["Monthly_Income"]
    return cleaned


def compute_segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["Age_Group", "Employment_Type"], as_index=False)
        .agg(
            Monthly_Income=("Monthly_Income", "mean"),
            Monthly_Expenses=("Monthly_Expenses", "mean"),
            Savings=("Savings", "mean"),
            Credit_Card_Spending=("Credit_Card_Spending", "mean"),
            Loan_Amount=("Loan_Amount", "mean"),
        )
        .copy()
    )
    summary["Savings_Ratio"] = summary["Savings"] / summary["Monthly_Income"]
    summary["Expense_to_Income_Ratio"] = summary["Monthly_Expenses"] / summary["Monthly_Income"]
    return summary


def detect_unusual_spending(df: pd.DataFrame) -> pd.DataFrame:
    spending_ratio = df["Monthly_Expenses"] / df["Monthly_Income"]
    z_scores = (spending_ratio - spending_ratio.mean()) / spending_ratio.std(ddof=0)
    outliers = df.copy()
    outliers["Spending_Ratio"] = spending_ratio
    outliers["Spending_ZScore"] = z_scores
    outliers["Is_Unusual"] = (z_scores > 2) | (spending_ratio > 1.0)
    return outliers.sort_values("Spending_ZScore", ascending=False)


def compare_employment_patterns(df: pd.DataFrame) -> pd.DataFrame:
    comparison = (
        df.groupby("Employment_Type", as_index=False)
        .agg(
            Avg_Income=("Monthly_Income", "mean"),
            Avg_Expenses=("Monthly_Expenses", "mean"),
            Avg_Savings=("Savings", "mean"),
            Avg_Credit_Card=("Credit_Card_Spending", "mean"),
            Avg_Loan=("Loan_Amount", "mean"),
        )
        .copy()
    )
    comparison["Savings_Ratio"] = comparison["Avg_Savings"] / comparison["Avg_Income"]
    comparison["Expense_to_Income_Ratio"] = comparison["Avg_Expenses"] / comparison["Avg_Income"]
    return comparison


def train_linear_regression(df: pd.DataFrame):
    features = pd.get_dummies(
        df[["Monthly_Income", "Credit_Card_Spending", "Loan_Amount", "Age_Group", "Occupation", "Employment_Type"]],
        columns=["Age_Group", "Occupation", "Employment_Type"],
        drop_first=True,
    )
    target = df["Monthly_Expenses"]

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "R2": r2_score(y_test, predictions),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "MAE": float(np.mean(np.abs(y_test - predictions))),
    }
    return model, metrics, predictions


def save_visuals(df: pd.DataFrame, summary: pd.DataFrame, comparison: pd.DataFrame, unusual: pd.DataFrame, model):
    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="Monthly_Income", y="Monthly_Expenses", hue="Employment_Type", alpha=0.7)
    plt.title("Income vs. Monthly Expenses")
    plt.xlabel("Monthly Income")
    plt.ylabel("Monthly Expenses")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "income_vs_expenses.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 6))
    sns.barplot(data=summary, x="Age_Group", y="Savings_Ratio", hue="Employment_Type")
    plt.title("Savings Ratio by Age Group and Employment Type")
    plt.xlabel("Age Group")
    plt.ylabel("Savings Ratio")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "savings_ratio_by_segment.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Employment_Type", y="Expense_to_Income_Ratio")
    plt.title("Expense-to-Income Ratio by Employment Type")
    plt.xlabel("Employment Type")
    plt.ylabel("Expense / Income")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "employment_spending_pattern.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    unusual_top = unusual.head(25)
    sns.scatterplot(data=unusual_top, x="Monthly_Income", y="Monthly_Expenses", size="Spending_ZScore", sizes=(20, 200), legend=False, alpha=0.8)
    plt.title("High-Spending Outliers Relative to Income")
    plt.xlabel("Monthly Income")
    plt.ylabel("Monthly Expenses")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "unusual_spending_outliers.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    model_coef = pd.Series(model.coef_, index=model.feature_names_in_)
    model_coef.sort_values().plot(kind="barh")
    plt.title("Regression Coefficient Importance")
    plt.xlabel("Coefficient")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "regression_coefficients.png", dpi=200)
    plt.close()


def summarize_insights(df: pd.DataFrame, comparison: pd.DataFrame, unusual: pd.DataFrame, metrics: dict) -> list[str]:
    corr = df["Monthly_Income"].corr(df["Monthly_Expenses"])
    avg_savings_ratio = df["Savings_Ratio"].mean()
    highest_income_group = (
        df.groupby("Age_Group")["Monthly_Income"].mean().sort_values(ascending=False).index[0]
    )
    self_emp_ratio = comparison.loc[comparison["Employment_Type"] == "Self-Employed", "Expense_to_Income_Ratio"].iloc[0]
    salaried_ratio = comparison.loc[comparison["Employment_Type"] == "Salaried", "Expense_to_Income_Ratio"].iloc[0]
    outlier_share = unusual["Is_Unusual"].mean() * 100
    income_explain = metrics["R2"]

    insights = [
        f"Income and spending are positively related, with a correlation of {corr:.2f}, so higher-income customers spend more in absolute terms.",
        f"The average savings ratio is {avg_savings_ratio:.2%}, showing that customers do not automatically convert higher income into proportionally higher spending.",
        f"The strongest income segment is {highest_income_group}, which suggests age and life-stage patterns influence expenditure behavior.",
        f"Self-employed customers spend {self_emp_ratio:.2%} of income versus {salaried_ratio:.2%} for salaried customers, indicating a materially different expense profile.",
        f"About {outlier_share:.1f}% of customers show unusually high expenditure relative to income, which suggests caution about assuming all spending is income-driven.",
        f"The regression model explains {income_explain:.0%} of variation in monthly expenses from income and lifestyle variables, which supports a partial but not total income-driven narrative.",
        "Overall, the management assumption is partly supported: income matters, but occupation, employment type, and outlier behavior make the relationship non-linear and segment-specific.",
    ]
    return insights


def build_report(df: pd.DataFrame, comparison: pd.DataFrame, unusual: pd.DataFrame, metrics: dict, model) -> str:
    summary = compute_segment_summary(df)
    insights = summarize_insights(df, comparison, unusual, metrics)

    report_lines = [
        "# Banking Income vs Spending Analysis",
        "",
        "## Objective",
        "Evaluate whether higher income automatically translates into higher spending and produce evidence-based conclusions for management.",
        "",
        "## Key Insights",
    ]
    for index, insight in enumerate(insights, start=1):
        report_lines.append(f"{index}. {insight}")

    report_lines.extend([
        "",
        "## Regression Evaluation",
        f"- R^2: {metrics['R2']:.4f}",
        f"- RMSE: {metrics['RMSE']:.2f}",
        f"- MAE: {metrics['MAE']:.2f}",
        "",
        "## Practical Action Plan",
        "1. Tailor offers by employment type: use separate affordability benchmarks for salaried and self-employed customers.",
        "2. Monitor unusual spending behavior using spending-to-income alerts for customers above the 95th percentile.",
        "3. Review savings-product campaigns in mid-income age bands where savings ratios are weakest.",
        "4. Update underwriting rules using income and credit-card spend as combined signals rather than income alone.",
        "",
        "## Data Summary",
        f"- Records analyzed: {len(df):,}",
        f"- Average monthly income: ₹{df['Monthly_Income'].mean():,.2f}",
        f"- Average monthly expenses: ₹{df['Monthly_Expenses'].mean():,.2f}",
        f"- Average savings ratio: {df['Savings_Ratio'].mean():.2%}",
    ])

    return "\n".join(report_lines)


def main() -> None:
    df = build_dataset(n_rows=600)
    summary = compute_segment_summary(df)
    comparison = compare_employment_patterns(df)
    unusual = detect_unusual_spending(df)
    model, metrics, _ = train_linear_regression(df)

    OUTPUT_DIR.mkdir(exist_ok=True)
    save_visuals(df, summary, comparison, unusual, model)
    report = build_report(df, comparison, unusual, metrics, model)
    (OUTPUT_DIR / "banking_analysis_report.md").write_text(report, encoding="utf-8")

    print("Banking analysis completed successfully.")
    print(f"Records analyzed: {len(df)}")
    print(f"R^2: {metrics['R2']:.4f}")
    print(f"RMSE: ₹{metrics['RMSE']:.2f}")
    print(f"Average Savings Ratio: {df['Savings_Ratio'].mean():.2%}")
    print(f"Outlier spending share: {(unusual['Is_Unusual'].mean() * 100):.1f}%")
    print(f"Report saved to: {OUTPUT_DIR / 'banking_analysis_report.md'}")
    print("Plots saved in:", PLOTS_DIR)


if __name__ == "__main__":
    main()
