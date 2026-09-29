# Banking Income vs Spending Analysis

## Objective
Evaluate whether higher income automatically translates into higher spending and produce evidence-based conclusions for management.

## Key Insights
1. Income and spending are positively related, with a correlation of 0.92, so higher-income customers spend more in absolute terms.
2. The average savings ratio is 44.23%, showing that customers do not automatically convert higher income into proportionally higher spending.
3. The strongest income segment is 46-55, which suggests age and life-stage patterns influence expenditure behavior.
4. Self-employed customers spend 61.95% of income versus 53.18% for salaried customers, indicating a materially different expense profile.
5. About 2.2% of customers show unusually high expenditure relative to income, which suggests caution about assuming all spending is income-driven.
6. The regression model explains 87% of variation in monthly expenses from income and lifestyle variables, which supports a partial but not total income-driven narrative.
7. Overall, the management assumption is partly supported: income matters, but occupation, employment type, and outlier behavior make the relationship non-linear and segment-specific.

## Regression Evaluation
- R^2: 0.8700
- RMSE: 890.69
- MAE: 678.66

## Practical Action Plan
1. Tailor offers by employment type: use separate affordability benchmarks for salaried and self-employed customers.
2. Monitor unusual spending behavior using spending-to-income alerts for customers above the 95th percentile.
3. Review savings-product campaigns in mid-income age bands where savings ratios are weakest.
4. Update underwriting rules using income and credit-card spend as combined signals rather than income alone.

## Data Summary
- Records analyzed: 600
- Average monthly income: $9,920.74
- Average monthly expenses: $5,587.35
- Average savings ratio: 44.23%