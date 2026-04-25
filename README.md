# Personal Finance Intelligence Engine (PFIE)

## Problem Statement
Traditional expense trackers only show where money went, lacking predictive capabilities or actionable insights. Users struggle to identify overspending early, anticipate future expenses, or understand their financial health comprehensively.

## Solution
PFIE is a production-level, intelligent financial system that goes beyond simple tracking. It:
- **Analyzes** spending behavior and trends.
- **Predicts** future expenses using Machine Learning (Random Forest).
- **Detects** financial risks and overspending anomalies.
- **Provides** personalized, actionable recommendations to improve savings.
- **Features** an interactive, premium dark-themed dashboard.

## Features
1. **Overview Dashboard**: High-level KPI cards (This Month's Spend, Health Score, Risk Level) and spending distributions.
2. **Spending Analysis**: Deep dive into category breakdowns, day-vs-time heatmaps, and top merchants.
3. **Future Predictions**: ML-powered forecast for the upcoming month's expenses.
4. **Risk Alerts**: Automated detection of sudden spikes in spending.
5. **What-If Simulator**: Interactive tool to model budget reductions and visualize potential savings.
6. **AI Advisor**: Rule-based assistant offering tailored advice and natural language query responses.

## Tech Stack
- **Frontend/App**: Streamlit
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (Random Forest Regressor)
- **Visualizations**: Plotly Express & Graph Objects
- **Language**: Python 3

## How to Run

1. Clone the repository or navigate to the project directory.
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
   *(On the first run, the system will automatically generate a realistic synthetic dataset of transactions).*

## Business Impact
By shifting from reactive tracking to proactive intelligence, PFIE empowers users to make informed financial decisions, potentially increasing their monthly savings and significantly improving their long-term financial health.
