import pandas as pd
import numpy as np

def load_and_clean_data(filepath='data/transactions.csv'):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    
    # Handle missing values if any
    df = df.dropna(subset=['amount', 'category', 'date'])
    
    # Remove extreme anomalies (e.g., negative amounts or crazy high amounts)
    df = df[(df['amount'] > 0) & (df['amount'] < 1000000)]
    
    return df

def feature_engineering(df):
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['weekday'] = df['date'].dt.day_name()
    df['is_weekend'] = df['date'].dt.dayofweek >= 5
    
    # Define necessities
    essential_categories = ['Housing', 'Food', 'Transportation', 'Utilities', 'Healthcare', 'Insurance', 'Education']
    df['necessity'] = df['category'].apply(lambda x: 'Essential' if x in essential_categories else 'Luxury')
    
    # Define recurring (approximate logic based on category)
    recurring_categories = ['Housing', 'Utilities', 'Insurance']
    df['recurring'] = df['category'].apply(lambda x: 'Yes' if x in recurring_categories else 'No')
    
    # Sort by date
    df = df.sort_values('date')
    
    return df

def aggregate_monthly_data(df):
    # Group by year-month
    df['year_month'] = df['date'].dt.to_period('M')
    
    monthly_summary = df.groupby('year_month').agg(
        total_spending=('amount', 'sum'),
        transaction_count=('amount', 'count')
    ).reset_index()
    
    monthly_summary['year_month'] = monthly_summary['year_month'].astype(str)
    
    # Calculate spending trend (rolling average of last 3 months)
    monthly_summary['spending_trend'] = monthly_summary['total_spending'].rolling(window=3, min_periods=1).mean()
    
    return monthly_summary

def category_analysis(df):
    cat_summary = df.groupby('category')['amount'].sum().reset_index()
    cat_summary = cat_summary.sort_values(by='amount', ascending=False)
    return cat_summary

def get_financial_health_score(df, assumed_income=80000):
    # Calculate average monthly spending
    df['year_month'] = df['date'].dt.to_period('M')
    avg_monthly_spending = df.groupby('year_month')['amount'].sum().mean()
    
    savings_rate = (assumed_income - avg_monthly_spending) / assumed_income
    
    # Calculate luxury %
    recent_month = df['year_month'].max()
    recent_data = df[df['year_month'] == recent_month]
    luxury_spend = recent_data[recent_data['necessity'] == 'Luxury']['amount'].sum()
    total_spend = recent_data['amount'].sum()
    luxury_ratio = luxury_spend / total_spend if total_spend > 0 else 0
    
    # Base score on savings
    if savings_rate > 0.3:
        score = 80
    elif savings_rate > 0.2:
        score = 65
    elif savings_rate > 0.1:
        score = 50
    elif savings_rate > 0:
        score = 30
    else:
        score = 10
        
    # Adjust for luxury ratio
    if luxury_ratio < 0.2:
        score += 20
    elif luxury_ratio < 0.3:
        score += 10
    elif luxury_ratio > 0.5:
        score -= 10
        
    score = max(0, min(100, score)) # Clamp 0-100
    
    breakdown = {
        'savings_ratio': savings_rate,
        'luxury_ratio': luxury_ratio,
        'stability': 'Stable' if score > 60 else 'Volatile'
    }
        
    return score, breakdown

def get_risk_score(df, monthly_summary, health_score, assumed_income=80000):
    if len(monthly_summary) < 2:
        return "Low Risk", 0, []
        
    recent_month = monthly_summary.iloc[-1]['total_spending']
    prev_month = monthly_summary.iloc[-2]['total_spending']
    
    increase = (recent_month - prev_month) / prev_month if prev_month > 0 else 0
    
    risk_score = 0
    factors = []
    
    # 1. Spikes
    if increase > 0.2:
        risk_score += 40
        factors.append(f"Spending spiked by {increase*100:.0f}%")
    elif increase > 0.1:
        risk_score += 20
        factors.append(f"Spending increased by {increase*100:.0f}%")
        
    # 2. Income ratio
    if recent_month > assumed_income * 0.9:
        risk_score += 50
        factors.append("Spending dangerously close to income")
    elif recent_month > assumed_income * 0.8:
        risk_score += 30
        factors.append("High spending relative to income")
        
    # 3. Weekend Behavior
    recent_month_str = monthly_summary.iloc[-1]['year_month']
    df['year_month_str'] = df['date'].dt.to_period('M').astype(str)
    recent_data = df[df['year_month_str'] == recent_month_str]
    weekend_spend = recent_data[recent_data['is_weekend']]['amount'].sum()
    weekend_ratio = weekend_spend / recent_month if recent_month > 0 else 0
    
    if weekend_ratio > 0.4:
        risk_score += 25
        factors.append("High weekend spending behavior")
        
    # 4. Health Score Link
    if health_score < 40:
        risk_score += 30
        factors.append(f"Critically low health score ({health_score}/100)")
    elif health_score < 60:
        risk_score += 15
        factors.append(f"Suboptimal health score ({health_score}/100)")
        
    risk_score = min(100, risk_score)
        
    if risk_score > 60:
        risk_label = "High Risk"
    elif risk_score > 30:
        risk_label = "Moderate Risk"
    else:
        risk_label = "Low Risk"
        if not factors:
            factors.append("Stable spending patterns")
        
    return risk_label, risk_score, factors
