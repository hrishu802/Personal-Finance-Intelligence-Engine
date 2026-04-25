import pandas as pd
import numpy as np

def generate_smart_insights(df, monthly_summary):
    insights = []
    
    if len(monthly_summary) < 2:
        return [{"icon": "ℹ️", "text": "Not enough data for historical insights. Keep using the app to generate trends.", "type": "neutral"}]
        
    recent_month = monthly_summary.iloc[-1]
    prev_month = monthly_summary.iloc[-2]
    
    # 1. Total Spending Change
    spend_diff = recent_month['total_spending'] - prev_month['total_spending']
    perc_change = (spend_diff / prev_month['total_spending']) * 100 if prev_month['total_spending'] > 0 else 0
    
    if perc_change > 10:
        insights.append({
            'icon': '📈',
            'text': f"Overall spending **increased by {abs(perc_change):.1f}%** (₹{abs(spend_diff):,.0f}). **Recommendation**: Review your discretionary categories like Dining or Entertainment to curb this upward trend.",
            'type': 'warning'
        })
    elif perc_change < -5:
        insights.append({
            'icon': '📉',
            'text': f"Great! Spending **decreased by {abs(perc_change):.1f}%** (₹{abs(spend_diff):,.0f}). **Recommendation**: Transfer this saved amount to an index fund or fixed deposit to build wealth.",
            'type': 'success'
        })
        
    # 2. Top Category Contribution
    df['year_month_str'] = df['date'].dt.to_period('M').astype(str)
    recent_data = df[df['year_month_str'] == recent_month['year_month']]
    
    if not recent_data.empty:
        cat_spending = recent_data.groupby('category')['amount'].sum()
        top_cat = cat_spending.idxmax()
        top_amt = cat_spending.max()
        top_perc = (top_amt / recent_month['total_spending']) * 100
        
        insights.append({
            'icon': '📊',
            'text': f"**{top_cat}** is your largest expense, contributing **{top_perc:.1f}%** of total expenses. **Recommendation**: Set a hard budget cap for {top_cat} next month to free up cash flow.",
            'type': 'neutral'
        })
        
    # 3. Weekend vs Weekday Behavior
    weekend_spend = recent_data[recent_data['is_weekend']]['amount'].sum()
    total_spend = recent_data['amount'].sum()
    weekend_ratio = (weekend_spend / total_spend) * 100 if total_spend > 0 else 0
    
    if weekend_ratio > 40:
        insights.append({
            'icon': '🎉',
            'text': f"Weekend spending is unusually high at **{weekend_ratio:.1f}%**. **Recommendation**: Try shifting some discretionary spend to weekdays or plan low-cost weekend activities.",
            'type': 'warning'
        })
    elif weekend_ratio < 15:
        insights.append({
            'icon': '🛋️',
            'text': f"Weekend spending is very low (**{weekend_ratio:.1f}%**). **Recommendation**: Keep maintaining this balanced lifestyle; it significantly improves your savings rate.",
            'type': 'success'
        })
        
    return insights

def get_kpi_explanation(df, current_month_str, prev_month_str):
    inc, dec = what_changed_analysis(df, current_month_str, prev_month_str)
    
    curr_spend = df[df['date'].dt.to_period('M').astype(str) == current_month_str]['amount'].sum()
    prev_spend = df[df['date'].dt.to_period('M').astype(str) == prev_month_str]['amount'].sum()
    
    if curr_spend > prev_spend:
        if not inc.empty:
            driver = inc.iloc[0].name
            return f"Driven largely by higher {driver} expenses"
        return "Driven by a general increase across categories"
    elif curr_spend < prev_spend:
        if not dec.empty:
            driver = dec.iloc[0].name
            return f"Driven largely by lower {driver} activity"
        return "Driven by generalized savings this month"
    
    return "Spending remained stable"

def what_changed_analysis(df, current_month_str, prev_month_str):
    curr_data = df[df['date'].dt.to_period('M').astype(str) == current_month_str]
    prev_data = df[df['date'].dt.to_period('M').astype(str) == prev_month_str]
    
    curr_cat = curr_data.groupby('category')['amount'].sum()
    prev_cat = prev_data.groupby('category')['amount'].sum()
    
    comparison = pd.DataFrame({
        'Current': curr_cat,
        'Previous': prev_cat
    }).fillna(0)
    
    comparison['Diff'] = comparison['Current'] - comparison['Previous']
    comparison['% Change'] = np.where(comparison['Previous'] > 0, 
                                      (comparison['Diff'] / comparison['Previous']) * 100, 
                                      100) # If prev was 0, it's a 100% increase
    
    # Ignore small changes (< ₹500)
    comparison = comparison[abs(comparison['Diff']) > 500].sort_values('Diff', ascending=False)
    
    major_increases = comparison[comparison['Diff'] > 0].head(3)
    major_decreases = comparison[comparison['Diff'] < 0].tail(3).sort_values('Diff') # Most negative first
    
    return major_increases, major_decreases
