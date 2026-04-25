import pandas as pd
import numpy as np

def generate_recommendations(df, income=80000):
    recommendations = []
    
    # Analyze recent month's data
    df['year_month'] = df['date'].dt.to_period('M')
    recent_month = df['year_month'].max()
    recent_data = df[df['year_month'] == recent_month]
    
    # Calculate category spending
    cat_spending = recent_data.groupby('category')['amount'].sum()
    
    # Rule 1: High Dining/Food expenses
    if 'Food' in cat_spending:
        food_spending = cat_spending['Food']
        if food_spending > income * 0.15: # If food is > 15% of income
            potential_savings = food_spending * 0.15
            recommendations.append({
                'title': 'High Dining Expenses',
                'description': f"Reduce dining/food expenses by 15% to save ₹{potential_savings:.0f}/month.",
                'icon': '🍔',
                'type': 'warning'
            })
            
    # Rule 2: Entertainment
    if 'Entertainment' in cat_spending:
        ent_spending = cat_spending['Entertainment']
        if ent_spending > income * 0.10:
            potential_savings = ent_spending * 0.20
            recommendations.append({
                'title': 'Excessive Entertainment Spending',
                'description': f"Your entertainment spending is high. Cut back by 20% to save ₹{potential_savings:.0f}/month.",
                'icon': '🎬',
                'type': 'warning'
            })
            
    # Rule 3: Subscriptions check
    # Check for recurring small payments in entertainment or utilities
    subs = recent_data[recent_data['category'].isin(['Entertainment', 'Utilities'])]
    if len(subs) > 5:
        recommendations.append({
            'title': 'Multiple Subscriptions Detected',
            'description': "You have several small recurring payments. Review them for low usage to save money.",
            'icon': '💳',
            'type': 'info'
        })
        
    if not recommendations:
        recommendations.append({
            'title': 'Great Job!',
            'description': "Your spending seems well within limits. Consider investing the surplus.",
            'icon': '🌟',
            'type': 'success'
        })
        
    return recommendations

def ai_advisor_response(query, df, monthly_summary):
    query = query.lower()
    
    if "overspend" in query or "over spending" in query:
        # Check risk score
        recent_spend = monthly_summary.iloc[-1]['total_spending']
        prev_spend = monthly_summary.iloc[-2]['total_spending']
        if recent_spend > prev_spend * 1.1:
            return "Based on your data, your recent spending is 10% higher than last month. You are showing signs of overspending. I recommend reviewing your discretionary expenses."
        else:
            return "You are currently within normal spending limits compared to previous months. Great job maintaining your budget!"
            
    elif "save" in query or "saving" in query:
        recs = generate_recommendations(df)
        if recs and recs[0]['type'] != 'success':
            return f"To save money, I suggest: {recs[0]['description']}"
        else:
            return "You are already saving well! You might want to look into mutual funds or fixed deposits to grow your wealth."
            
    elif "predict" in query or "future" in query:
        return "Based on my forecasting models, your expenses are analyzed continuously. Please check the 'Predictions' tab for a detailed forecast of your upcoming month's expenses."
        
    elif "highest" in query or "top" in query:
        cat_summary = df.groupby('category')['amount'].sum().sort_values(ascending=False)
        top_cat = cat_summary.index[0]
        return f"Your highest spending category overall is {top_cat}. Consider setting a strict budget for this category."
        
    else:
        return "I'm a simple rule-based AI advisor. I can help answer questions about overspending, saving tips, and your top expenses. Try asking: 'Am I overspending?'"

def simulate_savings(df, category_reductions):
    """
    category_reductions: dict of {category: reduction_percentage (0 to 1)}
    """
    df['year_month'] = df['date'].dt.to_period('M')
    recent_month = df['year_month'].max()
    recent_data = df[df['year_month'] == recent_month]
    
    original_spending = recent_data['amount'].sum()
    new_spending = 0
    savings = 0
    
    for category in recent_data['category'].unique():
        cat_spending = recent_data[recent_data['category'] == category]['amount'].sum()
        
        if category in category_reductions:
            reduction = category_reductions[category]
            saved = cat_spending * reduction
            savings += saved
            new_spending += (cat_spending - saved)
        else:
            new_spending += cat_spending
            
    return original_spending, new_spending, savings
