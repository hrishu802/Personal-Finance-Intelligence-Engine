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

def calculate_savings_opportunity(df):
    """
    Computes potential monthly savings by identifying excess in discretionary categories.
    """
    df['year_month'] = df['date'].dt.to_period('M')
    recent_month = df['year_month'].max()
    recent_data = df[df['year_month'] == recent_month]
    
    opportunities = []
    total_potential = 0
    
    # 1. Dining/Food (suggest 20% cut if high)
    food = recent_data[recent_data['category'] == 'Food']['amount'].sum()
    if food > 8000:
        cut = food * 0.20
        total_potential += cut
        opportunities.append({'category': 'Dining & Food', 'amount': cut})
        
    # 2. Weekend Spend (suggest 15% cut if high)
    weekend = recent_data[recent_data['is_weekend']]['amount'].sum()
    if weekend > 10000:
        cut = weekend * 0.15
        total_potential += cut
        opportunities.append({'category': 'Weekend Splurges', 'amount': cut})
        
    # 3. Entertainment/Misc (suggest 25% cut)
    misc = recent_data[recent_data['category'].isin(['Entertainment', 'Miscellaneous'])]['amount'].sum()
    if misc > 5000:
        cut = misc * 0.25
        total_potential += cut
        opportunities.append({'category': 'Entertainment & Misc', 'amount': cut})
        
    # Calculate projected health score impact (rough estimate: +10 for every 10% of income saved)
    # Since we don't have income here, we will just pass back the raw savings amount and let app.py format the health jump, or just return a static jump if it's high.
    # Actually, we can just say "Potential +X points"
    health_jump = min(35, int((total_potential / 50000) * 20)) if total_potential > 0 else 0
    # Make it realistic
    health_jump = max(5, health_jump) if total_potential > 2000 else 0
        
    return total_potential, opportunities, health_jump

def ai_advisor_response(query, df, monthly_summary, health_score=None):
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
        
    elif "health" in query or "score" in query:
        if health_score is not None:
            if health_score < 40:
                return f"Your health score is critical ({health_score}/100). You are severely overspending relative to your income. I strongly recommend cutting down discretionary weekend expenses immediately."
            elif health_score < 70:
                return f"Your health score is moderate ({health_score}/100). You have decent financial stability, but increasing your savings rate by 10% would improve your score significantly."
            else:
                return f"Your financial health is excellent ({health_score}/100)! You maintain a great savings ratio."
        return "I need more data to calculate your health score."
        
    else:
        return "I'm your proactive AI financial assistant. I can analyze your overspending, project savings, and review your health score. Try asking: 'How is my health score?' or 'How can I save?'"

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
