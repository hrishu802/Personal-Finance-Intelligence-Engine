import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

from utils.db import init_db, save_transactions_to_db

def generate_synthetic_data(num_months=12, transactions_per_month=100):
    np.random.seed(42)
    random.seed(42)
    
    categories = ['Housing', 'Food', 'Transportation', 'Utilities', 'Insurance', 
                  'Healthcare', 'Entertainment', 'Personal', 'Education', 'Miscellaneous']
    
    payment_modes = ['Credit Card', 'Debit Card', 'UPI', 'Net Banking', 'Cash']
    cities = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Pune']
    
    # Define category weights to make it realistic
    cat_weights = [0.1, 0.25, 0.15, 0.05, 0.05, 0.05, 0.15, 0.1, 0.05, 0.05]
    
    # Define merchants per category
    merchants = {
        'Housing': ['Landlord', 'Society Maintenance'],
        'Food': ['Swiggy', 'Zomato', 'Instamart', 'Blinkit', 'Local Grocery', 'Starbucks'],
        'Transportation': ['Uber', 'Ola', 'Indian Oil', 'Metro'],
        'Utilities': ['Electricity Board', 'Jio', 'Airtel', 'Water Bill'],
        'Insurance': ['LIC', 'HDFC Life', 'Star Health'],
        'Healthcare': ['Apollo Pharmacy', 'Practo', 'Local Clinic'],
        'Entertainment': ['Netflix', 'Amazon Prime', 'PVR', 'BookMyShow'],
        'Personal': ['Amazon', 'Myntra', 'Nykaa', 'Salon'],
        'Education': ['Coursera', 'Udemy', 'School Fee'],
        'Miscellaneous': ['Hardware Store', 'Gift Shop']
    }
    
    data = []
    start_date = datetime.now() - timedelta(days=num_months * 30)
    
    user_id = 'USER_101'
    
    for i in range(num_months * transactions_per_month):
        # Random date within the period
        days_to_add = random.randint(0, num_months * 30)
        txn_date = start_date + timedelta(days=days_to_add)
        
        category = random.choices(categories, weights=cat_weights)[0]
        
        # Generate realistic amounts based on category
        if category == 'Housing':
            amount = np.random.normal(25000, 2000)
        elif category == 'Food':
            amount = np.random.lognormal(mean=6, sigma=0.8) # Right skewed
        elif category == 'Transportation':
            amount = np.random.normal(500, 200)
        elif category == 'Utilities':
            amount = np.random.normal(1500, 300)
        else:
            amount = np.random.exponential(scale=1500)
            
        amount = max(50, round(amount, 2)) # Minimum 50 Rs
        
        payment_mode = random.choice(payment_modes)
        merchant = random.choice(merchants[category])
        city = random.choice(cities)
        
        data.append([txn_date, user_id, category, amount, payment_mode, merchant, city])
        
    df = pd.DataFrame(data, columns=['date', 'user_id', 'category', 'amount', 'payment_mode', 'merchant', 'city'])
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Save to SQLite
    init_db()
    save_transactions_to_db(df)
    print(f"Successfully generated {len(df)} transactions to SQLite database")
    
    return df

if __name__ == "__main__":
    generate_synthetic_data()
