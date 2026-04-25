import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

class ExpensePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.metrics = {}
        
    def prepare_data(self, monthly_summary):
        """
        Prepare time series data for prediction.
        Uses lag features to predict the next month's total spending.
        """
        # Ensure data is sorted
        df = monthly_summary.copy()
        df['total_spending'] = df['total_spending'].astype(float)
        
        # Create lag features
        df['lag_1'] = df['total_spending'].shift(1)
        df['lag_2'] = df['total_spending'].shift(2)
        df['lag_3'] = df['total_spending'].shift(3)
        
        # Drop rows with NaN values (the first 3 months)
        df = df.dropna()
        
        if len(df) < 3:
            raise ValueError("Not enough data to train the model. Need at least 4 months of data.")
            
        X = df[['lag_1', 'lag_2', 'lag_3']]
        y = df['total_spending']
        
        return X, y, df
        
    def train(self, monthly_summary):
        try:
            X, y, df = self.prepare_data(monthly_summary)
            
            # Simple train-test split (we want to use the most recent data for testing)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            self.model.fit(X_train, y_train)
            
            # Predict and evaluate
            y_pred = self.model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            self.metrics = {
                'MAE': mae,
                'RMSE': rmse
            }
            
            # Also train on full dataset for future prediction
            self.model.fit(X, y)
            
            return True
        except Exception as e:
            print(f"Error training model: {e}")
            return False
            
    def predict_next_month(self, monthly_summary):
        """
        Predict the expense for the next month based on the last 3 months
        """
        if len(monthly_summary) < 3:
            return None
            
        # Get the last 3 months
        recent = monthly_summary['total_spending'].values[-3:]
        
        # Features: [lag_1, lag_2, lag_3]
        # In our case, lag_1 is the most recent month, lag_3 is the oldest of the three
        # Wait, in prepare_data: lag_1 is t-1, lag_2 is t-2, lag_3 is t-3
        # So to predict t+1, lag_1 = t, lag_2 = t-1, lag_3 = t-2
        X_pred = pd.DataFrame({
            'lag_1': [recent[2]],
            'lag_2': [recent[1]],
            'lag_3': [recent[0]]
        })
        
        prediction = self.model.predict(X_pred)[0]
        
        # Calculate confidence interval (std deviation across trees)
        preds = []
        for estimator in self.model.estimators_:
            preds.append(estimator.predict(X_pred.values)[0])
        std_dev = np.std(preds)
        
        return prediction, std_dev

def detect_anomalies(df):
    """
    Detect sudden spikes in spending for specific categories or overall
    """
    alerts = []
    
    # Calculate monthly category spending
    df['year_month'] = df['date'].dt.to_period('M')
    monthly_cat = df.groupby(['year_month', 'category'])['amount'].sum().reset_index()
    
    months = sorted(monthly_cat['year_month'].unique())
    if len(months) >= 2:
        recent_month = months[-1]
        prev_month = months[-2]
        
        recent_data = monthly_cat[monthly_cat['year_month'] == recent_month]
        prev_data = monthly_cat[monthly_cat['year_month'] == prev_month]
        
        for index, row in recent_data.iterrows():
            cat = row['category']
            recent_amt = row['amount']
            
            prev_row = prev_data[prev_data['category'] == cat]
            if not prev_row.empty:
                prev_amt = prev_row['amount'].values[0]
                
                if prev_amt > 0:
                    increase = (recent_amt - prev_amt) / prev_amt
                    if increase > 0.3 and recent_amt > 1000: # Significant increase > 30% and amount > 1000
                        alerts.append({
                            'type': 'warning',
                            'message': f"Spending on {cat} increased by {increase*100:.1f}% compared to last month."
                        })
                        
    return alerts
