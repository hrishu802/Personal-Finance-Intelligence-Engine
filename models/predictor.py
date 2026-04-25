import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

class ExpensePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.metrics = {}
        
    def prepare_data(self, df):
        """
        Prepare advanced time series data for prediction.
        Features: rolling avg, weekend ratio, volatility (std dev), and total spend lag.
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['year_month'] = df['date'].dt.to_period('M')
        
        monthly = []
        months = sorted(df['year_month'].unique())
        
        for m in months:
            m_data = df[df['year_month'] == m]
            
            total_spend = m_data['amount'].sum()
            weekend_spend = m_data[m_data['is_weekend']]['amount'].sum()
            weekend_ratio = weekend_spend / total_spend if total_spend > 0 else 0
            
            # Volatility (std deviation of daily spending)
            daily_spend = m_data.groupby(m_data['date'].dt.date)['amount'].sum()
            volatility = daily_spend.std() if len(daily_spend) > 1 else 0
            
            monthly.append({
                'year_month': m,
                'total_spending': total_spend,
                'weekend_ratio': weekend_ratio,
                'volatility': volatility
            })
            
        m_df = pd.DataFrame(monthly)
        
        # Target is next month's spending
        m_df['target'] = m_df['total_spending'].shift(-1)
        
        # Lags and Rolling Avg
        m_df['lag_1'] = m_df['total_spending']
        m_df['lag_2'] = m_df['total_spending'].shift(1)
        m_df['rolling_3m_avg'] = m_df['total_spending'].rolling(window=3).mean()
        
        # Drop rows with NaN (early months and last month target)
        train_df = m_df.dropna()
        
        if len(train_df) < 3:
            raise ValueError("Not enough data to train advanced model.")
            
        features = ['lag_1', 'lag_2', 'rolling_3m_avg', 'weekend_ratio', 'volatility']
        X = train_df[features]
        y = train_df['target']
        
        return X, y, m_df
        
    def train(self, df):
        try:
            X, y, m_df = self.prepare_data(df)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            self.model.fit(X_train, y_train)
            
            y_pred = self.model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            self.metrics = {
                'MAE': mae,
                'RMSE': rmse
            }
            
            # Retrain on full for future
            self.model.fit(X, y)
            return True
        except Exception as e:
            print(f"Error training advanced model: {e}")
            return False
            
    def predict_next_month(self, df):
        try:
            # We just need to formulate the feature row for the *current* month to predict next month
            _, _, m_df = self.prepare_data(df) # This might drop the last row if target is NaN, wait.
            pass
        except ValueError:
            return None
            
        # Re-build feature row for the latest month
        df['year_month'] = df['date'].dt.to_period('M')
        monthly = []
        months = sorted(df['year_month'].unique())
        
        for m in months:
            m_data = df[df['year_month'] == m]
            total_spend = m_data['amount'].sum()
            weekend_spend = m_data[m_data['is_weekend']]['amount'].sum()
            weekend_ratio = weekend_spend / total_spend if total_spend > 0 else 0
            daily_spend = m_data.groupby(m_data['date'].dt.date)['amount'].sum()
            volatility = daily_spend.std() if len(daily_spend) > 1 else 0
            
            monthly.append({
                'total_spending': total_spend,
                'weekend_ratio': weekend_ratio,
                'volatility': volatility
            })
            
        m_df = pd.DataFrame(monthly)
        m_df['lag_1'] = m_df['total_spending']
        m_df['lag_2'] = m_df['total_spending'].shift(1)
        m_df['rolling_3m_avg'] = m_df['total_spending'].rolling(window=3).mean()
        
        latest_features = m_df.iloc[-1]
        
        if pd.isna(latest_features['rolling_3m_avg']):
            return None
            
        X_pred = pd.DataFrame({
            'lag_1': [latest_features['lag_1']],
            'lag_2': [latest_features['lag_2']],
            'rolling_3m_avg': [latest_features['rolling_3m_avg']],
            'weekend_ratio': [latest_features['weekend_ratio']],
            'volatility': [latest_features['volatility']]
        })
        
        prediction = self.model.predict(X_pred)[0]
        
        preds = []
        for estimator in self.model.estimators_:
            preds.append(estimator.predict(X_pred.values)[0])
        std_dev = np.std(preds)
        
        return prediction, std_dev

def detect_anomalies(df):
    alerts = []
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
                    if increase > 0.3 and recent_amt > 1000:
                        alerts.append({
                            'type': 'warning',
                            'message': f"Spending on {cat} increased by {increase*100:.1f}% compared to last month."
                        })
                        
    return alerts
