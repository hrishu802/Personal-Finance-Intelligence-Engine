import sqlite3
import pandas as pd
import os

DB_PATH = 'data/finance.db'

def get_db_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    
    # Create transactions table
    c.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            user_id TEXT NOT NULL,
            category TEXT NOT NULL,
            amount REAL NOT NULL,
            payment_mode TEXT NOT NULL,
            merchant TEXT NOT NULL,
            city TEXT NOT NULL
        )
    ''')
    
    # Create budgets table
    c.execute('''
        CREATE TABLE IF NOT EXISTS budgets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            category TEXT NOT NULL,
            monthly_limit REAL NOT NULL,
            UNIQUE(user_id, category)
        )
    ''')
    
    conn.commit()
    conn.close()

def save_transactions_to_db(df):
    conn = get_db_connection()
    df.to_sql('transactions', conn, if_exists='replace', index=False)
    conn.close()

def load_transactions_from_db():
    conn = get_db_connection()
    try:
        df = pd.read_sql_query("SELECT * FROM transactions", conn)
    except pd.errors.DatabaseError:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df

def save_budget(user_id, category, monthly_limit):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        INSERT INTO budgets (user_id, category, monthly_limit)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id, category) DO UPDATE SET monthly_limit=excluded.monthly_limit
    ''', (user_id, category, monthly_limit))
    conn.commit()
    conn.close()

def get_all_budgets(user_id):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT category, monthly_limit FROM budgets WHERE user_id=?", (user_id,))
    rows = c.fetchall()
    conn.close()
    return {row['category']: row['monthly_limit'] for row in rows}
