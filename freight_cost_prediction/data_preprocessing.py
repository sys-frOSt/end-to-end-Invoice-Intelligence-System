import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data_from_sqlite(db_path):
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM vendor_invoice"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def preprocess_data(df):
    X = df[['Dollars']]
    y = df['Freight']
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


