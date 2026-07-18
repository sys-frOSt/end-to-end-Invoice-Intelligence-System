import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

def load_invoice_data(db_path):
    conn = sqlite3.connect(db_path)
    query = """with purchases_agg as (
    select 
        p.PONumber,
        count(distinct p.Brand) as total_brands,
        sum(p.Quantity) as total_quantity,
        sum(p.Dollars) as total_items_dollars,
        avg(julianday(p.ReceivingDate) - julianday(p.PODate)) as avg_days_delay
    from purchases p
    group by p.PONumber
)
select
    vi.PONumber,
    vi.Quantity as invoice_quantity,
    vi.Dollars as invoice_dollars,
    vi.Freight ,
    julianday(vi.InvoiceDate) - julianday(vi.PODate) as days_to_po_invoice,
    julianday(vi.PayDate) - julianday(vi.InvoiceDate) as days_to_pay,
    pa.total_brands,
    pa.total_quantity,
    pa.total_items_dollars,
    pa.avg_days_delay
from vendor_invoice vi
left join purchases_agg pa on vi.PONumber = pa.PONumber
"""
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def create_invoice_risk_level(row):
    #missing values in the row
    if(abs(row['invoice_dollars'] - row['total_items_dollars']) > 5):
        return 1

    if(row['avg_days_delay'] > 10):
        return 1
    return 0

def apply_level(df):
    df['flag_invoice_risk'] = df.apply(create_invoice_risk_level, axis=1)
    return df


def split_data(df,features,target):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test


def scale_data(X_train, X_test, scaler_path):
    scaler_path = Path(scaler_path)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)

    if scaler_path.exists():
        scaler = joblib.load(str(scaler_path))
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        joblib.dump(scaler, str(scaler_path))

    return X_train_scaled, X_test_scaled


