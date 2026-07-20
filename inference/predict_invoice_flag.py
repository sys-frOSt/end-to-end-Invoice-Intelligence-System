import joblib
import pandas as pd
from pathlib import Path

# Match training features from Invoice_Flagging_Classification/train.py
FEATURES = [
    'invoice_quantity', 'invoice_dollars', 'Freight', 'total_quantity', 'total_items_dollars'
]

MODELS_DIR = Path(r"D:\end-to-end-Invoice-Intelligence-System\models")
MODEL_PATH = MODELS_DIR / "random_forest_invoice_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler_invoice.pkl"


def load_model(model_path: str = MODEL_PATH):
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model


def load_scaler(scaler_path: str = SCALER_PATH):
    with open(scaler_path, "rb") as f:
        scaler = joblib.load(f)
    return scaler



def predict_invoice_flag(input_data):
    """
    Predict invoice flag using the training FEATURES and saved scaler+model.

    Accepts a single-record dict, dict-of-lists for multiple rows, or a DataFrame.
    """
    # Normalize input_data into a DataFrame
    if isinstance(input_data, dict):
        if any(isinstance(v, (list, tuple, pd.Series)) for v in input_data.values()):
            input_df = pd.DataFrame(input_data)
        else:
            input_df = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        input_df = input_data.copy()
    else:
        input_df = pd.DataFrame(input_data)

    # Ensure all required feature columns exist (fill missing with 0)
    for feat in FEATURES:
        if feat not in input_df.columns:
            input_df[feat] = 0

    # Select and coerce feature columns
    X = input_df[FEATURES].apply(pd.to_numeric, errors="coerce")

    # If model missing, show prepared input and raise a clear error
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}. Prepared input:\n{X}")

    model = load_model()

    # If scaler exists, use it to transform features (the model was trained on scaled data)
    if SCALER_PATH.exists():
        scaler = load_scaler()
        X_in = scaler.transform(X)
    else:
        X_in = X

    preds = model.predict(X_in)

    result = input_df.copy()
    result["Predicted_Flag"] = preds
    return result


if __name__ == "__main__":
    # Example inputs matching training FEATURES
    sample_input = {
        "invoice_quantity": [10],
        "invoice_dollars": [1500],
        "Freight": [50],
        "total_quantity": [10],
        "total_items_dollars": [1500],
    }

    try:
        print(predict_invoice_flag(sample_input))
    except FileNotFoundError as e:
        print(e)
