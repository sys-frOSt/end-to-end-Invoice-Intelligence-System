import joblib
import pandas as pd


MODEL_PATH = r"D:\end-to-end-Invoice-Intelligence-System\models\predict_freight_model.pkl"


def load_model(model_path: str=MODEL_PATH):
    model = joblib.load(model_path)
    return model


def predict_freight_cost(input_data):
    model = load_model()
    # Normalize input_data into a DataFrame:
    if isinstance(input_data, dict):
        # If any value is list-like, create DataFrame directly from the dict
        if any(isinstance(v, (list, tuple)) for v in input_data.values()):
            input_df = pd.DataFrame(input_data)
        else:
            input_df = pd.DataFrame([input_data])
    else:
        input_df = pd.DataFrame(input_data)

    preds = model.predict(input_df)
    input_df['Predicted_Freight_Cost'] = pd.Series(preds).round().astype(int)
    return input_df


if __name__ == "__main__":
    sample_input = {
        "Dollars": [18500, 9000]
    }

    predictions = predict_freight_cost(sample_input)
    print(predictions)