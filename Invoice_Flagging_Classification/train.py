from pathlib import Path
from data_preprocessing import load_invoice_data, apply_level, split_data, scale_data
from model_evaluation import train_random_forest, evaluate_classifier
import joblib
    
FEATURES = ['invoice_quantity', 'invoice_dollars', 'Freight', 'total_quantity', 'total_items_dollars']
TARGET = 'flag_invoice_risk'
    
    
def main():
    base_dir = Path(__file__).resolve().parents[1]  # d:\end-to-end-Invoice-Intelligence-System
    data_path = base_dir / 'data' / 'inventory.db'
    # explicit repo-level models directory
    models_dir = Path(r"D:\end-to-end-Invoice-Intelligence-System\models")
    models_dir.mkdir(parents=True, exist_ok=True)

    df = load_invoice_data(str(data_path))
    df = apply_level(df)

    X_train, X_test, y_train, y_test = split_data(df, FEATURES, TARGET)
    scaler_path = models_dir / 'scaler_invoice.pkl'
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test, str(scaler_path))

    grid_search = train_random_forest(X_train_scaled, y_train)
    best_model = getattr(grid_search, "best_estimator_", grid_search)

    evaluate_classifier(best_model, X_test_scaled, y_test, "Random Forest Classifier")

    joblib.dump(best_model, str(models_dir / 'random_forest_invoice_model.pkl'))


if __name__ == "__main__":
    main()