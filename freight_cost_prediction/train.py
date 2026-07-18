import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error

from data_preprocessing import load_data_from_sqlite, preprocess_data, split_data
from model_evaluation import (
    train_linear_regression,
    train_decision_tree,
    train_random_forest,
    evaluate_model
)


def main():
    project_root = Path(__file__).resolve().parent.parent
    db_path = project_root / "data" / "inventory.db"
    # use repo-level models directory so all projects dump to the same parent
    model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_data_from_sqlite(db_path)
    
    # Prepare data
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train models
    lr_model = train_linear_regression(X_train, y_train)
    dt_model = train_decision_tree(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    
    results = []
    models = [
        (lr_model, "Linear Regression"),
        (dt_model, "Decision Tree Regression"),
        (rf_model, "Random Forest Regression")
    ]
    
    model_dict = {}
    for model, name in models:
        evaluate_model(model, X_test, y_test, name)
        mae = mean_absolute_error(y_test, model.predict(X_test))
        results.append({"model_name": name, "mae": mae})
        model_dict[name] = model
    
    # Select best model (lowest MAE)
    best_model_info = min(results, key=lambda x: x["mae"])
    best_model_name = best_model_info["model_name"]
    best_model = model_dict[best_model_name]
    
    # Save best model to the shared top-level models directory
    model_path = model_dir / "predict_freight_model.pkl"
    joblib.dump(best_model, model_path)
    
    print(f"\nBest model saved: {best_model_name}")
    print(f"Model path: {model_path}")


if __name__ == "__main__":
    main()
