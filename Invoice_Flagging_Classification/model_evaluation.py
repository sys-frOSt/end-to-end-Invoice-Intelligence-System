from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,make_scorer, f1_score


def train_random_forest(X_train, y_train):
    # Define the model
    rf = RandomForestClassifier(random_state=42)

    # Define the hyperparameter grid
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 4, 5, 6],
        "min_samples_split": [2, 3, 5],
        "min_samples_leaf": [1, 2, 5],
        "criterion": ["gini", "entropy"]
}

    # Create a custom scorer for F1 score
    f1_scorer = make_scorer(f1_score)

    # Set up GridSearchCV
    grid_search = GridSearchCV(estimator=rf,
                               param_grid=param_grid,
                               scoring=f1_scorer,
                               cv=5,
                               n_jobs=-1,
                               verbose=2)

    # Fit the model
    grid_search.fit(X_train, y_train)

    return grid_search



def evaluate_classifier(model, X_test, y_test, model_name):
    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)

    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.2f}")
    print(report)