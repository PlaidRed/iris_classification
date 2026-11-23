import joblib

from tune_knn import load_data, tune_model
from sklearn.model_selection import train_test_split

def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
            )

    print("Training and tuning model...")

    grid_search = tune_model(X_train, y_train)

    best_model = grid_search.best_estimator_

    print("\nBest parameters:")
    print(grid_search.best_params_)

    print(f"\nCross-validation accuracy: {grid_search.best_score_:.2%}")

    joblib.dump(best_model, "iris_model.joblib")

    print("\nModel saved to iris_model.joblib")

if __name__ == "__main__":
    main()
