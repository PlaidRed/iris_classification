import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import (
        train_test_split,
        StratifiedKFold,
        cross_val_score,
        GridSearchCV
        )
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix
        )

def load_data():
    iris = load_iris(as_frame=True)

    X = iris.data
    y = iris.target

    return X, y

def create_model(k=5):
    model = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=k)
            )
    return model

def tune_model(X_train, y_train):
    pipeline = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier()
            )

    param_grid = {
            "kneighborsclassifier__n_neighbors": [
                1, 3, 5, 7, 9, 11, 13, 15
                ],
            "kneighborsclassifier__weights": [
                "uniform",
                "distance"
                ],
            "kneighborsclassifier__metric": [
                "euclidean",
                "manhattan"
                ]
            }

    cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
            )

    grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            return_train_score=True
            )

    grid_search.fit(X_train, y_train)

    return grid_search

def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
            )

    print("Starting hyperparameter search...")

    grid_search = tune_model(X_train, y_train)

    print("\nBest hyperparameters:")
    print(grid_search.best_params_)

    print(
            f"\nBest cross-validation accuracy: "
            f"{grid_search.best_score_:.2%}"
            )

    # GridSearchCV refits the best pipeline on all training data.
    best_model = grid_search.best_estimator_

    # Evaluate the selected model on the untouched test set.
    predictions = best_model.predict(X_test)

    print(
            f"\nFinal test accuracy: "
            f"{accuracy_score(y_test, predictions):.2%}"
            )

    print("\nClassification report:")
    print(
            classification_report(
                y_test,
                predictions,
                target_names=["Setosa", "Versicolor", "Virginica"]
                )
            )

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    # Display the ten best configurations from cross-validation.
    results = pd.DataFrame(grid_search.cv_results_)

    results = results.sort_values(
            by="rank_test_score"
            )

    print("\nTop 10 configurations:")
    print(
            results[
                [
                    "params",
                    "mean_test_score",
                    "std_test_score",
                    "rank_test_score"
                    ]
                ].head(10).to_string(index=False)
            )

if __name__ == "__main__":
    main()
