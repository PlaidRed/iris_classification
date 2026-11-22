import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, f1_score

def load_data():
    iris = load_iris(as_frame=True)

    X = iris.data
    y = iris.target

    return X, y

def get_models():
    models = {
            "KNN": make_pipeline(
                StandardScaler(),
                KNeighborsClassifier(n_neighbors=5)
                ),
            "Logistic Regression": make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000)
                ),
            "Decision Tree": DecisionTreeClassifier(
                random_state=42
                ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                random_state=42
                ),
            "SVM": make_pipeline(
                StandardScaler(),
                SVC()
                )
            }

    return models

def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
            )

    models = get_models()

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average="macro")

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "F1-score": f1
            })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="Accuracy",
        ascending=False
    )

    print("\nModel comparison:")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()
