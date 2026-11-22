import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def load_data():
    iris = load_iris(as_frame=True)

    X = iris.data
    y = iris.target

    return X, y

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
            )

    return X_train, X_test, y_train, y_test

def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Training features:", X_train.shape)
    print("Testing features:", X_test.shape)

    print("\nTraining labels:")
    print(y_train.value_counts())

    print("\nTesting labels:")
    print(y_test.value_counts())

if __name__ == "__main__":
    main()
