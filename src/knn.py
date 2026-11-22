import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report)

def load_data():
    iris = load_iris(as_frame=True)

    X = iris.data.to_numpy()
    y = iris.target.to_numpy()

    return X, y

def prepare_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
            )

    scaler = StandardScaler()

    # Learn the scaling parameters from training data only.
    X_train = scaler.fit_transform(X_train)

    # Apply those same parameters to the test data.
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def evaluate_model(y_test, predictions):
    class_names = ["Setosa", "Versicolor", "Virginica"]

    accuracy = accuracy_score(y_test, predictions)

    print(f"\nAccuracy: {accuracy:.2%}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, predictions)
    print(cm)

    print("\nclassification_report:")
    print(
            classification_report(
                y_test,
                predictions,
                target_names=class_names
                )
            )
    return cm

class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def euclidean_distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def predict_one(self, sample):
        distances = []

        for train_sample in self.X_train:
            distance = self.euclidean_distance(sample, train_sample)
            distances.append(distance)

        distances = np.array(distances)

        nearest_indices = np.argsort(distances)[:self.k]

        nearest_labels = self.y_train[nearest_indices]

        prediction = np.bincount(nearest_labels).argmax()

        return prediction


    def predict(self, X):
        predictions = []

        for sample in X:
            predictions.append(self.predict_one(sample))

        return np.array(predictions)

def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test, scaler = prepare_data(X, y)

    # Create and train our classifier
    model = KNN(k=5)
    model.fit(X_train, y_train)

    # Predict the test train_samples
    predictions = model.predict(X_test)

    # Compare some predictions with their actual labels
    target_names = ["Setosa", "Versicolor", "Virginica"]

    print("First 10 predictions:")
    for i in range(10):
        predicted = target_names[predictions[i]]
        actual = target_names[y_test[i]]

        print(f"Predicted: {predicted:12} | Actual: {actual}")

    sklearn_model = KNeighborsClassifier(n_neighbors=5)

    sklearn_model.fit(X_train, y_train)

    sklearn_predictions = sklearn_model.predict(X_test)

    print("\nCustom KNN:")
    evaluate_model(y_test, predictions)

    print("\nScikit-learn KNN:")
    evaluate_model(y_test, sklearn_predictions)

if __name__ == "__main__":
        main()
