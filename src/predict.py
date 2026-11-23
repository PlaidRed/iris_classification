import joblib
import numpy as np

# Load the trained pipeline
model = joblib.load("iris_model.joblib")

species = [
        "Setosa",
        "Versicolor",
        "Virginica"
        ]

# Measurements in centimeters:
# sepal length, sepal width, petal length, petal width

flower = np.array([
    [5.1, 3.5, 1.4, 0.2]
    ])

prediction = model.predict(flower)

print(f"Predicted species: {species[prediction[0]]}")
