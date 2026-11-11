import pandas as pd
from sklearn.datasets import load_iris

def main():
    # Load the iris dataset
    iris = load_iris(as_frame=True)

    # Separate features and target

    x = iris.data
    y = iris.target

    # Create a DataFrame
    df = x.copy()

    # Convert numerical labels into species names
    df["species"] = y.map(
            dict(enumerate(iris.target_names))
    )

    print("First ten samples:")
    print(df.head(10))

    print("\nColumns:")
    col = df.columns
    print(col)

    print("\nAverage petal length for each species:")
    average = df.groupby("species")["petal length (cm)"].mean()
    print(average)

    print("\nDataset dimensions:")
    print(df.shape)

    print("\nSpecies distribution:")
    print(df["species"].value_counts())

    print("\nStatistical summary:")
    print(x.describe())

if __name__ == "__main__":
    main()


