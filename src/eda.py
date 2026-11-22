import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.datasets import load_iris

def load_data():
    iris = load_iris(as_frame=True)

    df = iris.data.copy()

    df["species"] = iris.target.map(
             dict(enumerate(iris.target_names))
             )
    return df

def plot_distributions(df):
    features = df.select_dtypes(include="number").columns

    fig, axes = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=(12, 8)
            )

    axes = axes.flatten()

    for i, feature in enumerate(features):
        sns.histplot(
                data=df,
                x=feature,
                kde=True,
                ax=axes[i]
                )

        axes[i].set_title(f"Distribution of {feature}")

    plt.tight_layout()
    plt.show()

def plot_species_comparison(df):
    features = df.select_dtypes(include="number").columns

    fig, axes = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=(12, 8)
            )
    axes = axes.flatten()

    for i, feature in enumerate(features):
        sns.boxplot(
                data=df,
                x="species",
                y=feature,
                ax=axes[i]
                )
        axes[i].set_title(f"{feature} by species")

    plt.tight_layout()
    plt.show()

def plot_correlation(df):
    correlation = df.select_dtypes(
            include="number"
            ).corr()
    plt.figure(figsize=(9, 7))

    sns.heatmap(
            correlation,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5
            )

    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.show()

def plot_pairplot(df):
    sns.pairplot(
            df,
            vars=[
                "petal length (cm)",
                "petal width (cm)"
                ],
            hue="species",
            diag_kind="hist",
            corner=True
            )
    plt.show()

def main():
    df = load_data()

    print("Dataset information:")
    df.info()

    print("\nDescriptive statistics:")
    print(df.describe())

    print("\nStatistics groped by species:")
    print(df.groupby("species").mean())


    print("\nDistribution of each measurement.")
    # plot_distributions(df)

    print("Comparison between species:")
    # plot_species_comparison(df)

    print("Feature correlations:")
    #plot_correlation(df)

    print("Comparison of every numerical feature against every other numerical feature:")
    #plot_pairplot(df)

    # Generate a histogram for petal length, separating the three species using hue.
    sns.histplot(data=df, x="petal length (cm)", hue="species",
                 kde=True, alpha=0.5
                 )
    plt.show()

    # Create a boxplot comparing sepal width between species.
    sns.boxplot(data=df, x="species", y="sepal width (cm)")
    plt.title("Sepal width by species")
    plt.show()

    # The two most strongly correlated numerical features.
    # calculate the correlation Matrix
    correlation = df.select_dtypes(
            include="number"
            ).corr()

    # Keep only the upper triangle, excluding the diagonal
    upper_triangle = correlation.where(
            np.triu(np.ones(correlation.shape), k=1).astype(bool)
            )

    # Convert the matrix into feature pairs
    correlation_pairs = upper_triangle.stack()
    strongest_pair = correlation_pairs.abs().idxmax()

    # Find the strongest correlation
    strongest_pair = correlation_pairs.abs().idxmax()

    print("\nStrongest correlation:")
    print(f"Features: {strongest_pair}")
    print(f"Correlation: {correlation_pairs.loc[strongest_pair]:.4f}")

    # use groupby() to calculate the standard deviaton of every feature for each species.
    print("\nStandard deviation grouped by species:")
    print(df.groupby("species").std())

    # Modify the pairplot to display only length and petal width.
    print("\nPairplot:")
    plot_pairplot(df)

if __name__ == "__main__":
    main()
