# ---------------------------------------------
# Assignment: Data Analysis and Visualization
# ---------------------------------------------
# Objective:
# - Load and analyze a dataset using pandas.
# - Create simple plots and charts with matplotlib for visualization.
# ---------------------------------------------

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# ---------------------------------------------
# Task 1: Load and Explore the Dataset
# ---------------------------------------------
# Using the Iris dataset
iris_data = load_iris()
df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
df['species'] = pd.Categorical.from_codes(iris_data.target, iris_data.target_names)

# Display first few rows
print("First five rows of the dataset:")
display(df.head())

# Check data types and missing values
print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

# Clean dataset (no missing values in this dataset)
# If missing: df = df.dropna() or df.fillna(...)

# ---------------------------------------------
# Task 2: Basic Data Analysis
# ---------------------------------------------
print("\nBasic statistics:")
display(df.describe())

# Group by species and compute mean
grouped_means = df.groupby('species').mean()
print("\nMean values per species:")
display(grouped_means)

# Observations / Findings
print("\nObservation:")
print("- Setosa species have significantly smaller petal length and width compared to Versicolor and Virginica.")

# ---------------------------------------------
# Task 3: Data Visualization
# ---------------------------------------------
sns.set(style="whitegrid")

# 1. Line Chart (simulating a trend)
plt.figure(figsize=(8,4))
plt.plot(df.index, df['sepal length (cm)'], label='Sepal Length')
plt.title('Trend of Sepal Length over Samples')
plt.xlabel('Sample Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.show()

# 2. Bar Chart: Average petal length per species
plt.figure(figsize=(6,4))
sns.barplot(x='species', y='petal length (cm)', data=df, ci=None)
plt.title('Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.show()

# 3. Histogram: Distribution of sepal length
plt.figure(figsize=(6,4))
plt.hist(df['sepal length (cm)'], bins=10, edgecolor='black')
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# 4. Scatter Plot: Sepal length vs. Petal length
plt.figure(figsize=(6,4))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
plt.title('Sepal Length vs. Petal Length by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()

# ---------------------------------------------
# Findings / Observations
# ---------------------------------------------
print("\nKey Findings:")
print("1. Setosa species have distinctly smaller petals compared to other species.")
print("2. Versicolor and Virginica overlap in some measurements but still form identifiable clusters.")
print("3. Sepal length tends to increase with petal length, suggesting a possible relationship useful for classification.")
