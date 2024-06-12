# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris_df = pd.read_csv(url, names=names)

# Display the first few rows of the dataset
print(iris_df.head())

# Summary statistics
print(iris_df.describe())

# Data visualization

# Pair plot
sns.pairplot(iris_df, hue='species', markers=['o', 's', 'D'])
plt.show()

# Box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=iris_df.drop('species', axis=1))
plt.title('Box Plot of Iris Features')
plt.xlabel('Features')
plt.ylabel('Length/Width (cm)')
plt.show()

# Correlation analysis
correlation_matrix = iris_df.drop('species', axis=1).corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Iris Features')
plt.show()

# Outlier detection
plt.figure(figsize=(10, 6))
sns.boxplot(data=iris_df.drop('species', axis=1))
plt.title('Box Plot of Iris Features')
plt.xlabel('Features')
plt.ylabel('Length/Width (cm)')
plt.show()
