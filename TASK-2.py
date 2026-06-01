import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
df = pd.read_csv("dataset.csv")

# 2. Basic information
print(df.shape)
print(df.info())
print(df.isnull().sum())

# 3. Summary statistics
print(df.describe())
print(df.median(numeric_only=True))

# 4. Select numeric columns
num_cols = df.select_dtypes(include=np.number).columns

# 5. Histograms
for col in num_cols:
    plt.figure(figsize=(6,4))
    plt.hist(df[col].dropna(), bins=20)
    plt.title(f"Histogram - {col}")
    plt.show()

# 6. Boxplots
for col in num_cols:
    plt.figure(figsize=(6,3))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot - {col}")
    plt.show()

# 7. Correlation matrix
corr = df[num_cols].corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# 8. Pairplot
sns.pairplot(df[num_cols])
plt.show()

# 9. Outlier detection
for col in num_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}: {len(outliers)} outliers")

# 10. Feature-level inference
for col in num_cols:
    print(f"\n{col}")
    print("Mean:", df[col].mean())
    print("Median:", df[col].median())
    print("Std:", df[col].std())
    
"""
1. What is the purpose of EDA?
Exploratory Data Analysis (EDA) is used to understand a dataset before building machine learning models. 
It helps identify patterns, trends, outliers, missing values, and relationships between variables, 
leading to better data preprocessing and model selection.

2. How do boxplots help in understanding a dataset?
Boxplots visually summarize the distribution of data using:
Minimum value
First Quartile (Q1)
Median (Q2)
Third Quartile (Q3)
Maximum value
They are useful for:
Detecting outliers
Understanding data spread
Comparing distributions across groups
Identifying skewness

3. What is correlation and why is it useful?
Correlation measures the strength and direction of the relationship between two variables.
Positive Correlation (+1): Both variables increase together.
Negative Correlation (-1): One increases while the other decreases.
No Correlation (0): No relationship exists.
It is useful for:
Feature selection
Identifying dependencies
Detecting multicollinearity
Understanding data relationships

4. How do you detect skewness in data?
Skewness can be detected by:
Visualizing data using histograms or boxplots.
Calculating the skewness value:
Skewness ≈ 0 → Symmetric distribution
Skewness > 0 → Right-skewed (positive)
Skewness < 0 → Left-skewed (negative)
In Pandas:
Python
df['column_name'].skew()

5. What is multicollinearity?
Multicollinearity occurs when two or more independent variables are highly correlated with each other.
Problems:
Makes model coefficients unstable.
Reduces interpretability.
Can decrease model performance.
Detection methods:
Correlation Matrix
Variance Inflation Factor (VIF)

6. What tools do you use for EDA?
Common EDA tools include:
Pandas – Data manipulation and summary statistics.
NumPy – Numerical computations.
Matplotlib – Basic visualizations.
Seaborn – Statistical visualizations.
Plotly – Interactive charts.
Jupyter Notebook – Analysis environment.

7. Can you explain a time when EDA helped you find a problem?
Example Answer:
"While analyzing a customer dataset, I used EDA to examine missing values and distributions. I discovered that one feature had more than 40% missing values and another contained several extreme outliers. After handling the missing data and treating the outliers, the machine learning model's accuracy improved significantly. This showed how EDA helps identify data quality issues before modeling."

8. What is the role of visualization in ML?
Visualization helps:
Understand data distributions.
Identify trends and patterns.
Detect outliers and anomalies.
Analyze feature relationships.
Communicate insights effectively.
Support feature engineering and model evaluation.
Examples include histograms, scatter plots, heatmaps, boxplots, and pair plots.
"""

