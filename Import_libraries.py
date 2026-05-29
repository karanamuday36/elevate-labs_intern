
# Data Cleaning & Preprocessing



# Dataset: Titanic Dataset

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
# 1. Load Dataset

# Read CSV file
df = pd.read_csv("Titanic-Dataset.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# 2. Explore Dataset

# Dataset information
print("\nDataset Info:")
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# 3. Handle Missing Values

# Fill missing Age values with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Fare values with median
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Fill missing Embarked values with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin column due to too many missing values
if 'Cabin' in df.columns:
    df.drop(columns=['Cabin'], inplace=True)

# Verify missing values again
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# 4. Encode Categorical Features

# Convert Sex column into numerical values
label_encoder = LabelEncoder()

df['Sex'] = label_encoder.fit_transform(df['Sex'])

# One-Hot Encoding for Embarked column
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

print("\nDataset After Encoding:")
print(df.head())

# 5. Standardize Numerical Features

# Select numerical columns
numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch']

# Initialize scaler
scaler = StandardScaler()

# Apply standardization
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

print("\nStandardized Features:")
print(df[numerical_cols].head())

# 6. Visualize Outliers

plt.figure(figsize=(10, 5))

sns.boxplot(data=df[numerical_cols])

plt.title("Boxplot of Numerical Features")

plt.show()

# 7. Remove Outliers Using IQR Method
# Calculate Q1 and Q3
Q1 = df[numerical_cols].quantile(0.25)
Q3 = df[numerical_cols].quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1

# Remove outliers
df_clean = df[
    ~(
        (
            (df[numerical_cols] < (Q1 - 1.5 * IQR)) |
            (df[numerical_cols] > (Q3 + 1.5 * IQR))
        ).any(axis=1)
    )
]

print("\nOriginal Dataset Shape:", df.shape)

print("Dataset Shape After Removing Outliers:", df_clean.shape)

# 8. Save Cleaned Dataset

df_clean.to_csv("Cleaned_Titanic_Dataset.csv", index=False)

print("\nCleaned dataset saved successfully!")