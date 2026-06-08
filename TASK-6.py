import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# load dataset
df = pd.read_excel("Titanic-Dataset.xlsx")

# Features and target
X = df[['Age', 'Fare']]
y = df['Survived']

# Handle missing values
X = SimpleImputer(strategy='median').fit_transform(X)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# experiment with different K values
for k in range(1, 16):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(k, accuracy_score(y_test, y_pred))

#Best model
knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#---
"""  1. How does the KNN algorithm work?

KNN classifies a new data point based on the majority class of its K nearest neighbors.

2. How do you choose the right K?

Try different K values and select the one that gives the best validation/test accuracy. Odd values are often used to avoid ties.

3. Why is normalization important in KNN?

KNN uses distance calculations. Normalization ensures that features with larger scales do not dominate the distance metric.

4. What is the time complexity of KNN?
Training: O(1) (stores data only)
Prediction: O(n × d)
n = number of training samples
d = number of features
5. What are the pros and cons of KNN?

Pros:

Simple and easy to understand
No training phase
Works well for small datasets

Cons:

Slow for large datasets
Sensitive to irrelevant features
Requires feature scaling
6. Is KNN sensitive to noise?

Yes. Noisy or mislabeled data can affect nearest neighbors and reduce accuracy, especially for small K values.

7. How does KNN handle multi-class problems?

It assigns the class that appears most frequently among the K nearest neighbors, regardless of the number of classes.

8. What’s the role of distance metrics in KNN?

Distance metrics determine how "closeness" is measured between data points. Common metrics include:

Euclidean Distance (most common)
Manhattan Distance
Minkowski Distance

The choice of metric can significantly affect model performance.
"""
