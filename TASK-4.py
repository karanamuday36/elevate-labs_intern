import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt

# 1. Load_dataset
df = pd.read_excel("dataset.xlsx")  # or your file name

# 2. Select features and target
features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
df = df[features + ["Survived"]]

# 3. Handle missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df = df.dropna(subset=["Survived"])

X = df[features]
y = df["Survived"]

# 4. Train test   split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5.Standardize features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6.Train Logistic Regression
model = LogisticRegression()

model.fit(X_train_scaled, y_train)

# 7. Predicted probabilities
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# 8. Default threshold = 0.5
y_pred = (y_prob >= 0.5).astype(int)

# 9. Evaluationn metrics
cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("Confusion Matrix:\n", cm)
print("Precision:", round(precision, 3))
print("Recall:", round(recall, 3))
print("ROC-AUC:", round(roc_auc, 3))

# 10. ROc Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# 11.Threshold tuning
threshold = 0.3

y_pred_new = (y_prob >= threshold).astype(int)

print("\nThreshold =", threshold)
print("Precision:", precision_score(y_test, y_pred_new))
print("Recall:", recall_score(y_test, y_pred_new))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_new))


##---
