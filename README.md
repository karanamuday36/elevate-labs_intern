### 1. How does Logistic Regression differ from Linear Regression?

* **Linear Regression** predicts continuous values (e.g., salary, house price).
* **Logistic Regression** predicts categories (e.g., Yes/No, Survived/Not Survived).

---

### 2. What is the Sigmoid Function?

* A mathematical function used in Logistic Regression.
* Converts output into a probability between **0 and 1**.
* Helps classify data into two classes.

---

### 3. What is Precision vs Recall?

**Precision:**

* Out of all predicted positive cases, how many are actually positive.

**Recall:**

* Out of all actual positive cases, how many are correctly predicted.

---

### 4. What is the ROC-AUC Curve?

* ROC curve shows model performance at different thresholds.
* AUC (Area Under Curve) measures how well the model separates classes.
* Higher AUC means better performance.

---

### 5. What is the Confusion Matrix?

* A table used to evaluate classification models.
* Shows:

  * True Positives (TP)
  * True Negatives (TN)
  * False Positives (FP)
  * False Negatives (FN)

---

### 6. What Happens if Classes are Imbalanced?

* Model may favor the majority class.
* Accuracy can be misleading.
* Precision, Recall, and ROC-AUC become more important.

---

### 7. How Do You Choose the Threshold?

* Default threshold is **0.5**.
* Lower threshold → Higher Recall.
* Higher threshold → Higher Precision.
* Choose based on project requirements.

---

### 8. Can Logistic Regression Be Used for Multi-Class Problems?

* **Yes.**
* Use techniques like:

  * One-vs-Rest (OvR)
  * Multinomial Logistic Regression
* It can classify data into more than two classes.
