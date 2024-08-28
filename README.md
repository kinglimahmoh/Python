import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier


# Example dataset (replace with your own data)
X = np.array(...)  # Features
y = np.array(...)  # Target

# Step 1: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Set up k-fold cross-validation
k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Example model (replace with your own model)
model = RandomForestClassifier()

# Perform cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')

# Output results
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Score: {np.mean(cv_scores)}")
print(f"Standard Deviation of CV Scores: {np.std(cv_scores)}")

# Optionally, train the model on the entire training set and evaluate on the test set
model.fit(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Test Set Score: {test_score}")




import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Data for different algorithms
algorithms = ['Linear Regression', 'Decision Trees', 'Random Forests', 'Neural Networks', 'Proposed DSS']
accuracy = [0.75, 0.80, 0.82, 0.83, 0.85]
precision = [0.70, 0.75, 0.78, 0.80, 0.88]
recall = [0.72, 0.76, 0.79, 0.81, 0.85]
f1_scores = [0.71, 0.755, 0.785, 0.805, 0.865]
mae = [5.0, 4.5, 4.2, 4.0, 3.5]
rmse = [5.5, 5.0, 4.8, 4.5, 4.2]

# Creating the figures

# Accuracy Comparison
plt.figure(figsize=(10, 6))
plt.bar(algorithms, accuracy, color='skyblue')
plt.title('Figure 1: Accuracy Comparison of Different Algorithms')
plt.ylabel('Accuracy')
plt.savefig('Accuracy_Comparison.png')
plt.show()

# Precision and Recall Comparison
plt.figure(figsize=(10, 6))
plt.plot(algorithms, precision, marker='o', label='Precision', color='g')
plt.plot(algorithms, recall, marker='o', label='Recall', color='b')
plt.title('Figure 2: Precision and Recall Comparison of Different Algorithms')
plt.ylabel('Score')
plt.legend()
plt.savefig('Precision_Recall_Comparison.png')
plt.show()

# F1 Score Comparison
plt.figure(figsize=(10, 6))
plt.bar(algorithms, f1_scores, color='orange')
plt.title('Figure 3: F1 Score Comparison of Different Algorithms')
plt.ylabel('F1 Score')
plt.savefig('F1_Score_Comparison.png')
plt.show()

# MAE and RMSE Comparison
plt.figure(figsize=(10, 6))
plt.plot(algorithms, mae, marker='o', label='MAE', color='r')
plt.plot(algorithms, rmse, marker='o', label='RMSE', color='purple')
plt.title('Figure 4: MAE and RMSE Comparison of Different Algorithms')
plt.ylabel('Error')
plt.legend()
plt.savefig('MAE_RMSE_Comparison.png')
plt.show()

# Confusion Matrix for the Proposed DSS
# Simulating a confusion matrix
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
y_pred = [1, 0, 1, 1, 0, 0, 0, 1, 1, 1]
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Figure 5: Confusion Matrix for the Proposed DSS')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('Confusion_Matrix.png')
plt.show()
