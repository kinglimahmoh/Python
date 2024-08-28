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
