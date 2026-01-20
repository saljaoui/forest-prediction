import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, learning_curve
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from preprocessing_feature_engineering import get_processed_data

X, y = get_processed_data('data/train.csv')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

models = {
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'KNN': Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier())]),
    'RandomForest': RandomForestClassifier(random_state=42),
    'SVM': Pipeline([('scaler', StandardScaler()), ('svm', SVC(random_state=42))]),
    'LogisticRegression': Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=1000, random_state=42))])
}

param_grids = {
    'GradientBoosting': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]},
    'KNN': {'knn__n_neighbors': [3, 5, 7]},
    'RandomForest': {'n_estimators': [50, 100], 'max_depth': [10, None], 'min_samples_split': [2, 5]},
    'SVM': {'svm__C': [0.1, 1, 10], 'svm__kernel': ['linear', 'rbf']},
    'LogisticRegression': {'lr__C': [0.1, 1, 10], 'lr__class_weight': [None, 'balanced']}
}
print("Starting model selection...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_model = None
best_score = 0
best_name = None

for name, model in models.items():
    grid = GridSearchCV(model, param_grids[name], cv=cv, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    score = grid.best_score_
    print(f"{name} best CV score: {score}")
    if score > best_score:
        best_score = score
        best_model = grid.best_estimator_
        best_name = name

best_model.fit(X_train, y_train)

test_acc_internal = accuracy_score(y_test, best_model.predict(X_test))
print(f"Best model ({best_name}) internal test accuracy: {test_acc_internal}")

# best_model.fit(X, y)
train_acc = accuracy_score(y, best_model.predict(X))
print(f"Train accuracy on full train.csv: {train_acc}")

if train_acc >= 0.98:
    print("Warning: Train accuracy too high; consider more regularization.")

cm = confusion_matrix(y_test, best_model.predict(X_test))
classes = sorted(set(y))
cm_df = pd.DataFrame(cm, index=[f"True {c}" for c in classes], columns=[f"Predicted {c}" for c in classes])

print("Confusion Matrix:\n", cm_df)

train_sizes, train_scores, val_scores = learning_curve(best_model, X_train, y_train, cv=cv, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5))
os.makedirs("results/plots", exist_ok=True)
with open('results/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=0), label='Training score')
plt.plot(train_sizes, np.mean(val_scores, axis=0), label='Validation score')
plt.title('Learning Curve for Best Model')
plt.xlabel('Training examples')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.savefig('results/plots/learning_curve.png')


print("Model selection complete. Best model saved.")