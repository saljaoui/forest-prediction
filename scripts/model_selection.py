import sys
import os

# Fix imports when running: python3 scripts/model_selection.py
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from scripts.preprocessing_feature_engineering import load_train


def save_confusion_matrix_table(cm: np.ndarray, labels: np.ndarray, out_path: str):
    """
    Save confusion matrix as a TABLE image (like the audit example)
    """
    plt.figure(figsize=(12, 5))
    ax = plt.gca()
    ax.axis("off")

    # Create table
    table = ax.table(
        cellText=cm,
        rowLabels=[str(l) for l in labels],
        colLabels=[str(l) for l in labels],
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Titles like example
    plt.title("Predicted Label", pad=20, fontsize=14)
    # Left label (True Label)
    plt.text(
        -0.02, 0.5, "True Label",
        rotation=90, va="center", ha="center",
        fontsize=12, transform=ax.transAxes
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_learning_curve(best_model, X_train, y_train, cv, out_path: str):
    train_sizes, train_scores, val_scores = learning_curve(
        best_model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=-1
    )

    # Means + std (باش يكون بحال المثال وفيه shading)
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    plt.figure(figsize=(8, 5))

    # Shaded area for CV
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)

    plt.plot(train_sizes, train_mean, marker="o", linewidth=2, label="Training score")
    plt.plot(train_sizes, val_mean, marker="o", linewidth=2, label="Cross-validation score")

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.title("Learning Curves of Best Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    # Prepare folders
    os.makedirs("results", exist_ok=True)

    # Load data
    X, y = load_train("data/train.csv")

    # Split Train(1) / Test0(1) from Train file (0)
    X_train, X_test0, y_train, y_test0 = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5-Fold CV (stratified)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 5 models required + grids (small grids)
    models = {
        "LogisticRegression": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000))
            ]),
            {"model__C": [0.1, 1.0]}
        ),
        "KNN": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier())
            ]),
            {"model__n_neighbors": [5, 7]}
        ),
        "SVM": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVC())
            ]),
            {"model__C": [1], "model__kernel": ["rbf"]}
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=42),
            {"n_estimators": [100], "max_depth": [None, 20]}
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=42),
            {"n_estimators": [100], "learning_rate": [0.1]}
        ),
    }

    # GridSearch on each model
    best_model = None
    best_name = None
    best_cv_score = -1

    for name, (model, params) in models.items():
        print(f"\n[GridSearch] Model: {name}")

        grid = GridSearchCV(
            estimator=model,
            param_grid=params,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1
        )
        grid.fit(X_train, y_train)

        print("  Best params:", grid.best_params_)
        print("  Best CV score:", grid.best_score_)

        if grid.best_score_ > best_cv_score:
            best_cv_score = grid.best_score_
            best_model = grid.best_estimator_
            best_name = name

    # Final evaluation
    y_pred_train = best_model.predict(X_train)
    y_pred_test0 = best_model.predict(X_test0)

    train_acc = accuracy_score(y_train, y_pred_train)
    test0_acc = accuracy_score(y_test0, y_pred_test0)

    print("\n================ AUDIT RESULTS ================")
    print("Best model:", best_name)
    print("Train accuracy (should be < 0.98):", train_acc)
    print("Test0 accuracy (target > 0.65):", test0_acc)
    print("Best CV accuracy:", best_cv_score)
    print("================================================\n")

    # Confusion Matrix (DataFrame with correct names)
    labels = np.sort(y.unique())
    cm = confusion_matrix(y_test0, y_pred_test0, labels=labels)

    cm_df = pd.DataFrame(
        cm,
        index=pd.Index(labels, name="True label"),
        columns=pd.Index(labels, name="Predicted label")
    )

    print("Confusion matrix (DataFrame):")
    print(cm_df)

    # Save confusion matrix TABLE image (audit file name)
    save_confusion_matrix_table(
        cm=cm,
        labels=labels,
        out_path="results/confusion_matrix_heatmap.png"
    )

    # Save learning curve (audit file name)
    save_learning_curve(
        best_model=best_model,
        X_train=X_train,
        y_train=y_train,
        cv=cv,
        out_path="results/learning_curve_best_model.png"
    )

    # Save pickle
    with open("results/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    print("Saved files:")
    print("- results/best_model.pkl")
    print("- results/confusion_matrix_heatmap.png")
    print("- results/learning_curve_best_model.png")


if __name__ == "__main__":
    main()
