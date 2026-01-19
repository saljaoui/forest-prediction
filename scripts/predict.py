import sys
import os
import pickle
import pandas as pd

# Fix imports when running: python3 scripts/predict.py
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scripts.preprocessing_feature_engineering import load_test

MODEL_PATH = "results/best_model.pkl"
TEST_PATH = "data/test.csv"
OUT_PATH = "results/test_predictions.csv"


def main():
    # Load model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Load test + features
    df_test = load_test(TEST_PATH)

    # Keep Id if exists
    id_col = "Id" if "Id" in df_test.columns else None

    X_test = df_test.copy()
    if id_col:
        X_test = X_test.drop(columns=[id_col])

    # Safety: never keep Cover_Type in test features
    if "Cover_Type" in X_test.columns:
        X_test = X_test.drop(columns=["Cover_Type"])

    # Predict
    preds = model.predict(X_test)

    # Save predictions
    out_df = pd.DataFrame({"Cover_Type": preds})
    if id_col:
        out_df.insert(0, "Id", df_test[id_col].values)

    os.makedirs("results", exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)

    print(f"Saved predictions to: {OUT_PATH}")
    print(out_df.head())


if __name__ == "__main__":
    main()
