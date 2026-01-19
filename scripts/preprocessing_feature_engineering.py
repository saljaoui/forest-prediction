import pandas as pd
import numpy as np

TARGET_COL = "Cover_Type"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering (نفس الشي كيطبق على train و test)
    """

    # Distance to hydrology (hint)
    df["Distance_To_Hydrology"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2
        + df["Vertical_Distance_To_Hydrology"] ** 2
    )

    # Fire points - roadways (hint)
    df["Fire_Road_Distance_Diff"] = (
        df["Horizontal_Distance_To_Fire_Points"]
        - df["Horizontal_Distance_To_Roadways"]
    )

    return df


def load_train(path: str):
    """
    Load train.csv and return X, y (target removed from X)
    """
    df = pd.read_csv(path)
    df = add_features(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def load_test(path: str) -> pd.DataFrame:
    """
    Load test.csv and return features dataframe.
    If Cover_Type exists (sometimes), remove it to avoid feature mismatch.
    """
    df = pd.read_csv(path)
    df = add_features(df)

    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])

    return df
