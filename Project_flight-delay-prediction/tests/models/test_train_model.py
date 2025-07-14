import pandas as pd
import os
import pytest
from src.models.train_model import train_and_evaluate


def test_train_and_evaluate(tmp_path):
    
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5, 6],
        "feature2": [10, 20, 30, 40, 50, 60],
        "is_delay": [0, 1, 0, 1, 0, 1]
    })

    
    os.makedirs(tmp_path / "data" / "processed", exist_ok=True)
    train_path = tmp_path / "data" / "processed" / "train.csv"
    test_path = tmp_path / "data" / "processed" / "test.csv"
    df.to_csv(train_path, index=False)
    df.to_csv(test_path, index=False)

   
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    X_train = df_train.drop(columns=["is_delay"])
    y_train = df_train["is_delay"]
    X_test = df_test.drop(columns=["is_delay"])
    y_test = df_test["is_delay"]

   
    model = train_and_evaluate(X_train, y_train, X_test, y_test)

    assert model is not None
    assert hasattr(model, "predict")
    assert callable(model.predict)
