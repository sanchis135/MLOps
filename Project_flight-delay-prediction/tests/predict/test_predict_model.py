from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from src.predict.predict_model import predict

def test_predict():
    model = RandomForestClassifier()
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4],
        "feature2": [10, 20, 30, 40]
    })
    y = [0, 1, 0, 1]
    model.fit(df, y)

   
    model_path = Path("tests/tmp_test_model.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    
    loaded_model = joblib.load(model_path)
    predictions = predict(loaded_model, df)

    assert len(predictions) == len(df)
    assert set(predictions).issubset({0, 1})
