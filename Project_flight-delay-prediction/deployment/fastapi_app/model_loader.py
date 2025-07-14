import joblib
import pandas as pd

def load_model():
    model = joblib.load("random_forest_model.pkl") 
    return model

def predict(model, input_data: dict):
    df = pd.DataFrame([input_data])
    return int(model.predict(df)[0])
