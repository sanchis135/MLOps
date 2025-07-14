import pandas as pd
import joblib
import os

def load_model(model_path: str):
    return joblib.load(model_path)

def predict(model, input_data: pd.DataFrame):
    return model.predict(input_data)

if __name__ == "__main__":
    # Paths
    model_path = "models/random_forest_model.pkl"
    input_path = "data/processed/test.csv"       
    output_path = "data/predictions/predictions.csv"

    # Load input data
    df = pd.read_csv(input_path)
    if "is_delay" in df.columns:
        df = df.drop(columns=["is_delay"])  

    # Load model
    model = load_model(model_path)

    # Predict
    predictions = predict(model, df)

    # Save predictions
    os.makedirs("data/predictions", exist_ok=True)
    df_preds = df.copy()
    df_preds["predicted_is_delay"] = predictions
    df_preds.to_csv(output_path, index=False)

    print(f"✅ Predictions saved to {output_path}")
