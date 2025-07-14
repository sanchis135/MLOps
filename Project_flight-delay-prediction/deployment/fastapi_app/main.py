from fastapi import FastAPI
from pydantic import BaseModel
from model_loader import load_model, predict
import uvicorn

app = FastAPI()
model = load_model()

class FlightFeatures(BaseModel):
    month: int
    day: int
    day_of_week: int
    dep_hour: int
    airline: str
    origin: str
    destination: str
    distance: float

@app.post("/predict")
def predict_delay(features: FlightFeatures):
    input_data = features.dict()
    prediction = predict(model, input_data)
    return {"delay_prediction": prediction}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)