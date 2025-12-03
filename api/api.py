from fastapi import FastAPI
import pickle
import pandas as pd
import os

app = FastAPI()



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_pipeline.pkl")

model = pickle.load(open(MODEL_PATH, "rb"))


@app.post("/predict")
def predict(input_data: dict):
    df = pd.DataFrame([input_data])
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]
    return {"prediction": int(pred), "probability": float(proba)}
