from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pickle
import joblib
import numpy as np

app = FastAPI()

# Load the pre-trained model from the pickle file
try:
    disorder_identification_model = joblib.load("disorder_identification_model.joblib")
    if not hasattr(disorder_identification_model, 'predict'):
        raise ValueError("Loaded object is not a model with a 'predict' method")
except Exception as e:
    raise RuntimeError(f"Failed to load the model: {str(e)}")

#API input param types
class DisorderPredictApiDataRequestModel(BaseModel):
    features: List[int]

class AnalyzeTextApiDataRequestModel(BaseModel):
    features: str


#API definitions
@app.post("/predict_disorder")
def predict_disorder(data: DisorderPredictApiDataRequestModel):
    input_data = np.array(data.features).reshape(1, -1)
    print(input_data)
    try:
        prediction = disorder_identification_model.predict(input_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"prediction": prediction.tolist()}

# @app.post("/analyze_text")
# def analyze_text(data:AnalyzeTextApiDataRequestModel):
