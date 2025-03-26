from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
from model_def import NeuralNet

app = FastAPI()

model = NeuralNet(input_dim=11)
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

class WineInput(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

@app.post("/predict")
def predict(input: WineInput):
    features = np.array([[v for v in input.dict().values()]], dtype=np.float32)
    with torch.no_grad():
        prediction = model(torch.from_numpy(features)).item()
    return {"predicted_quality": round(prediction, 2)}
