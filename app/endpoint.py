from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from typing import List
import os
import pickle

import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define a Pydantic model for the input data
class InputData(BaseModel):
    data: List[float]

app = FastAPI()

MLP_PATH = os.path.join('models', 'baseline_mlp.pth')
XGBR_PATH = os.path.join('models', 'baseline_xgbr.pkl')

MINMAX_SCALER_PATH = os.path.join('scalers', 'scaler_minmax.pkl')
# Load the model
model = torch.load(MLP_PATH)

@app.post("/predict_mlp")
async def predict(input_data: InputData):
    # Convert the input data to a PyTorch tensor
    data = torch.tensor(input_data.data, dtype=torch.float32).unsqueeze(0)
    
    # Pass the data to the model
    with torch.no_grad():
        output = model(data)
    
    # Convert the output to a list and return it
    return output.tolist()

@app.post("/predict_xgbr")
async def predict(input_data: InputData):
    # Load the model
    xgbr = pickle.load(open(XGBR_PATH, 'rb'))
    scaler = pickle.load(open(MINMAX_SCALER_PATH, 'rb'))

    # Convert the input data to a numpy array
    data = np.array(input_data.data).reshape(1, -1)

    # Scale the input data
    data = scaler.transform(data)

    # Pass the data to the model
    output = xgbr.predict(data)

    # Convert the output to a list and return it
    return output.tolist()