from fastapi import FastAPI
import torch
from pydantic import BaseModel
import numpy as np

# Define a Pydantic model for the input data
class InputData(BaseModel):
    data: List[float]

app = FastAPI()

# Load the model
model = torch.load("model.pth")

@app.post("/predict")
async def predict(input_data: InputData):
    # Convert the input data to a PyTorch tensor
    data = torch.tensor(input_data.data, dtype=torch.float32).unsqueeze(0)
    
    # Pass the data to the model
    with torch.no_grad():
        output = model(data)
    
    # Convert the output to a list and return it
    return output.tolist()