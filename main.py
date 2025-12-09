# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Piezoelectric Tile ML API")

# Allow frontend apps to access this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and data
model = joblib.load("piezo_model.pkl")
data_df = pd.read_csv("power_tile_data.csv")
data_df.columns = ["voltage", "current_uA", "weight_kg", "step_location", "power_mW"]

# ---------- Schemas ----------

class TileInput(BaseModel):
    voltage: float
    current_uA: float
    weight_kg: float
    step_location: str  # "Center", "Edge", "Corner"

class TilePrediction(BaseModel):
    predicted_power_mW: float

# ---------- Routes ----------

@app.get("/")
def root():
    return {"message": "Piezoelectric Tile API is running"}

@app.post("/predict", response_model=TilePrediction)
def predict_tile(inp: TileInput):
    row = {
        "voltage": inp.voltage,
        "current_uA": inp.current_uA,
        "weight_kg": inp.weight_kg,
        "step_location": inp.step_location,
    }
    X = pd.DataFrame([row])
    y_pred = model.predict(X)[0]
    return {"predicted_power_mW": float(y_pred)}

@app.get("/data")
def get_data():
    # For graphs on dashboard
    return data_df.to_dict(orient="records")

@app.get("/stats")
def get_stats():
    return {
        "count": int(len(data_df)),
        "avg_power": float(data_df["power_mW"].mean()),
        "max_power": float(data_df["power_mW"].max()),
        "min_power": float(data_df["power_mW"].min()),
    }

import os

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow your Netlify domain
origins = [
    "http://localhost:5173",  # local dev
    "https://6937b24bbad323d1d072f514--shiny-sunflower-ebdbcf.netlify.app/"  # replace with your site
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
