import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import yfinance as yf

app = FastAPI(
    title="Stock Price Prediction",
    description="Predict stock price using machine learning",
    version="0.1",
    contact={"name": "Luiz Monferrari"},
)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
