import uvicorn
import yfinance as yf
from classes.indicadores import calcula_indicadores

from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load


class Features(BaseModel):
    Model: str


app = FastAPI(
    title="API para previsão de BITCOIN",
    description="learning Previsão de preço de ações usando machine learning.",
    version="0.1",
    contact={"name": "Luiz Monferrari"},
)


@app.get("/")
def message():
    return "API para previsão de BITCOIN - Utilize o metodo correto."


@app.post("/predict")
async def predict(Features: Features):

    # obtendo os dados
    btc_ticker = yf.Ticker("BTC-USD")
    historical_values_btc = btc_ticker.history(period="200d", actions=False)
    historical_values_btc = historical_values_btc.tz_localize(None)

    # calculo de indicadores
    historical_values_btc = calcula_indicadores(historical_values_btc).sort_index(
        ascending=False
    )

    # dados de entrada
    input_data = historical_values_btc.iloc[0, :].fillna(0).values.reshape(1, -1)
    scaler = load("./artifacts/scaler.bin")
    input_data = scaler.transform(input_data)

    Model = Features.Model

    if Model == "Machine Learning":
        model = load("./artifacts/model.joblib")

    predict = model.predict(input_data)
    last_price = historical_values_btc.iloc[0, 3]

    return {
        "Model": Model,
        "Ultimo preço": last_price,
        "Previsão para o próximo dia": predict[0],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
