from fastapi import FastAPI
import joblib

app = FastAPI()

# load model
model = joblib.load("models/ftis_model.pkl")


@app.post("/predict")
def predict(data: dict):
    input_data = [[
        data["altitude"],
        data["speed"],
        data["temperature"]
    ]]

    result = model.predict(input_data)

    return {"prediction": result.tolist()}