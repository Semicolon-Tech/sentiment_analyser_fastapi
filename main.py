import joblib
import sklearn
import numpy as np

# utilities
from utils import clean_text

from pydantic.main import BaseModel

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Response

app = FastAPI()
model = joblib.load("model/multinomial_naive_bayes_with_tfidf_vectorizer.joblib")


# A Pydantic model
class PredictRequest(BaseModel):
    text: str


# A Pydantic model
class PredictResponse(BaseModel):
    output: str


@app.get("/ping")
def ping():
    return Response(content="pong", media_type="text/plain")


@app.post('/predict', response_model=PredictResponse)
def predict(parameters: PredictRequest):

    # the model prediction
    predicted_sentiment = model.predict([parameters.text])  # prediction

    # the final response to send back
    response = {"output": "positive" if predicted_sentiment else "negative"}
    return response
