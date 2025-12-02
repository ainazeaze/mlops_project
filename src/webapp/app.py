from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import os

app=FastAPI(title="Sentiment analyzer WebApp", version="1", description="This is a Webapp for sentiment analysis" )

class PredictInput(BaseModel):
    reviews: list[str]

model_path = os.environ.get('SENTIMENT_ANALYZER_MODEL_PATH', None)
if model_path is None:
    raise Exception("Model cannot be found")

@app.post("/predict", summary="Take a dictionnary of key 'reviews' and a list of " \
"sentences the predict the polarity of and return a dictionnary of key 'sentiment' and a list of polarity")
def predict(input:PredictInput):
    model=pickle.load(open(f'{model_path}/model.pkl','rb'))
    predictions = model.predict(input.reviews)
    return {"sentiments" : ["positif" if x==1 else "negatif" for x in predictions]}
