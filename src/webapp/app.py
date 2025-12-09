from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import os
from loguru import logger
from pymongo import MongoClient
from datetime import datetime

MONGO_USER = os.getenv("MONGO_INITDB_ROOT_USERNAME", "admin")
MONGO_PASSWORD = os.getenv("MONGO_INITDB_ROOT_PASSWORD", "secretpassword")
MONGO_HOST = "mongodb"
MONGO_PORT = 27017

MONGO_URI = f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}"

client = MongoClient(MONGO_URI)
db = client["sentiment_db"]     
collection = db["logs"]        

app = FastAPI(title="Sentiment analyzer WebApp", version="1", description="This is a Webapp for sentiment analysis" )

class PredictInput(BaseModel):
    reviews: list[str]

model_path = os.environ.get('SENTIMENT_ANALYZER_MODEL_PATH', None)
if model_path is None:
    logger.error("Model cannot be found")
    raise Exception("Model cannot be found")

logger.info("Loading model")
model = pickle.load(open(f'{model_path}/model.pkl','rb'))

@app.post("/predict", summary="Predict polarity and save to history")
def predict(input: PredictInput):
    logger.info("Beginning prediction")
    
    predictions = model.predict(input.reviews)
    sentiments = ["positif" if x==1 else "negatif" for x in predictions]
    
    logger.info("Prediction finished")
    
    log_entry = {
        "reviews": input.reviews,       
        "sentiments": sentiments,       
        "timestamp": datetime.utcnow()  
    }
    
    try:
        collection.insert_one(log_entry)
        logger.info("Saved to MongoDB")
    except Exception as e:
        logger.error(f"Failed to save to MongoDB: {e}")

    return {"sentiments": sentiments}

@app.get("/history", summary="Get last n requests")
def get_history(n: int = 5):
    cursor = collection.find().sort("timestamp", -1).limit(n)
    
    history = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        history.append(doc)
        
    return history