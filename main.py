from fastapi import FastAPI
from pydantic import BaseModel
from detoxify import Detoxify
 
app = FastAPI()
model = Detoxify("original")
 
 
class TextRequest(BaseModel):
    text: str
 
 
@app.get("/")
def root():
    return {"message": "Toxicity API. POST /predict з полем text"}
 
 
@app.post("/predict")
def predict(req: TextRequest):
    results = model.predict(req.text)
    return {
        "text": req.text,
        "scores": {k: round(float(v), 4) for k, v in results.items()}
    }