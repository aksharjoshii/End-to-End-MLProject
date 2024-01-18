import pandas as pd 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.pipeline.prediction import PredictionPipeline
import  uvicorn
model_name = "NYC Taxi tip prediction (classification) "
version = "v1.0.0"

app = FastAPI() 

class InputData(BaseModel):
    VendorID: int = Field(description="ID of the vendor 1 or 2", ge=1, le=2)
    tpep_pickup_datetime: str = Field(description="Pickup datetime")
    tpep_dropoff_datetime: str = Field(description="Dropoff datetime")
    passenger_count: int = Field(description="Number of passengers", ge=0, le=6)
    RatecodeID: int = Field(description="Rate code ID : (1-5)", ge=1, le=5)
    PULocationID: int = Field(description="Pickup Location ID :(1-265)", ge=1, le=265)
    DOLocationID: int = Field(description="Dropoff Location ID :(1-265)", ge=1, le=265)
    mean_duration: float = Field(description="Mean duration of trips", ge=0)
    mean_distance: float = Field(description="Mean distance of trips", ge=0)
    predicted_fare: float = Field(description="Predicted fare", ge=0)
    
    class Config:
        json_schema_extra = {
            'example':{
                "VendorID": 1,
                "tpep_pickup_datetime": "08/17/2017 4:06:26 AM",
                "tpep_dropoff_datetime": "08/17/2017 4:06:29 AM",
                "passenger_count": 2,
                "RatecodeID": 1,
                "PULocationID": 101,
                "DOLocationID": 102,
                "mean_duration": 30.5,
                "mean_distance": 5.0,
                "predicted_fare": 20.0
            }
        }

class PredictionResult(BaseModel):
    prediction: str
    predict_prob: float

@app.get('/info')
async def model_info():
    """Return model information, version, how to call"""
    return {
        "name": model_name,
        "version": version
    }

@app.post('/predict', response_model=PredictionResult)
def predict_single_sample(input_data: InputData):
    try:
        # Convert input data to DataFrame
        features_dict = input_data.model_dump()
        input_features = pd.DataFrame([features_dict])

        # Make prediction using the machine learning model
        predictor = PredictionPipeline()
        prediction, predict_prob = predictor.model_prediction(input_features)
        
        if prediction[0] ==1:
            result = 'generous'
        else : 
            result = 'not-generous'   

        return {"prediction": result, "predict_prob": round(predict_prob[0], 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", post=8000, reload=True, workers=1)
    