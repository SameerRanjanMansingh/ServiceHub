from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
import uvicorn
import pandas as pd
import mlflow
import json
import joblib
from mlflow import MlflowClient
from sklearn import set_config
from scripts.data_clean_utils import perform_data_cleaning

# set the output as pandas
set_config(transform_output='pandas')

# initialize dagshub
import dagshub
import mlflow.client

dagshub.init(repo_owner='SameerRanjanMansingh', repo_name='ServiceHub', mlflow=True)

# set the tracking server

mlflow.set_tracking_uri("https://dagshub.com/SameerRanjanMansingh/ServiceHub.mlflow")


class Data(BaseModel):  
    Request_ID: str
    Worker_ID: str
    Worker_Age: str
    Worker_Ratings: str
    Company_latitude: float
    Company_longitude: float
    Service_location_latitude: float
    Service_location_longitude: float
    Request_Date: str
    Time_Requested: str
    Worker_dispatch_time: str
    weather_conditions: str
    Road_traffic_density: str
    Work_experience: int
    Type_of_vehicle: str
    Multiple_requests: str
    Festival: str
    City: str

    
    
def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
        
    return run_info


def load_transformer(transformer_path):
    transformer = joblib.load(transformer_path)
    return transformer



# columns to preprocess in data
num_cols = ["age","ratings","response_time_minutes","distance"]

nominal_cat_cols = ['weather',
                    'type_of_vehicle',"festival",
                    "city_type",
                    "is_weekend",
                    "requested_time_of_day"]

ordinal_cat_cols = ["traffic","distance_type"]

#mlflow client
client = MlflowClient()

# load the model info to get the model name
model_name = load_model_information("run_information.json")['model_name']

# stage of the model
stage = "Production"

# get the latest model version
# latest_model_ver = client.get_latest_versions(name=model_name,stages=[stage])
# print(f"Latest model in production is version {latest_model_ver[0].version}")

# load model path
model_path = f"models:/{model_name}/{stage}"

# load the latest model from model registry
model = mlflow.sklearn.load_model(model_path)

# load the preprocessor
preprocessor_path = "models/preprocessor.joblib"
preprocessor = load_transformer(preprocessor_path)

# build the model pipeline
model_pipe = Pipeline(steps=[
    ('preprocess',preprocessor),
    ("regressor",model)
])

# create the app
app = FastAPI()

# create the home endpoint
@app.get(path="/")
def home():
    return "Welcome to the ServiceHub"

# create the predict endpoint
@app.post(path="/predict")
def do_predictions(data: Data):
    pred_data = pd.DataFrame({
        'Request_ID' : data.Request_ID,
        'Worker_ID' : data.Worker_ID,
        'Worker_Age' : data.Worker_Age,
        'Worker_Ratings' : data.Worker_Ratings,
        'Company_latitude' : data.Company_latitude,
        'Company_longitude' : data.Company_longitude,
        'Service_location_latitude' : data.Service_location_latitude,
        'Service_location_longitude' : data.Service_location_longitude,
        'Request_Date' : data.Request_Date,
        'Time_Requested' : data.Time_Requested,
        'Worker_dispatch_time' : data.Worker_dispatch_time,
        'weather_conditions' : data.weather_conditions,
        'Road_traffic_density' : data.Road_traffic_density,
        'Work_experience' : data.Work_experience,
        'Type_of_vehicle' : data.Type_of_vehicle,
        'Multiple_requests' : data.Multiple_requests,
        'Festival' : data.Festival,
        'City' : data.City
        },index=[0]
    )
    # clean the raw input data
    cleaned_data = perform_data_cleaning(pred_data)
    # get the predictions
    predictions = model_pipe.predict(cleaned_data)[0]

    return predictions
   
   
if __name__ == "__main__":
    uvicorn.run(app="app:app",host="0.0.0.0",port=8000)