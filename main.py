from fastapi import FastAPI
import pandas as pd
from typing import Optional
from pydantic import BaseModel, Field
import joblib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

columns = ["latitude", "longitude", "price", "minimum_nights",
    "number_of_reviews", "reviews_per_month",
    "calculated_host_listings_count", "availability_365",
    "neighbourhood_group", "neighbourhood"]

model = joblib.load("Model_Pipeline.pkl")

# Pydantic model for validation
class NYCAirbnbListing(BaseModel):
    # Location features
    latitude: float = Field(
        ..., 
        description="Latitude coordinate of the listing", 
        ge=-90.0, 
        le=90.0
    )
    longitude: float = Field(
        ..., 
        description="Longitude coordinate of the listing", 
        ge=-180.0, 
        le=180.0
    )
    
    # Financial and Availability features
    price: float = Field(
        ..., 
        description="Price per night for the listing", 
        ge=0.0
    )
    minimum_nights: int = Field(
        ..., 
        description="Minimum number of nights required to book", 
        ge=1
    )
    availability_365: int = Field(
        ..., 
        description="Number of days the listing is available for booking out of 365 days", 
        ge=0, 
        le=365
    )
    
    # Host and Review metrics
    number_of_reviews: int = Field(
        ..., 
        description="Total number of reviews the listing has received", 
        ge=0
    )
    reviews_per_month: Optional[float] = Field(
        None, 
        description="Average number of reviews per month (can be None if 0 total reviews)", 
        ge=0.0
    )
    calculated_host_listings_count: int = Field(
        ..., 
        description="Amount of listings the host has in total", 
        ge=1
    )
    
    # Categorical / Location text features
    neighbourhood_group: str = Field(
        ..., 
        description="The borough where the listing is located (e.g., Manhattan, Brooklyn)"
    )
    neighbourhood: str = Field(
        ..., 
        description="The specific neighborhood name"
    )

    class Config:
        # Example schema for interactive FastAPI API documentation
        json_schema_extra = {
            "example": {
                "latitude": 40.7128,
                "longitude": -74.0060,
                "price": 150.0,
                "minimum_nights": 2,
                "number_of_reviews": 45,
                "reviews_per_month": 1.5,
                "calculated_host_listings_count": 1,
                "availability_365": 240,
                "neighbourhood_group": "Manhattan",
                "neighbourhood": "Hell's Kitchen"
            }
        }


@app.get("/")
def greet():
    return {"message": "Hello World"}

@app.post('/predict')
def predict(features: NYCAirbnbListing):
    row = pd.DataFrame([features.dict()], columns=columns)
    prediction  = model.predict(row)
    probability = model.predict_proba(row)

    return {
        "Predicted_room_type": prediction[0],
        "Probability": probability.tolist()[0]}