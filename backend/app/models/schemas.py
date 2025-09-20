from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class SoilType(str, Enum):
    """Soil type enumeration"""
    CLAY = "clay"
    SANDY = "sandy"
    LOAMY = "loamy"
    SILTY = "silty"
    PEATY = "peaty"
    CHALKY = "chalky"

class Season(str, Enum):
    """Season enumeration"""
    SPRING = "spring"
    SUMMER = "summer"
    MONSOON = "monsoon"
    AUTUMN = "autumn"
    WINTER = "winter"

class WeatherData(BaseModel):
    """Weather conditions input model"""
    temperature: float = Field(..., description="Temperature in Celsius", ge=-50, le=60)
    humidity: float = Field(..., description="Humidity percentage", ge=0, le=100)
    ph: float = Field(..., description="pH level", ge=0, le=14)
    rainfall: float = Field(..., description="Rainfall in mm", ge=0)
    season: Season = Field(..., description="Current season")
    
    class Config:
        json_schema_extra = {
            "example": {
                "temperature": 25.5,
                "humidity": 65.0,
                "ph": 6.8,
                "rainfall": 180.5,
                "season": "monsoon"
            }
        }

class SoilData(BaseModel):
    """Soil conditions input model"""
    nitrogen: float = Field(..., description="Nitrogen content (N)", ge=0)
    phosphorus: float = Field(..., description="Phosphorus content (P)", ge=0)
    potassium: float = Field(..., description="Potassium content (K)", ge=0)
    soil_type: SoilType = Field(..., description="Type of soil")
    
    class Config:
        json_schema_extra = {
            "example": {
                "nitrogen": 90.0,
                "phosphorus": 42.0,
                "potassium": 43.0,
                "soil_type": "loamy"
            }
        }

class PredictionInput(BaseModel):
    """Complete input model for crop prediction"""
    weather: WeatherData
    soil: SoilData
    
    class Config:
        json_schema_extra = {
            "example": {
                "weather": {
                    "temperature": 25.5,
                    "humidity": 65.0,
                    "ph": 6.8,
                    "rainfall": 180.5,
                    "season": "monsoon"
                },
                "soil": {
                    "nitrogen": 90.0,
                    "phosphorus": 42.0,
                    "potassium": 43.0,
                    "soil_type": "loamy"
                }
            }
        }

class CropPrediction(BaseModel):
    """Single crop prediction result"""
    crop_name: str = Field(..., description="Name of the predicted crop")
    confidence: float = Field(..., description="Prediction confidence (0-1)", ge=0, le=1)
    suitability_score: float = Field(..., description="Suitability score (0-100)", ge=0, le=100)

class PredictionResponse(BaseModel):
    """Response model for crop prediction"""
    success: bool = Field(..., description="Whether prediction was successful")
    predictions: List[CropPrediction] = Field(..., description="List of crop predictions")
    best_crop: CropPrediction = Field(..., description="Best recommended crop")
    input_summary: Dict[str, Any] = Field(..., description="Summary of input conditions")
    model_info: Dict[str, Any] = Field(..., description="Information about the model used")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "predictions": [
                    {
                        "crop_name": "Rice",
                        "confidence": 0.85,
                        "suitability_score": 85.0
                    },
                    {
                        "crop_name": "Wheat",
                        "confidence": 0.72,
                        "suitability_score": 72.0
                    }
                ],
                "best_crop": {
                    "crop_name": "Rice",
                    "confidence": 0.85,
                    "suitability_score": 85.0
                },
                "input_summary": {
                    "temperature": 25.5,
                    "humidity": 65.0,
                    "season": "monsoon",
                    "soil_type": "loamy"
                },
                "model_info": {
                    "model_type": "Random Forest",
                    "version": "1.0.0",
                    "last_trained": "2025-09-20"
                }
            }
        }

class TrainingRequest(BaseModel):
    """Request model for model training"""
    retrain: bool = Field(True, description="Whether to retrain the model from scratch")
    validation_split: Optional[float] = Field(0.2, description="Validation split ratio", ge=0.1, le=0.5)
    
class TrainingResponse(BaseModel):
    """Response model for training"""
    success: bool = Field(..., description="Whether training was successful")
    message: str = Field(..., description="Training result message")
    metrics: Dict[str, float] = Field(..., description="Training metrics")
    model_info: Dict[str, Any] = Field(..., description="Updated model information")

class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = Field(False, description="Always false for errors")
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")