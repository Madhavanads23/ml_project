from fastapi import APIRouter, HTTPException, Depends
from typing import List
import logging

from ..models.schemas import (
    PredictionInput, 
    PredictionResponse, 
    CropPrediction, 
    ErrorResponse
)
from ..ml_models.crop_predictor import CropPredictor

logger = logging.getLogger(__name__)

router = APIRouter()

# Global model instance (will be shared with main.py)
def get_crop_model():
    """Dependency to get the crop model instance"""
    from main import crop_model
    if crop_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please train the model first."
        )
    return crop_model

@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict optimal crops",
    description="Get crop predictions based on weather and soil conditions"
)
async def predict_crops(
    input_data: PredictionInput,
    model: CropPredictor = Depends(get_crop_model)
):
    """
    Predict the best crops based on weather and soil conditions.
    
    - **weather**: Weather conditions including temperature, humidity, pH, rainfall, and season
    - **soil**: Soil conditions including NPK values and soil type
    
    Returns a list of crop predictions sorted by suitability score.
    """
    try:
        logger.info("Received prediction request")
        
        # Check if model is trained
        if not model.is_trained():
            # Try to train with sample data if no model exists
            logger.info("Model not trained, training with sample data...")
            model.train_model()
            model.save_model()
        
        # Make predictions
        predictions = model.predict(input_data)
        
        if not predictions:
            raise HTTPException(
                status_code=422,
                detail="Could not generate predictions for the given input"
            )
        
        # Get the best prediction
        best_crop = predictions[0]
        
        # Create input summary
        input_summary = {
            "temperature": input_data.weather.temperature,
            "humidity": input_data.weather.humidity,
            "ph": input_data.weather.ph,
            "rainfall": input_data.weather.rainfall,
            "season": input_data.weather.season.value,
            "nitrogen": input_data.soil.nitrogen,
            "phosphorus": input_data.soil.phosphorus,
            "potassium": input_data.soil.potassium,
            "soil_type": input_data.soil.soil_type.value
        }
        
        # Prepare response
        response = PredictionResponse(
            success=True,
            predictions=predictions[:5],  # Return top 5 predictions
            best_crop=best_crop,
            input_summary=input_summary,
            model_info=model.model_info
        )
        
        logger.info(f"Prediction completed. Best crop: {best_crop.crop_name}")
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during prediction: {str(e)}"
        )

@router.post(
    "/predict/batch",
    response_model=List[PredictionResponse],
    summary="Batch crop prediction",
    description="Get crop predictions for multiple input sets"
)
async def predict_crops_batch(
    input_data_list: List[PredictionInput],
    model: CropPredictor = Depends(get_crop_model)
):
    """
    Predict crops for multiple input sets in a single request.
    
    Useful for processing multiple locations or scenarios at once.
    """
    try:
        logger.info(f"Received batch prediction request for {len(input_data_list)} inputs")
        
        if len(input_data_list) > 100:  # Limit batch size
            raise HTTPException(
                status_code=422,
                detail="Batch size too large. Maximum 100 predictions per request."
            )
        
        if not model.is_trained():
            logger.info("Model not trained, training with sample data...")
            model.train_model()
            model.save_model()
        
        responses = []
        
        for i, input_data in enumerate(input_data_list):
            try:
                # Make predictions for this input
                predictions = model.predict(input_data)
                best_crop = predictions[0] if predictions else None
                
                input_summary = {
                    "temperature": input_data.weather.temperature,
                    "humidity": input_data.weather.humidity,
                    "ph": input_data.weather.ph,
                    "rainfall": input_data.weather.rainfall,
                    "season": input_data.weather.season.value,
                    "nitrogen": input_data.soil.nitrogen,
                    "phosphorus": input_data.soil.phosphorus,
                    "potassium": input_data.soil.potassium,
                    "soil_type": input_data.soil.soil_type.value
                }
                
                response = PredictionResponse(
                    success=True,
                    predictions=predictions[:5],
                    best_crop=best_crop,
                    input_summary=input_summary,
                    model_info=model.model_info
                )
                responses.append(response)
                
            except Exception as e:
                logger.error(f"Error processing batch item {i}: {e}")
                # Add error response for this item
                error_response = PredictionResponse(
                    success=False,
                    predictions=[],
                    best_crop=CropPrediction(crop_name="Error", confidence=0.0, suitability_score=0.0),
                    input_summary={},
                    model_info={"error": str(e)}
                )
                responses.append(error_response)
        
        logger.info(f"Batch prediction completed for {len(responses)} inputs")
        return responses
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during batch prediction: {str(e)}"
        )

@router.get(
    "/model/info",
    summary="Get model information",
    description="Get information about the current model"
)
async def get_model_info(model: CropPredictor = Depends(get_crop_model)):
    """Get detailed information about the current model."""
    try:
        info = model.model_info.copy()
        info['is_trained'] = model.is_trained()
        
        if model.is_trained():
            info['feature_importance'] = model.get_feature_importance()
            info['supported_crops'] = model.crops
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving model information: {str(e)}"
        )

@router.get(
    "/crops",
    response_model=List[str],
    summary="Get supported crops",
    description="Get list of crops that the model can predict"
)
async def get_supported_crops(model: CropPredictor = Depends(get_crop_model)):
    """Get the list of crops that the model can predict."""
    try:
        if not model.is_trained():
            raise HTTPException(
                status_code=503,
                detail="Model not trained. Cannot retrieve crop list."
            )
        
        return model.crops
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting crops: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving supported crops: {str(e)}"
        )