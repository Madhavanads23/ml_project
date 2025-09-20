from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
import logging
import pandas as pd
from typing import Optional

from ..models.schemas import TrainingRequest, TrainingResponse, ErrorResponse
from ..ml_models.crop_predictor import CropPredictor

logger = logging.getLogger(__name__)

router = APIRouter()

def get_crop_model():
    """Dependency to get the crop model instance"""
    from main import crop_model
    if crop_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model service not available"
        )
    return crop_model

@router.post(
    "/train",
    response_model=TrainingResponse,
    summary="Train the crop prediction model",
    description="Train or retrain the Random Forest model with available data"
)
async def train_model(
    training_request: TrainingRequest,
    background_tasks: BackgroundTasks,
    model: CropPredictor = Depends(get_crop_model)
):
    """
    Train the crop prediction model.
    
    This endpoint will:
    - Train a new Random Forest model
    - Evaluate the model performance
    - Save the trained model
    - Return training metrics
    
    **Note**: In production, this should connect to your friend's database 
    to get the actual training data.
    """
    try:
        logger.info("Starting model training...")
        
        # TODO: Replace this with actual database connection
        # For now, we'll use sample data
        training_data = None  # This should come from your friend's database
        
        # Train the model
        metrics = model.train_model(training_data)
        
        # Save the model in background
        background_tasks.add_task(model.save_model)
        
        response = TrainingResponse(
            success=True,
            message="Model training completed successfully",
            metrics=metrics,
            model_info=model.model_info
        )
        
        logger.info("Model training completed successfully")
        return response
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error during model training: {str(e)}"
        )

@router.post(
    "/retrain",
    response_model=TrainingResponse,
    summary="Retrain model with new data",
    description="Retrain the model when new data is available from database"
)
async def retrain_model(
    background_tasks: BackgroundTasks,
    model: CropPredictor = Depends(get_crop_model)
):
    """
    Retrain the model with fresh data from the database.
    
    This endpoint should be called when:
    - New training data has been added to the database
    - Model performance has degraded
    - Periodic retraining is scheduled
    """
    try:
        logger.info("Starting model retraining...")
        
        # TODO: Implement database connection to fetch latest data
        # This is where you would connect to your friend's database
        # Example:
        # training_data = fetch_training_data_from_database()
        
        # For now, retrain with sample data
        metrics = model.train_model()
        
        # Save the updated model
        background_tasks.add_task(model.save_model)
        
        response = TrainingResponse(
            success=True,
            message="Model retraining completed successfully",
            metrics=metrics,
            model_info=model.model_info
        )
        
        logger.info("Model retraining completed successfully")
        return response
        
    except Exception as e:
        logger.error(f"Retraining error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error during model retraining: {str(e)}"
        )

@router.get(
    "/training/status",
    summary="Get training status",
    description="Check if the model is trained and get training information"
)
async def get_training_status(model: CropPredictor = Depends(get_crop_model)):
    """Get the current training status of the model."""
    try:
        status = {
            "is_trained": model.is_trained(),
            "model_info": model.model_info
        }
        
        if model.is_trained():
            status["feature_importance"] = model.get_feature_importance()
            status["supported_crops"] = model.crops
            status["model_size"] = len(model.crops)
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving training status: {str(e)}"
        )

@router.delete(
    "/model",
    summary="Clear trained model",
    description="Clear the current model from memory"
)
async def clear_model(model: CropPredictor = Depends(get_crop_model)):
    """Clear the current model from memory. Use with caution!"""
    try:
        model.model = None
        model.crops = []
        model.model_info['last_trained'] = None
        
        logger.info("Model cleared from memory")
        return {"success": True, "message": "Model cleared successfully"}
        
    except Exception as e:
        logger.error(f"Error clearing model: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing model: {str(e)}"
        )