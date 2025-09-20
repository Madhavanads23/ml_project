from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from contextlib import asynccontextmanager

# Import our API routes
from app.api import prediction, training
from app.core.config import settings
from app.ml_models.crop_predictor import CropPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
crop_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global crop_model
    
    # Startup
    logger.info("Starting up Crop Prediction API")
    try:
        crop_model = CropPredictor()
        # Try to load existing model
        crop_model.load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load existing model: {e}")
        logger.info("Model will be trained when first prediction is requested")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Crop Prediction API")

# Create FastAPI app
app = FastAPI(
    title="Crop Prediction API",
    description="API for predicting optimal crops based on weather and soil conditions",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(prediction.router, prefix="/api/v1", tags=["Prediction"])
app.include_router(training.router, prefix="/api/v1", tags=["Training"])

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Crop Prediction API is running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    global crop_model
    model_status = "loaded" if crop_model and crop_model.is_trained() else "not_loaded"
    
    return {
        "status": "healthy",
        "model_status": model_status,
        "api_version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )