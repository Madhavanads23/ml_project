from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    app_name: str = "Crop Prediction API"
    debug: bool = True
    
    # Model Settings
    model_path: str = "saved_models/crop_predictor.joblib"
    model_type: str = "random_forest"
    
    # Random Forest specific settings
    rf_n_estimators: int = 100
    rf_max_depth: Optional[int] = None
    rf_min_samples_split: int = 2
    rf_min_samples_leaf: int = 1
    rf_random_state: int = 42
    
    # Database settings (for connecting to your friend's database)
    database_url: Optional[str] = None
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "crop_prediction"
    db_user: str = "user"
    db_password: str = "password"
    
    # Data processing
    test_size: float = 0.2
    validation_split: float = 0.2
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Create global settings instance
settings = Settings()