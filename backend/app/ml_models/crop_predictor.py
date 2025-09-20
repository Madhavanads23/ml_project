import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Dict, List, Tuple, Optional, Any
import os
from datetime import datetime

from ..core.config import settings
from ..models.schemas import PredictionInput, CropPrediction

logger = logging.getLogger(__name__)

class CropPredictor:
    """Random Forest based crop prediction model"""
    
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_names = [
            'temperature', 'humidity', 'ph', 'rainfall', 'nitrogen', 
            'phosphorus', 'potassium', 'season_encoded', 'soil_type_encoded'
        ]
        self.crops = []
        self.model_info = {
            'model_type': 'Random Forest',
            'version': '1.0.0',
            'last_trained': None,
            'features': self.feature_names
        }
        
    def _encode_categorical_features(self, data: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Encode categorical features (season and soil_type)"""
        data_encoded = data.copy()
        
        # Season encoding
        season_mapping = {'spring': 0, 'summer': 1, 'monsoon': 2, 'autumn': 3, 'winter': 4}
        data_encoded['season_encoded'] = data['season'].map(season_mapping)
        
        # Soil type encoding
        soil_mapping = {'clay': 0, 'sandy': 1, 'loamy': 2, 'silty': 3, 'peaty': 4, 'chalky': 5}
        data_encoded['soil_type_encoded'] = data['soil_type'].map(soil_mapping)
        
        return data_encoded
    
    def _prepare_features(self, input_data: PredictionInput) -> np.ndarray:
        """Prepare features from input data for prediction"""
        # Convert input to DataFrame format
        data_dict = {
            'temperature': [input_data.weather.temperature],
            'humidity': [input_data.weather.humidity],
            'ph': [input_data.weather.ph],
            'rainfall': [input_data.weather.rainfall],
            'nitrogen': [input_data.soil.nitrogen],
            'phosphorus': [input_data.soil.phosphorus],
            'potassium': [input_data.soil.potassium],
            'season': [input_data.weather.season.value],
            'soil_type': [input_data.soil.soil_type.value]
        }
        
        df = pd.DataFrame(data_dict)
        df_encoded = self._encode_categorical_features(df, fit=False)
        
        # Select and order features
        features = df_encoded[self.feature_names].values
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        return features_scaled
    
    def train_model(self, training_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Train the Random Forest model"""
        try:
            # For now, create sample data if no training data provided
            if training_data is None:
                training_data = self._create_sample_data()
            
            logger.info("Starting model training...")
            
            # Prepare features and target
            X = training_data[['temperature', 'humidity', 'ph', 'rainfall', 'nitrogen', 
                             'phosphorus', 'potassium', 'season', 'soil_type']]
            y = training_data['crop']
            
            # Encode categorical features
            X_encoded = self._encode_categorical_features(X, fit=True)
            X_features = X_encoded[self.feature_names]
            
            # Encode target labels
            y_encoded = self.label_encoder.fit_transform(y)
            self.crops = self.label_encoder.classes_.tolist()
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X_features)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, 
                test_size=settings.test_size, 
                random_state=settings.rf_random_state,
                stratify=y_encoded
            )
            
            # Create and train Random Forest model
            self.model = RandomForestClassifier(
                n_estimators=settings.rf_n_estimators,
                max_depth=settings.rf_max_depth,
                min_samples_split=settings.rf_min_samples_split,
                min_samples_leaf=settings.rf_min_samples_leaf,
                random_state=settings.rf_random_state,
                n_jobs=-1
            )
            
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            train_accuracy = self.model.score(X_train, y_train)
            test_accuracy = self.model.score(X_test, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X_scaled, y_encoded, cv=5)
            
            # Update model info
            self.model_info['last_trained'] = datetime.now().isoformat()
            self.model_info['train_accuracy'] = train_accuracy
            self.model_info['test_accuracy'] = test_accuracy
            self.model_info['cv_mean'] = cv_scores.mean()
            self.model_info['cv_std'] = cv_scores.std()
            self.model_info['n_samples'] = len(training_data)
            self.model_info['n_features'] = len(self.feature_names)
            
            logger.info(f"Model training completed. Test accuracy: {test_accuracy:.4f}")
            
            return {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'n_samples': len(training_data)
            }
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise
    
    def predict(self, input_data: PredictionInput) -> List[CropPrediction]:
        """Make crop predictions"""
        if not self.is_trained():
            raise ValueError("Model is not trained. Please train the model first.")
        
        try:
            # Prepare features
            features = self._prepare_features(input_data)
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(features)[0]
            
            # Create predictions for all crops
            predictions = []
            for i, crop in enumerate(self.crops):
                prediction = CropPrediction(
                    crop_name=crop,
                    confidence=float(probabilities[i]),
                    suitability_score=float(probabilities[i] * 100)
                )
                predictions.append(prediction)
            
            # Sort by confidence
            predictions.sort(key=lambda x: x.confidence, reverse=True)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
    
    def save_model(self, filepath: Optional[str] = None) -> None:
        """Save the trained model to disk"""
        if not self.is_trained():
            raise ValueError("No trained model to save")
        
        if filepath is None:
            filepath = settings.model_path
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'crops': self.crops,
            'feature_names': self.feature_names,
            'model_info': self.model_info
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Optional[str] = None) -> None:
        """Load a trained model from disk"""
        if filepath is None:
            filepath = settings.model_path
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.scaler = model_data['scaler']
        self.crops = model_data['crops']
        self.feature_names = model_data['feature_names']
        self.model_info = model_data['model_info']
        
        logger.info(f"Model loaded from {filepath}")
    
    def is_trained(self) -> bool:
        """Check if the model is trained"""
        return self.model is not None
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model"""
        if not self.is_trained():
            raise ValueError("Model is not trained")
        
        importance = self.model.feature_importances_
        feature_importance = dict(zip(self.feature_names, importance))
        
        return feature_importance
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample training data for demonstration"""
        # This is sample data - replace with actual data from your friend's database
        np.random.seed(42)
        
        crops = ['Rice', 'Wheat', 'Maize', 'Cotton', 'Soybean', 'Sugarcane', 'Potato', 'Tomato']
        seasons = ['spring', 'summer', 'monsoon', 'autumn', 'winter']
        soil_types = ['clay', 'sandy', 'loamy', 'silty', 'peaty', 'chalky']
        
        n_samples = 1000
        data = []
        
        for _ in range(n_samples):
            crop = np.random.choice(crops)
            
            # Generate realistic data based on crop type
            if crop == 'Rice':
                temp = np.random.normal(25, 3)
                humidity = np.random.normal(80, 10)
                rainfall = np.random.normal(200, 50)
                season = np.random.choice(['monsoon', 'summer'])
            elif crop == 'Wheat':
                temp = np.random.normal(20, 4)
                humidity = np.random.normal(60, 15)
                rainfall = np.random.normal(100, 30)
                season = np.random.choice(['winter', 'spring'])
            else:
                temp = np.random.normal(22, 5)
                humidity = np.random.normal(70, 15)
                rainfall = np.random.normal(150, 40)
                season = np.random.choice(seasons)
            
            data.append({
                'temperature': max(5, min(45, temp)),
                'humidity': max(20, min(95, humidity)),
                'ph': np.random.normal(6.5, 0.8),
                'rainfall': max(0, rainfall),
                'nitrogen': np.random.normal(80, 20),
                'phosphorus': np.random.normal(50, 15),
                'potassium': np.random.normal(60, 18),
                'season': season,
                'soil_type': np.random.choice(soil_types),
                'crop': crop
            })
        
        return pd.DataFrame(data)