# ML Crop Prediction Project

A comprehensive machine learning project for predicting optimal crops based on weather and soil conditions.

## Project Overview

This project consists of:
- **Backend API**: FastAPI-based service with Random Forest ML model for crop prediction
- **Database**: Data storage and management (handled by team member)
- **Frontend**: Web interface for user interaction (to be developed)

## Project Structure

```
ml_project/
├── backend/              # FastAPI backend with ML model
│   ├── app/             # Application code
│   ├── saved_models/    # Trained ML models
│   ├── main.py         # FastAPI server
│   ├── requirements.txt # Dependencies
│   └── README.md       # Backend documentation
├── database/            # Database components (handled by team member)
└── frontend/            # Web interface (to be developed)
```

## Backend Features

- 🌾 **Random Forest Model**: Predicts optimal crops based on weather and soil parameters
- 🔄 **REST API**: RESTful endpoints for predictions and model management
- 📊 **Batch Processing**: Support for multiple predictions
- 📋 **Auto Documentation**: Swagger/OpenAPI documentation
- 🐳 **Docker Support**: Containerized deployment

## Quick Start

### Backend Setup

1. Navigate to backend directory:
   ```bash
   cd backend
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the server:
   ```bash
   python main.py
   ```

4. Access API documentation:
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Docker Deployment

```bash
cd backend
docker-compose up -d
```

## API Usage

### Predict Crops

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

## Input Parameters

### Weather Conditions
- Temperature (°C)
- Humidity (%)
- pH level
- Rainfall (mm)
- Season

### Soil Conditions  
- Nitrogen content (N)
- Phosphorus content (P)
- Potassium content (K)
- Soil type

## Supported Crops

- Rice
- Wheat
- Maize
- Cotton
- Soybean
- Sugarcane
- Potato
- Tomato

## Team Collaboration

- **Backend & ML Model**: Current implementation
- **Database**: Data storage and management (team member responsibility)
- **Frontend**: Web interface (future development)

## Next Steps

1. Connect backend to production database
2. Develop web frontend interface
3. Integrate real-time weather data
4. Add more crop varieties
5. Implement user authentication
6. Deploy to production environment

## Documentation

- [Backend API Documentation](backend/README.md)
- API Docs: http://localhost:8000/docs (when running)

## License

This project is developed for crop prediction and agricultural optimization.