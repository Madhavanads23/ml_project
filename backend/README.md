# Crop Prediction Backend API

A FastAPI-based backend service for predicting optimal crops based on weather and soil conditions using Random Forest machine learning model.

## Features

- 🌾 **Crop Prediction**: Predict optimal crops based on weather and soil parameters
- 🤖 **Random Forest Model**: Uses scikit-learn Random Forest for accurate predictions
- 📊 **Batch Processing**: Support for multiple predictions in a single request
- 🔄 **Model Training**: Endpoints for training and retraining the model
- 📋 **API Documentation**: Auto-generated OpenAPI/Swagger documentation
- 🌐 **CORS Support**: Ready for web frontend integration

## Project Structure

```
backend/
├── app/
│   ├── api/              # API routes
│   │   ├── prediction.py # Prediction endpoints
│   │   └── training.py   # Training endpoints
│   ├── core/            # Core configuration
│   │   └── config.py    # Application settings
│   ├── models/          # Data models
│   │   └── schemas.py   # Pydantic models
│   └── ml_models/       # ML model implementation
│       └── crop_predictor.py # Random Forest model
├── saved_models/        # Saved model files
├── main.py             # FastAPI application
├── requirements.txt    # Python dependencies
└── .env.example       # Environment variables template
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and configure your settings:

```bash
cp .env.example .env
```

### 3. Run the Server

```bash
python main.py
```

The API will be available at: `http://localhost:8000`

### 4. View API Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### Prediction Endpoints

- `POST /api/v1/predict` - Predict optimal crops
- `POST /api/v1/predict/batch` - Batch crop prediction
- `GET /api/v1/crops` - Get supported crops
- `GET /api/v1/model/info` - Get model information

### Training Endpoints

- `POST /api/v1/train` - Train the model
- `POST /api/v1/retrain` - Retrain with new data
- `GET /api/v1/training/status` - Get training status
- `DELETE /api/v1/model` - Clear model from memory

### Health Check

- `GET /` - Basic health check
- `GET /health` - Detailed health status

## Usage Examples

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

### Train Model

```bash
curl -X POST "http://localhost:8000/api/v1/train" \
  -H "Content-Type: application/json" \
  -d '{
    "retrain": true,
    "validation_split": 0.2
  }'
```

## Input Parameters

### Weather Data
- **temperature**: Temperature in Celsius (-50 to 60)
- **humidity**: Humidity percentage (0 to 100)
- **ph**: pH level (0 to 14)
- **rainfall**: Rainfall in mm (≥ 0)
- **season**: Season (spring, summer, monsoon, autumn, winter)

### Soil Data
- **nitrogen**: Nitrogen content (≥ 0)
- **phosphorus**: Phosphorus content (≥ 0)
- **potassium**: Potassium content (≥ 0)
- **soil_type**: Soil type (clay, sandy, loamy, silty, peaty, chalky)

## Supported Crops

The model currently supports prediction for these crops:
- Rice
- Wheat  
- Maize
- Cotton
- Soybean
- Sugarcane
- Potato
- Tomato

## Database Integration

To connect to your database (managed by your friend), update the database settings in `.env`:

```env
DATABASE_URL=postgresql://username:password@host:port/database
DB_HOST=your_db_host
DB_PORT=5432
DB_NAME=crop_prediction
DB_USER=your_username
DB_PASSWORD=your_password
```

Then modify the training endpoints in `app/api/training.py` to fetch data from your database.

## Model Configuration

The Random Forest model can be configured via environment variables:

- `RF_N_ESTIMATORS`: Number of trees (default: 100)
- `RF_MAX_DEPTH`: Maximum depth of trees (default: None)
- `RF_MIN_SAMPLES_SPLIT`: Minimum samples to split (default: 2)
- `RF_MIN_SAMPLES_LEAF`: Minimum samples per leaf (default: 1)
- `RF_RANDOM_STATE`: Random state for reproducibility (default: 42)

## Development

### Running in Development Mode

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black .
flake8 .
```

## Production Deployment

1. Set `DEBUG=False` in environment variables
2. Configure proper CORS origins
3. Use a production WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## Contributing

1. Connect the training endpoints to your actual database
2. Add more sophisticated feature engineering
3. Implement model versioning
4. Add more comprehensive testing
5. Add authentication and rate limiting

## License

This project is part of the ML crop prediction system.