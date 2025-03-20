# Amazon Recommendation Engine - Usage Guide

This guide explains how to run and use the Amazon Recommendation Engine system.

## System Components

The Amazon Recommendation Engine consists of the following components:

1. **Data Generation**: Scripts to generate sample user, product, and interaction data
2. **Model Training**: Scripts to train recommendation models (NCF and LightGCN)
3. **Recommendation API**: A Flask-based API that serves recommendations
4. **Frontend Interface**: A web interface to interact with the recommendation system

## Getting Started

### Prerequisites

- Python 3.8+
- Required Python packages (install with `pip install -r requirements.txt`):
  - pandas
  - numpy
  - flask
  - pyarrow (for Parquet support)

### Step 1: Generate Sample Data

To generate sample data for the recommendation system:

```bash
python scripts/create_sample_data.py
```

This will create:
- 100 sample users
- 500 sample products
- ~5,000 reviews (interactions)
- ~10,000 events

Data is saved in the `data/processed/` directory in Parquet format.

### Step 2: Train Models

For a quick demonstration, you can run the simplified training simulation:

```bash
python scripts/simulate_training.py
```

This will create dummy model files in the `models/` directory that can be used with the API.

For actual model training:
- NCF: `python scripts/train_ncf_model.py`
- LightGCN: `python scripts/train_lightgcn_model.py`

### Step 3: Start the Services

To start all services (API and frontend) in one go:

```bash
python scripts/start_services.py
```

This will:
1. Start the recommendation API on port 5050
2. Start the frontend web server on port 8000
3. Open the frontend interface in your browser

Alternatively, you can start each component separately:

- API: `python src/api/app.py`
- Frontend: `python src/frontend/serve.py`

## Using the Web Interface

The web interface allows you to:

1. Select a recommendation model (NCF or LightGCN)
2. Select a user to get personalized product recommendations
3. Enter a product ID to find similar products

### Getting Recommendations

1. From the dropdown menu, select a model (NCF or LightGCN)
2. Select a user from the user dropdown
3. Click "Get Recommendations"
4. View the personalized product recommendations in the results panel

### Finding Similar Products

1. From the dropdown menu, select a model (NCF or LightGCN)
2. Enter a product ID in the input field (e.g., "product_123")
3. Click "Find Similar Products"
4. View the similar products in the results panel

## API Endpoints

The recommendation API provides the following endpoints:

- **GET /api/health**: Check API health status
- **GET /api/models**: List available recommendation models and their details
- **GET /api/users**: List available users in the system
- **GET /api/recommend/{model_name}/{user_id}**: Get recommendations for a user
- **GET /api/similar-products/{model_name}/{product_id}**: Get products similar to a given product

### Example API Requests

#### Get Recommendations

```
GET http://localhost:5050/api/recommend/ncf/user_1?n=10
```

Response:
```json
{
  "user_id": "user_1",
  "model": "ncf",
  "recommendations": ["product_145", "product_67", "product_422", ...]
}
```

#### Get Similar Products

```
GET http://localhost:5050/api/similar-products/lightgcn/product_123?n=5
```

Response:
```json
{
  "product_id": "product_123",
  "model": "lightgcn",
  "similar_products": ["product_356", "product_112", ...]
}
```

## Stopping the Services

To stop all services, press `Ctrl+C` in the terminal where you started `scripts/start_services.py`.

## Troubleshooting

### API Not Starting

- Ensure port 5050 is not in use by another application
- Check if all required dependencies are installed
- Verify that model files exist in the `models/` directory

### Frontend Not Loading

- Ensure port 8000 is not in use by another application
- Check if the API is running and accessible

### CORS Issues

If you're experiencing CORS issues when the frontend tries to access the API:
- Ensure the API is configured to allow requests from the frontend origin
- Check browser console for specific CORS error messages 