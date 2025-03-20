"""
Flask API for the Amazon Product Recommendation System.

This module provides RESTful endpoints for serving recommendations based on trained models.
"""

import os
import pickle
import numpy as np
import threading
import time
from datetime import datetime
from flask import Flask, jsonify, request, g
from flask_cors import CORS
import pandas as pd
import logging
import uuid
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import the feedback loop system
try:
    import feedback_loop
    events_storage = feedback_loop.get_events_storage()
    recommendation_cache = events_storage['recommendation_cache']
    logger.info("Imported feedback_loop module")
    USE_FEEDBACK_LOOP = True
except ImportError as e:
    logger.warning(f"Could not import feedback_loop module: {e}. Feedback loop will be disabled")
    recommendation_cache = {}
    USE_FEEDBACK_LOOP = False

# Import Kafka components if available
try:
    import sys
    print("Python path in API:", sys.path)
    print("Working directory in API:", os.getcwd())
    from src.kafka.producer import get_producer
    from src.kafka.stream_processor import RealtimeRecommender
    KAFKA_AVAILABLE = True
    logger.info("Kafka integration enabled")
except ImportError as e:
    logger.warning(f"Kafka modules not found. Error: {e}. Running without real-time processing.")
    print(f"Import Error Details: {e}")
    import traceback
    traceback.print_exc()
    KAFKA_AVAILABLE = False

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables to store loaded models
models = {}
# For tracking when models were last updated
model_last_updated = {}

# Kafka producers
event_producer = None
metrics_producer = None
realtime_recommender = None

# For tracking request times
app.config['REQUEST_TIMES'] = []

# Cold-start strategy: Track popular items
popular_products = []
product_popularity_scores = defaultdict(float)

def load_models():
    """Load all available trained models."""
    global models, model_last_updated, popular_products, product_popularity_scores
    models_directory = os.path.join(os.getcwd(), 'data', 'models')
    
    if not os.path.exists(models_directory):
        logger.warning(f"Models directory not found: {models_directory}")
        return models
    
    try:
        logger.info(f"Loading models from {models_directory}")
        
        # Load NCF model
        ncf_path = os.path.join(models_directory, 'ncf_model.pkl')
        if os.path.exists(ncf_path):
            with open(ncf_path, 'rb') as f:
                models['ncf'] = pickle.load(f)
            model_last_updated['ncf'] = datetime.now()
            logger.info(f"Loaded NCF model with {len(models['ncf']['user_map'])} users and {len(models['ncf']['product_map'])} items")
        else:
            logger.warning("NCF model not found")
        
        # Load LightGCN model
        lightgcn_path = os.path.join(models_directory, 'lightgcn_model.pkl')
        if os.path.exists(lightgcn_path):
            with open(lightgcn_path, 'rb') as f:
                models['lightgcn'] = pickle.load(f)
            model_last_updated['lightgcn'] = datetime.now()
            logger.info(f"Loaded LightGCN model with {len(models['lightgcn']['user_map'])} users and {len(models['lightgcn']['product_map'])} items")
        else:
            logger.warning("LightGCN model not found")
            
        # Load Simple Matrix Factorization model
        simple_mf_path = os.path.join(models_directory, 'simple_mf_model.pkl')
        if os.path.exists(simple_mf_path):
            with open(simple_mf_path, 'rb') as f:
                models['simple_mf'] = pickle.load(f)
            model_last_updated['simple_mf'] = datetime.now()
            logger.info(f"Loaded Simple MF model with {len(models['simple_mf']['user_map'])} users and {len(models['simple_mf']['product_map'])} items")
        else:
            logger.warning("Simple MF model not found")
            
        # Load Content-Based model
        content_based_path = os.path.join(models_directory, 'content_based_model.pkl')
        if os.path.exists(content_based_path):
            with open(content_based_path, 'rb') as f:
                models['content_based'] = pickle.load(f)
            model_last_updated['content_based'] = datetime.now()
            logger.info(f"Loaded Content-Based model with {len(models['content_based']['user_map'])} users and {len(models['content_based']['product_map'])} items")
        else:
            logger.warning("Content-Based model not found")
            
        # Combine lightgcn and content-based for hybrid model if both are available
        if 'lightgcn' in models and 'content_based' in models:
            logger.info("Creating hybrid model from LightGCN and Content-Based models")
            models['hybrid'] = {
                'name': 'hybrid',
                'version': '1.0.0',
                'user_map': models['lightgcn']['user_map'],  # Use LightGCN user mappings
                'product_map': models['lightgcn']['product_map'],  # Use LightGCN product mappings
                'metrics': {
                    'lightgcn_weight': 0.7,  # 70% weight for LightGCN
                    'content_weight': 0.3,   # 30% weight for Content-Based
                },
                'components': ['lightgcn', 'content_based']
            }
            model_last_updated['hybrid'] = datetime.now()
            
        # Initialize popular products for cold-start
        best_model = None
        if 'lightgcn' in models:
            best_model = 'lightgcn'  # Prefer LightGCN for popularity
        elif 'content_based' in models:
            best_model = 'content_based'  # Then content-based
        elif 'simple_mf' in models:
            best_model = 'simple_mf'  # Then simple matrix factorization
        elif 'ncf' in models:
            best_model = 'ncf'  # Last, NCF
        
        if best_model:
            model = models[best_model]
            # Get avg item vector to use for cold-start
            item_factors = model['item_factors']
            
            # Calculate average popularity score for all products
            for idx, prod_id in enumerate(model['product_map'].keys()):
                try:
                    # Different models may have different layouts for item_factors
                    if best_model in ['content_based', 'simple_mf']:
                        # item_factors is (n_components, n_items)
                        factor_column = item_factors[:, idx]
                    else:
                        # item_factors is (n_items, n_components)
                        factor_column = item_factors[idx]
                    
                    popularity_score = np.sum(factor_column)
                    product_popularity_scores[prod_id] = popularity_score
                except IndexError:
                    logger.error(f"IndexError for product {prod_id}, index {idx}, shape of item_factors: {item_factors.shape}")
            
            # Sort by popularity score
            sorted_products = sorted(product_popularity_scores.items(), key=lambda x: x[1], reverse=True)
            popular_products = [p[0] for p in sorted_products[:100]]  # Top 100 products
            logger.info(f"Initialized popular products list with {len(popular_products)} items")
    
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()
    
    return models

def initialize_kafka():
    """Initialize Kafka producers and stream processors."""
    global event_producer, metrics_producer, realtime_recommender, KAFKA_AVAILABLE
    
    if not KAFKA_AVAILABLE:
        return
    
    try:
        # Initialize Kafka producers
        event_producer = get_producer('product_event')
        metrics_producer = get_producer('metrics', metrics_type='model')
        
        # Initialize real-time recommender
        realtime_recommender = RealtimeRecommender(
            update_interval_seconds=30,
            cache_size=1000
        )
        
        # Start the real-time recommender in a non-blocking mode
        realtime_recommender.start(blocking=False)
        
        logger.info("Kafka producers and stream processors initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Kafka: {e}")
        KAFKA_AVAILABLE = False

def track_event(user_id, product_id, event_type, metadata=None):
    """
    Track a user event in Kafka.
    
    Args:
        user_id: User ID
        product_id: Product ID
        event_type: Type of event (view, click, purchase)
        metadata: Additional data about the event
    """
    # Generate a unique session ID if not available
    session_id = request.headers.get('X-Session-ID', str(uuid.uuid4()))
    
    # First, store in feedback loop if available
    if USE_FEEDBACK_LOOP:
        event = {
            'user_id': user_id,
            'product_id': product_id,
            'event_type': event_type,
            'session_id': session_id,
            'metadata': metadata or {}
        }
        feedback_loop.store_event(event)
    
    # Then send to Kafka if available
    if not KAFKA_AVAILABLE or not event_producer:
        return
    
    try:
        event_producer.send_product_event(
            user_id=user_id,
            product_id=product_id,
            event_type=event_type,
            session_id=session_id,
            metadata=metadata or {}
        )
        logger.debug(f"Tracked event: {event_type} for user {user_id} on product {product_id}")
    except Exception as e:
        logger.error(f"Failed to track event: {e}")

def handle_new_user(user_id):
    """
    Create a placeholder for a new user that doesn't exist in the model.
    
    Args:
        user_id: The user ID to create
        
    Returns:
        True if user was created, False otherwise
    """
    logger.info(f"Handling new user: {user_id}")
    
    # Only create new users if we're using the feedback loop
    if not USE_FEEDBACK_LOOP:
        return False
    
    # Use the feedback_loop module to handle the new user
    return feedback_loop.handle_new_user(user_id, models)

def get_cold_start_recommendations(model_name, top_n=10):
    """
    Get recommendations for a new user with no history.
    
    Args:
        model_name: The name of the model
        top_n: Number of recommendations to return
        
    Returns:
        List of top-N recommended product IDs
    """
    logger.info(f"Getting cold-start recommendations for model {model_name}, top_n={top_n}")
    
    if USE_FEEDBACK_LOOP:
        return feedback_loop.get_cold_start_recommendations(popular_products, model_name, top_n)
    
    # If feedback loop is not available, use pre-computed popular products
    if popular_products:
        return popular_products[:top_n]
    
    # Otherwise, just take some random products from the model
    if model_name in models:
        model = models[model_name]
        product_ids = list(model['product_map'].keys())
        np.random.shuffle(product_ids)
        return product_ids[:top_n]
    
    # Last resort: generate fake product IDs
    return [f"product_{i}" for i in range(1, top_n+1)]

def get_recommendations(model_name, user_id, top_n=10):
    """
    Get top-N product recommendations for a user.
    
    Args:
        model_name: The name of the model to use ('ncf', 'lightgcn', 'simple_mf', 'content_based', 'hybrid')
        user_id: The user ID to get recommendations for
        top_n: Number of recommendations to return
        
    Returns:
        List of top-N recommended product IDs
    """
    # Check if this is a new user flagged in the cache
    new_user_cache_key = f"{model_name}_{user_id}_new_user"
    is_new_user = new_user_cache_key in recommendation_cache
    
    # Check if we have cached recommendations from the feedback loop
    cache_key = f"{model_name}_{user_id}_{top_n}"
    if USE_FEEDBACK_LOOP and cache_key in recommendation_cache:
        logger.debug(f"Serving feedback-based recommendations for {cache_key}")
        return recommendation_cache[cache_key]
    
    # If not using feedback loop or no cached recommendations, check model
    if model_name not in models:
        return {"error": f"Model {model_name} not loaded"}
    
    model = models[model_name]
    
    # Special handling for hybrid model
    if model_name == 'hybrid':
        return get_hybrid_recommendations(user_id, top_n)
    
    # Check if user exists in model
    user_map = model.get('user_map', {})
    if user_id not in user_map:
        # Handle new user by creating a new user entry if supported
        if not is_new_user and handle_new_user(user_id):
            # Flag this as a new user in the cache
            recommendation_cache[new_user_cache_key] = True
            # Return cold-start recommendations
            return get_cold_start_recommendations(model_name, top_n)
        else:
            # Return cold-start recommendations if can't handle new user
            return get_cold_start_recommendations(model_name, top_n)
    
    # Generate recommendations based on model type
    try:
        user_idx = user_map[user_id]
        product_map = model.get('product_map', {})
        user_factors = model.get('user_factors')
        item_factors = model.get('item_factors')
        
        # In most cases, we compute scores based on matrix multiplication of user factors and item factors
        user_vector = user_factors[user_idx]
        
        # Handle different item_factors layouts based on model type
        if model_name in ['content_based', 'simple_mf']:
            # item_factors is (n_components, n_items)
            scores = np.dot(user_vector, item_factors)
        else:
            # item_factors is (n_items, n_components)
            scores = np.dot(user_vector, item_factors.T)
        
        # Get top N items
        top_indices = np.argsort(scores)[::-1][:top_n]
        
        # Map to product IDs
        reverse_product_map = {idx: prod_id for prod_id, idx in product_map.items()}
        recommendations = [reverse_product_map[idx] for idx in top_indices]
        
        return recommendations
    
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        import traceback
        traceback.print_exc()
        # Fall back to cold-start recommendations
        return get_cold_start_recommendations(model_name, top_n)

def get_hybrid_recommendations(user_id, top_n=10):
    """
    Get recommendations using a combination of models (hybrid approach).
    
    Args:
        user_id: The user ID to get recommendations for
        top_n: Number of recommendations to return
        
    Returns:
        List of top-N recommended product IDs
    """
    hybrid_model = models.get('hybrid')
    if not hybrid_model:
        logger.error("Hybrid model not initialized")
        return get_cold_start_recommendations('hybrid', top_n)
    
    components = hybrid_model.get('components', [])
    if not components:
        logger.error("Hybrid model has no components")
        return get_cold_start_recommendations('hybrid', top_n)
    
    # Get weights for each component
    weights = {}
    for component in components:
        weight_key = f"{component}_weight"
        weights[component] = hybrid_model.get('metrics', {}).get(weight_key, 1.0 / len(components))
    
    # Initialize score map
    scores = {}
    
    # Get top N*2 recommendations from each component to ensure enough diversity
    for component in components:
        if component not in models:
            logger.warning(f"Component model {component} not loaded")
            continue
        
        try:
            # Get recommendations from this component
            component_recs = get_recommendations(component, user_id, top_n * 2)
            
            # Skip if we got an error or no recommendations
            if isinstance(component_recs, dict) and 'error' in component_recs:
                logger.warning(f"Error from component {component}: {component_recs['error']}")
                continue
            
            # Add weighted scores
            weight = weights[component]
            for i, product_id in enumerate(component_recs):
                # Score inversely proportional to rank position
                rank_score = (top_n * 2 - i) / (top_n * 2)
                if product_id not in scores:
                    scores[product_id] = 0
                scores[product_id] += rank_score * weight
                
        except Exception as e:
            logger.error(f"Error getting recommendations from component {component}: {e}")
    
    # Sort by final score
    sorted_products = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Get top N
    hybrid_recommendations = [p[0] for p in sorted_products[:top_n]]
    
    return hybrid_recommendations

@app.before_request
def before_request():
    """Record request start time."""
    g.start_time = time.time()
    
    # Initialize request_times list in g if not already present
    if not hasattr(g, 'request_times'):
        g.request_times = []

@app.after_request
def after_request(response):
    """Log request duration and other metrics."""
    if hasattr(g, 'start_time'):
        duration = time.time() - g.start_time
        logger.info(f"Request to {request.path} completed in {duration:.4f}s")
        
        # Track response time for metrics
        if hasattr(g, 'request_times'):
            g.request_times.append(duration)
            # Keep only the last 1000 request times
            if len(g.request_times) > 1000:
                g.request_times = g.request_times[-1000:]
        
        # Log metrics if Kafka is available
        if KAFKA_AVAILABLE and metrics_producer:
            try:
                metrics_producer.send_metric(
                    metric_name="api_request_duration",
                    metric_value=duration,
                    tags={
                        "path": request.path,
                        "method": request.method,
                        "status_code": response.status_code
                    }
                )
            except Exception as e:
                logger.error(f"Failed to log metrics: {e}")
    
    return response

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    status = {
        "status": "healthy",
        "models_loaded": list(models.keys()),
        "kafka_available": KAFKA_AVAILABLE,
        "feedback_loop_enabled": USE_FEEDBACK_LOOP,
        "model_last_updated": {k: v.isoformat() for k, v in model_last_updated.items()} if model_last_updated else {},
        "cache_size": len(recommendation_cache),
        "total_events_tracked": sum(len(events) for events in events_storage['user_events'].values()) if USE_FEEDBACK_LOOP else 0,
        "registered_users": len(events_storage['user_events']) if USE_FEEDBACK_LOOP else 0,
        "timestamp": datetime.now().isoformat()
    }
    return jsonify(status)

@app.route('/api/models', methods=['GET'])
def list_models():
    """List all available models and their basic information."""
    model_info = {}
    
    for name, model in models.items():
        model_info[name] = {
            "version": model.get('version', 'unknown'),
            "training_date": str(model.get('training_date', 'unknown')),
            "num_users": len(model['user_map']),
            "num_products": len(model['product_map']),
            "metrics": model.get('metrics', {}),
            "last_updated": model_last_updated.get(name, datetime.now()).isoformat()
        }
    
    return jsonify(model_info)

@app.route('/api/users', methods=['GET'])
def list_users():
    """List available user IDs (limited to first 100)."""
    if not models:
        return jsonify({"error": "No models loaded"})
    
    # Use the first available model to get user IDs
    model_name = list(models.keys())[0]
    users = list(models[model_name]['user_map'].keys())
    
    # Add new users from feedback system if available
    if USE_FEEDBACK_LOOP:
        for user_id in events_storage['user_events'].keys():
            if user_id not in users:
                users.append(user_id)
    
    return jsonify({
        "total_users": len(users),
        "users": users[:100]  # Limit to first 100 to avoid overwhelming response
    })

@app.route('/api/recommend/<model_name>/<user_id>', methods=['GET'])
def recommend(model_name, user_id):
    """Get recommendations for a user using specified model."""
    try:
        top_n = int(request.args.get('n', 10))
    except ValueError:
        top_n = 10
    
    # Track this recommendation request event
    track_event(
        user_id=user_id,
        product_id=None,
        event_type="recommendation_request",
        metadata={"model": model_name, "top_n": top_n}
    )
    
    recommendations = get_recommendations(model_name, user_id, top_n)
    
    if isinstance(recommendations, dict) and 'error' in recommendations:
        return jsonify(recommendations), 404
    
    return jsonify({
        "user_id": user_id,
        "model": model_name,
        "recommendations": recommendations,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/similar-products/<model_name>/<product_id>', methods=['GET'])
def similar_products(model_name, product_id):
    """Get similar products to a given product using specified model."""
    if model_name not in models:
        return jsonify({"error": f"Model {model_name} not loaded"}), 404
    
    model = models[model_name]
    
    # Check if product exists in the model
    if product_id not in model['product_map']:
        return jsonify({"error": f"Product {product_id} not found in the model"}), 404
    
    try:
        top_n = int(request.args.get('n', 10))
    except ValueError:
        top_n = 10
    
    # Track this similar products request event
    track_event(
        user_id=request.headers.get('X-User-ID', 'anonymous'),
        product_id=product_id,
        event_type="similar_products_request",
        metadata={"model": model_name, "top_n": top_n}
    )
    
    product_idx = model['product_map'][product_id]
    product_vec = model['item_factors'][product_idx]
    
    # Compute similarity scores with all other products
    all_product_vecs = model['item_factors']
    similarity_scores = np.dot(all_product_vecs, product_vec)
    
    # Get top-N similar products (excluding the query product)
    similarity_scores[product_idx] = -np.inf  # Exclude the query product
    top_indices = np.argsort(-similarity_scores)[:top_n]
    
    # Map indices back to product IDs
    idx_to_product = {idx: prod_id for prod_id, idx in model['product_map'].items()}
    similar_products_list = [idx_to_product[idx] for idx in top_indices]
    
    return jsonify({
        "product_id": product_id,
        "model": model_name,
        "similar_products": similar_products_list,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/track-event', methods=['POST'])
def track_user_event():
    """Endpoint to track user events."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.json
    required_fields = ['user_id', 'event_type']
    
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    user_id = data['user_id']
    event_type = data['event_type']
    product_id = data.get('product_id')
    metadata = data.get('metadata', {})
    
    # Handle new users automatically
    if event_type in ["view", "purchase", "add_to_cart"] and product_id:
        if USE_FEEDBACK_LOOP and user_id not in events_storage['user_events']:
            handle_new_user(user_id)
    
    # Track the event
    track_event(user_id, product_id, event_type, metadata)
    
    return jsonify({
        "status": "success",
        "message": "Event tracked successfully",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/clear-cache', methods=['POST'])
def clear_cache():
    """Endpoint to clear recommendation cache."""
    global recommendation_cache
    recommendation_cache = {}
    return jsonify({
        "status": "success",
        "message": "Cache cleared successfully",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/reload-models', methods=['POST'])
def reload_models():
    """Endpoint to reload models from disk."""
    load_models()
    return jsonify({
        "status": "success",
        "message": "Models reloaded successfully",
        "models_loaded": list(models.keys()),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/feedback-stats', methods=['GET'])
def feedback_stats():
    """Get statistics about the feedback system."""
    if not USE_FEEDBACK_LOOP:
        return jsonify({
            "status": "disabled",
            "message": "Feedback loop is not enabled"
        })
    
    # Count events by type
    event_counts = defaultdict(int)
    for user_events in events_storage['user_events'].values():
        for event in user_events:
            event_counts[event['event_type']] += 1
    
    # Get top viewed products
    top_products = sorted(
        events_storage['product_views'].items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:10]
    
    # Get active users
    active_users = sorted(
        [(user_id, len(products)) for user_id, products in events_storage['user_products'].items()],
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    return jsonify({
        "status": "active",
        "users_count": len(events_storage['user_events']),
        "products_count": len(events_storage['product_views']),
        "total_events": sum(event_counts.values()),
        "event_distribution": dict(event_counts),
        "top_products": [{"product_id": p[0], "views": p[1]} for p in top_products],
        "active_users": [{"user_id": u[0], "interactions": u[1]} for u in active_users],
        "cache_size": len(recommendation_cache),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/update-recommendations', methods=['POST'])
def update_recommendations():
    """Manually trigger the recommendation update process."""
    if not USE_FEEDBACK_LOOP:
        return jsonify({
            "status": "disabled",
            "message": "Feedback loop is not enabled"
        })
    
    try:
        # Run the update
        updated_users = feedback_loop.update_recommendations(models, cache_size=1000)
        
        return jsonify({
            "status": "success",
            "message": f"Recommendations updated for {updated_users} users",
            "updated_users": updated_users,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to update recommendations: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """
    Get recommendation quality metrics for Prometheus to scrape.
    This endpoint provides metrics about the recommendation system performance
    and quality that can be visualized in Grafana.
    """
    # Get the latest model metrics
    metrics_data = {}
    
    # Aggregate metrics from all models
    for name, model in models.items():
        if 'metrics' in model:
            for metric_name, metric_value in model['metrics'].items():
                if metric_name not in metrics_data:
                    metrics_data[metric_name] = []
                metrics_data[metric_name].append(metric_value)
    
    # Calculate average metrics across models
    avg_metrics = {}
    for metric_name, values in metrics_data.items():
        if values:
            try:
                # Convert all values to float before summing
                numeric_values = [float(v) for v in values]
                avg_metrics[metric_name] = sum(numeric_values) / len(numeric_values)
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not calculate average for metric {metric_name}: {e}")
                # If conversion fails, store the first value as a fallback
                avg_metrics[metric_name] = values[0]
    
    # Include response time metrics if available
    if hasattr(g, 'request_times') and g.request_times:
        avg_metrics['avg_response_time'] = sum(g.request_times) / len(g.request_times)
    
    # Calculate diversity and coverage from recent recommendations
    if recommendation_cache:
        all_recommended_products = []
        for recs in recommendation_cache.values():
            all_recommended_products.extend(recs)
        
        # Calculate diversity (unique products / total recommendations)
        if all_recommended_products:
            unique_products = len(set(all_recommended_products))
            total_products = len(all_recommended_products)
            avg_metrics['diversity'] = unique_products / total_products
            
            # Calculate coverage (unique products recommended / total products in catalog)
            total_catalog_products = 0
            for model_data in models.values():
                total_catalog_products = max(total_catalog_products, len(model_data.get('product_map', {})))
            
            if total_catalog_products > 0:
                avg_metrics['coverage'] = unique_products / total_catalog_products
    
    # Calculate personalization based on recommendation similarity between users
    # Higher value means more personalized (less similar recommendations between users)
    if len(recommendation_cache) > 1:
        similarities = []
        cache_items = list(recommendation_cache.items())
        for i in range(len(cache_items)):
            for j in range(i + 1, len(cache_items)):
                set1 = set(cache_items[i][1])
                set2 = set(cache_items[j][1])
                if set1 and set2:  # Non-empty sets
                    jaccard = len(set1.intersection(set2)) / len(set1.union(set2))
                    similarities.append(jaccard)
        
        if similarities:
            avg_metrics['personalization'] = 1 - (sum(similarities) / len(similarities))
    
    # Add timestamp
    avg_metrics['timestamp'] = datetime.now().isoformat()
    
    return jsonify(avg_metrics)

# Initialize Kafka producers and processors
if KAFKA_AVAILABLE:
    initialize_kafka()

# Load models with app context instead of using before_first_request
# which was removed in Flask 2.0+
with app.app_context():
    models = load_models()

if __name__ == '__main__':
    # Run the Flask application
    app.run(host='0.0.0.0', port=5050, debug=True)  # Changed port from 5000 to 5050 