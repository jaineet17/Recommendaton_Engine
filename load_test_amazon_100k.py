#!/usr/bin/env python3
"""
Large-scale Load Testing Script for Amazon Recommendation Engine

This script:
1. Generates synthetic data for 100,000 users (or uses real Amazon review data)
2. Tests system performance under high load
3. Measures latency and throughput
4. Evaluates recommendation accuracy and quality
5. Monitors all system components (Kafka, Prometheus, etc.)

Requirements:
    - requests
    - numpy
    - pandas
    - matplotlib
    - scipy
    - kafka-python
    - prometheus-client
    - psutil
    - tqdm
"""

import os
import sys
import time
import json
import random
import string
import logging
import argparse
import threading
import concurrent.futures
import traceback
from datetime import datetime, timedelta
from collections import defaultdict, Counter

# Data processing and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# HTTP requests and system monitoring
import requests
from tqdm import tqdm
import psutil

# Try to import Kafka client - optional
try:
    from kafka import KafkaConsumer, KafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    print("Warning: Kafka Python client not installed. Kafka testing will be skipped.")

# Try to import Prometheus client - optional
try:
    from prometheus_client import CollectorRegistry, Counter, Gauge, push_to_gateway
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("Warning: Prometheus client not installed. Prometheus testing will be skipped.")

# Set up argument parser
parser = argparse.ArgumentParser(description='Run a large-scale load test for the recommendation engine')
parser.add_argument('--api-url', type=str, default='http://localhost:5050/api', help='Base URL for the API')
parser.add_argument('--kafka-broker', type=str, default='localhost:29092', help='Kafka broker address')
parser.add_argument('--prometheus-url', type=str, default='http://localhost:9090', help='Prometheus server URL')
parser.add_argument('--prometheus-metrics-port', type=int, default=8001, help='Prometheus metrics port')
parser.add_argument('--grafana-url', type=str, default='http://localhost:3000', help='Grafana URL')
parser.add_argument('--jenkins-url', type=str, default='http://localhost:8080', help='Jenkins URL')
parser.add_argument('--users', type=int, default=100000, help='Number of users to generate')
parser.add_argument('--products', type=int, default=50000, help='Number of products to generate')
parser.add_argument('--interactions', type=int, default=1000000, help='Number of interactions to generate')
parser.add_argument('--amazon-data', type=str, default='', help='Path to Amazon review dataset (optional)')
parser.add_argument('--concurrency', type=int, default=50, help='Number of concurrent requests')
parser.add_argument('--duration', type=int, default=120, help='Duration of load test in seconds')
parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for processing')
parser.add_argument('--output-dir', type=str, default='load_test_results', help='Output directory for test results')
args = parser.parse_args()

# Set up logging
os.makedirs(args.output_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(args.output_dir, 'load_test.log')),
    ]
)
logger = logging.getLogger('load_test')

# Test results dictionary
test_results = {
    "timestamp": datetime.now().isoformat(),
    "api_url": args.api_url,
    "api_health": False,
    "component_health": {
        "api_server": False,
        "prometheus": False,
        "grafana": False,
        "jenkins": False,
        "kafka": False,
        "feedback_loop": False
    },
    "synthetic_data": {
        "users": 0,
        "products": 0,
        "interactions": 0
    },
    "performance": {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "success_rate": 0,
        "error_rate": 0,
        "throughput": 0,
        "avg_response_time": 0,
        "p50_response_time": 0,
        "p95_response_time": 0,
        "p99_response_time": 0,
        "min_response_time": 0,
        "max_response_time": 0
    },
    "recommendation_quality": {
        "diversity": 0,
        "coverage": 0,
        "novelty": 0,
        "serendipity": 0,
        "personalization": 0
    },
    "system_resources": {
        "peak_cpu": 0,
        "peak_memory": 0,
        "avg_cpu": 0,
        "avg_memory": 0
    }
}

# -------------------------------------------------------------------------
# DATA GENERATION FUNCTIONS
# -------------------------------------------------------------------------

def generate_user_id(index):
    """Generate a user ID based on an index."""
    return f"user_{index}"

def generate_product_id(index):
    """Generate a product ID based on an index."""
    return f"product_{index}"

def generate_users(count):
    """Generate synthetic user data."""
    logger.info(f"Generating {count} synthetic users...")
    
    users = []
    for i in range(1, count + 1):
        user_id = generate_user_id(i)
        # Additional user attributes could be added here
        users.append({"id": user_id})
    
    logger.info(f"Generated {len(users)} users")
    test_results["synthetic_data"]["users"] = len(users)
    return users

def generate_products(count):
    """Generate synthetic product data."""
    logger.info(f"Generating {count} synthetic products...")
    
    products = []
    # Define some product categories
    categories = ["Electronics", "Books", "Clothing", "Home", "Beauty", 
                "Sports", "Toys", "Grocery", "Automotive", "Health"]
    
    for i in range(1, count + 1):
        product_id = generate_product_id(i)
        # Create product with random attributes
        product = {
            "id": product_id,
            "category": random.choice(categories),
            "price": round(random.uniform(5.0, 500.0), 2),
            "popularity": random.random()  # Normalized popularity score
        }
        products.append(product)
    
    logger.info(f"Generated {len(products)} products")
    test_results["synthetic_data"]["products"] = len(products)
    return products

def generate_interactions(users, products, count):
    """Generate synthetic user-product interactions."""
    logger.info(f"Generating {count} synthetic interactions...")
    
    interactions = []
    event_types = ["view", "click", "add_to_cart", "purchase", "rate"]
    event_weights = [0.5, 0.3, 0.1, 0.05, 0.05]  # Probability distribution
    
    # Pareto distribution for user activity (80/20 rule)
    user_activity = np.random.pareto(1.16, len(users))
    user_activity = user_activity / np.sum(user_activity)
    
    # Zipf distribution for product popularity
    product_popularity = np.random.zipf(1.6, len(products))
    product_popularity = product_popularity / np.sum(product_popularity)
    
    # Generate interactions
    for _ in range(count):
        # Select user and product based on distributions
        user_idx = np.random.choice(len(users), p=user_activity)
        product_idx = np.random.choice(len(products), p=product_popularity)
        
        user = users[user_idx]
        product = products[product_idx]
        
        # Generate interaction
        event_type = np.random.choice(event_types, p=event_weights)
        timestamp = datetime.now() - timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        
        interaction = {
            "user_id": user["id"],
            "product_id": product["id"],
            "event_type": event_type,
            "timestamp": timestamp.isoformat(),
            "metadata": {
                "source": "synthetic_data"
            }
        }
        
        # Add rating value if it's a rating event
        if event_type == "rate":
            # Ratings follow a J-shaped distribution (more 5s, fewer 3s)
            ratings_dist = [0.05, 0.1, 0.2, 0.25, 0.4]  # 1-5 stars
            interaction["rating"] = np.random.choice([1, 2, 3, 4, 5], p=ratings_dist)
        
        interactions.append(interaction)
    
    logger.info(f"Generated {len(interactions)} interactions")
    test_results["synthetic_data"]["interactions"] = len(interactions)
    return interactions

def load_amazon_data(filepath, max_users=100000, max_products=50000):
    """Load and process real Amazon review data."""
    logger.info(f"Loading Amazon review data from {filepath}...")
    
    try:
        df = pd.read_json(filepath, lines=True)
        logger.info(f"Loaded {len(df)} reviews")
        
        # Extract user and product info
        users = df['reviewerID'].unique()
        products = df['asin'].unique()
        
        if len(users) > max_users:
            users = users[:max_users]
        if len(products) > max_products:
            products = products[:max_products]
        
        # Create user and product lists
        user_list = [{"id": user_id} for user_id in users]
        product_list = [{"id": product_id} for product_id in products]
        
        # Filter interactions to only include selected users and products
        df = df[df['reviewerID'].isin(users) & df['asin'].isin(products)]
        
        # Convert to interaction format
        interactions = []
        for _, row in df.iterrows():
            interaction = {
                "user_id": row['reviewerID'],
                "product_id": row['asin'],
                "event_type": "rate",
                "rating": row['overall'],
                "timestamp": str(row['unixReviewTime']),
                "metadata": {
                    "source": "amazon_data"
                }
            }
            interactions.append(interaction)
        
        logger.info(f"Processed {len(interactions)} interactions for {len(user_list)} users and {len(product_list)} products")
        
        test_results["synthetic_data"]["users"] = len(user_list)
        test_results["synthetic_data"]["products"] = len(product_list)
        test_results["synthetic_data"]["interactions"] = len(interactions)
        
        return user_list, product_list, interactions
    
    except Exception as e:
        logger.error(f"Error loading Amazon data: {str(e)}")
        logger.info("Falling back to synthetic data generation")
        return None, None, None

# -------------------------------------------------------------------------
# API AND COMPONENT TESTING
# -------------------------------------------------------------------------

def check_api_health():
    """Check if the API is healthy."""
    try:
        response = requests.get(f"{args.api_url}/health", timeout=15)
        response.raise_for_status()
        data = response.json()
        
        is_healthy = data.get("status") == "healthy"
        models_loaded = len(data.get("models_loaded", [])) > 0
        
        logger.info(f"API Health: {'Healthy' if is_healthy else 'Unhealthy'}")
        logger.info(f"Models loaded: {data.get('models_loaded', [])}")
        
        test_results["api_health"] = is_healthy and models_loaded
        test_results["component_health"]["api_server"] = is_healthy
        
        return test_results["api_health"]
    except requests.exceptions.ConnectTimeout:
        logger.error(f"API health check timed out. Check if the API server is running at {args.api_url}")
        test_results["api_health"] = False
        test_results["component_health"]["api_server"] = False
        return False
    except requests.exceptions.ConnectionError:
        logger.error(f"Cannot connect to API server at {args.api_url}. Check if it's running.")
        test_results["api_health"] = False
        test_results["component_health"]["api_server"] = False
        return False
    except Exception as e:
        logger.error(f"API health check failed: {str(e)}")
        logger.debug(traceback.format_exc())
        test_results["api_health"] = False
        test_results["component_health"]["api_server"] = False
        return False

def check_component_health():
    """Check the health of all system components."""
    components = {
        "prometheus": args.prometheus_url,
        "grafana": args.grafana_url,
        "jenkins": args.jenkins_url
    }
    
    for component, url in components.items():
        try:
            # In TEST_MODE, we won't fail hard on component checks
            test_mode = os.environ.get('TEST_MODE', 'false').lower() == 'true'
            
            # Special case for Prometheus - try multiple ports
            if component == "prometheus":
                try:
                    # Try the standard port first
                    response = requests.get(url, timeout=10)
                    is_healthy = response.status_code < 400
                except Exception:
                    # If that fails, try the metrics port
                    metrics_url = f"http://localhost:{args.prometheus_metrics_port}/metrics"
                    logger.info(f"Trying alternate Prometheus URL: {metrics_url}")
                    try:
                        response = requests.get(metrics_url, timeout=10)
                        is_healthy = response.status_code < 400
                    except Exception as e:
                        # Finally, try the exposed Docker port
                        docker_url = url.replace(":9090", ":19090")
                        logger.info(f"Trying Docker Prometheus URL: {docker_url}")
                        try:
                            response = requests.get(docker_url, timeout=10)
                            is_healthy = response.status_code < 400
                        except Exception:
                            if test_mode:
                                logger.warning(f"Prometheus not available in TEST_MODE: {str(e)}")
                                is_healthy = False
                            else:
                                is_healthy = False
            else:
                # Normal case for other components
                try:
                    response = requests.get(url, timeout=10)
                    is_healthy = response.status_code < 400
                except Exception as e:
                    if test_mode:
                        logger.warning(f"{component.capitalize()} not available in TEST_MODE: {str(e)}")
                        is_healthy = False
                    else:
                        is_healthy = False
                        
            logger.info(f"{component.capitalize()} Status: {'Available' if is_healthy else 'Unavailable'}")
            test_results["component_health"][component] = is_healthy
            
        except Exception as e:
            logger.error(f"Error checking {component}: {str(e)}")
            test_results["component_health"][component] = False
    
    # Check Kafka if available
    if KAFKA_AVAILABLE:
        try:
            # Try to create a simple consumer to check connection
            test_mode = os.environ.get('TEST_MODE', 'false').lower() == 'true'
            
            try:
                # Try to create a simple consumer to check connection
                consumer = KafkaConsumer(
                    bootstrap_servers=[args.kafka_broker],
                    auto_offset_reset='earliest',
                    consumer_timeout_ms=5000
                )
                kafka_available = True
                consumer.close()
            except Exception as e:
                if test_mode:
                    logger.warning(f"Kafka not available in TEST_MODE: {str(e)}")
                    kafka_available = False
                else:
                    kafka_available = False
        except Exception as e:
            logger.error(f"Kafka check failed: {str(e)}")
            kafka_available = False
        
        test_results["component_health"]["kafka"] = kafka_available
        logger.info(f"Kafka Status: {'Available' if kafka_available else 'Unavailable'}")
    else:
        logger.warning("Kafka client not available - skipping Kafka check")
        test_results["component_health"]["kafka"] = False

def check_feedback_loop(test_user_id, test_product_id):
    """Test the feedback loop by sending an event and checking for updates."""
    try:
        # Step 1: Get initial recommendations
        initial_response = requests.get(
            f"{args.api_url}/recommend/lightgcn/{test_user_id}?n=10",
            timeout=15
        )
        initial_response.raise_for_status()
        initial_recs = initial_response.json().get("recommendations", [])
        logger.info(f"Initial recommendations for user {test_user_id}: {initial_recs}")
        
        # Step 2: Send a strong interaction signal
        event_data = {
            "user_id": test_user_id,
            "product_id": test_product_id,
            "event_type": "purchase",
            "metadata": {
                "source": "load_test",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        event_response = requests.post(
            f"{args.api_url}/track-event",
            json=event_data,
            timeout=15
        )
        event_response.raise_for_status()
        logger.info(f"Sent purchase event for user {test_user_id} and product {test_product_id}")
        
        # Step 3: Force update of recommendations
        update_response = requests.post(
            f"{args.api_url}/update-recommendations",
            timeout=15
        )
        update_response.raise_for_status()
        logger.info("Triggered recommendations update")
        
        # Step 4: Wait for processing (depends on system config)
        logger.info("Waiting for feedback loop processing...")
        time.sleep(10)
        
        # Step 5: Get updated recommendations
        updated_response = requests.get(
            f"{args.api_url}/recommend/lightgcn/{test_user_id}?n=10",
            timeout=15
        )
        updated_response.raise_for_status()
        updated_recs = updated_response.json().get("recommendations", [])
        logger.info(f"Updated recommendations for user {test_user_id}: {updated_recs}")
        
        # Check if anything changed
        feedback_works = len(set(initial_recs) - set(updated_recs)) > 0 or len(set(updated_recs) - set(initial_recs)) > 0
        
        # Check if the test product is affecting related recommendations
        # Get related products
        related_products_response = requests.get(
            f"{args.api_url}/related-products/{test_product_id}?n=5",
            timeout=15
        )
        
        if related_products_response.status_code == 200:
            related_products = related_products_response.json().get("recommendations", [])
            logger.info(f"Related products to {test_product_id}: {related_products}")
            
            # Check if any related products are in the updated recommendations
            related_in_recs = any(product in updated_recs for product in related_products)
            feedback_works = feedback_works or related_in_recs
            
            if related_in_recs:
                logger.info("Found related products in updated recommendations - feedback loop working")
        
        test_results["component_health"]["feedback_loop"] = feedback_works
        
        logger.info(f"Feedback loop test: {'PASS' if feedback_works else 'FAIL'}")
        if feedback_works:
            logger.info("The recommendation changes after user interaction")
        else:
            logger.info("No observable change in recommendations after interaction")
        
        return feedback_works
    
    except Exception as e:
        logger.error(f"Feedback loop test failed: {str(e)}")
        logger.debug(traceback.format_exc())
        test_results["component_health"]["feedback_loop"] = False
        return False

# -------------------------------------------------------------------------
# LOAD TESTING
# -------------------------------------------------------------------------

def make_recommendation_request(user_id, model="lightgcn", n=10):
    """Make a recommendation request for a specific user."""
    start_time = time.time()
    success = False
    response_time = None
    recommendations = []
    
    try:
        response = requests.get(
            f"{args.api_url}/recommend/{model}/{user_id}?n={n}",
            timeout=20
        )
        response.raise_for_status()
        end_time = time.time()
        response_time = end_time - start_time
        
        data = response.json()
        recommendations = data.get("recommendations", [])
        success = len(recommendations) > 0
    
    except Exception as e:
        end_time = time.time()
        response_time = end_time - start_time
        logger.debug(f"Request failed for user {user_id}: {str(e)}")
    
    return {
        "user_id": user_id,
        "success": success,
        "response_time": response_time,
        "recommendations": recommendations
    }

def track_event(user_id, product_id, event_type="click"):
    """Track a user event."""
    start_time = time.time()
    success = False
    response_time = None
    
    try:
        event_data = {
            "user_id": user_id,
            "product_id": product_id,
            "event_type": event_type,
            "metadata": {
                "source": "load_test",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        response = requests.post(
            f"{args.api_url}/track-event",
            json=event_data,
            timeout=20
        )
        response.raise_for_status()
        end_time = time.time()
        response_time = end_time - start_time
        success = response.status_code == 200
    
    except Exception as e:
        end_time = time.time()
        response_time = end_time - start_time
        logger.debug(f"Event tracking failed for user {user_id}, product {product_id}: {str(e)}")
    
    return {
        "success": success,
        "response_time": response_time
    }

def run_load_test(users, products, interactions):
    """Run a load test with concurrent requests."""
    logger.info(f"Starting load test with {args.concurrency} concurrent users for {args.duration} seconds")
    
    start_time = time.time()
    end_time = start_time + args.duration
    
    # Set up metrics
    total_requests = 0
    successful_requests = 0
    response_times = []
    all_recommendations = []
    user_recommendations = defaultdict(list)
    
    # Resource monitoring
    cpu_usage = []
    memory_usage = []
    
    # Resource monitoring thread
    def monitor_resources():
        while time.time() < end_time:
            cpu_usage.append(psutil.cpu_percent(interval=1))
            memory_usage.append(psutil.virtual_memory().percent)
            time.sleep(1)
    
    # Start resource monitoring in a separate thread
    resource_thread = threading.Thread(target=monitor_resources)
    resource_thread.daemon = True
    resource_thread.start()
    
    # Prepare user batches for processing
    total_users = len(users)
    user_batches = []
    batch_size = min(args.batch_size, total_users)
    
    for i in range(0, total_users, batch_size):
        user_batches.append(users[i:i+batch_size])
    
    logger.info(f"Created {len(user_batches)} batches of users for processing")
    
    # Process user batches
    batch_index = 0
    
    # Run load test until duration is reached
    with tqdm(total=args.duration, desc="Load Testing") as pbar:
        while time.time() < end_time:
            # Update progress bar
            elapsed = time.time() - start_time
            pbar.update(min(1, elapsed - pbar.n))
            
            # Get the current batch of users
            if batch_index >= len(user_batches):
                batch_index = 0  # Wrap around to the first batch if we've processed all batches
                
            current_batch = user_batches[batch_index]
            batch_index += 1
            
            # Create a set of users for this iteration
            batch_users = random.sample(current_batch, min(args.concurrency, len(current_batch)))
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                # Submit recommendation requests
                rec_futures = [executor.submit(make_recommendation_request, user["id"]) for user in batch_users]
                
                # Process recommendation results
                for future in concurrent.futures.as_completed(rec_futures):
                    result = future.result()
                    total_requests += 1
                    
                    if result["success"]:
                        successful_requests += 1
                        response_times.append(result["response_time"])
                        all_recommendations.extend(result["recommendations"])
                        user_recommendations[result["user_id"]].extend(result["recommendations"])
                
                # Submit some event tracking requests if we have interactions
                if interactions:
                    # Sample some interactions
                    sample_size = min(args.concurrency // 2, len(interactions))
                    if sample_size > 0:
                        sample_interactions = random.sample(interactions, sample_size)
                        
                        # Submit event tracking requests
                        event_futures = [
                            executor.submit(track_event, 
                                          interaction["user_id"], 
                                          interaction["product_id"], 
                                          interaction["event_type"])
                            for interaction in sample_interactions
                        ]
                        
                        # Process event tracking results
                        for future in concurrent.futures.as_completed(event_futures):
                            result = future.result()
                            total_requests += 1
                            
                            if result["success"]:
                                successful_requests += 1
                                response_times.append(result["response_time"])
            
            # Small delay to prevent CPU overload
            time.sleep(0.1)
    
    # Calculate performance metrics
    test_duration = time.time() - start_time
    
    if response_times:
        avg_response_time = np.mean(response_times)
        p50_response_time = np.percentile(response_times, 50)
        p95_response_time = np.percentile(response_times, 95)
        p99_response_time = np.percentile(response_times, 99)
        max_response_time = max(response_times)
        min_response_time = min(response_times)
    else:
        avg_response_time = p50_response_time = p95_response_time = p99_response_time = max_response_time = min_response_time = 0
    
    # Update test results
    test_results["performance"]["total_requests"] = total_requests
    test_results["performance"]["successful_requests"] = successful_requests
    test_results["performance"]["failed_requests"] = total_requests - successful_requests
    test_results["performance"]["success_rate"] = successful_requests / total_requests if total_requests > 0 else 0
    test_results["performance"]["error_rate"] = 1 - test_results["performance"]["success_rate"]
    test_results["performance"]["throughput"] = successful_requests / test_duration if test_duration > 0 else 0
    test_results["performance"]["avg_response_time"] = avg_response_time
    test_results["performance"]["p50_response_time"] = p50_response_time
    test_results["performance"]["p95_response_time"] = p95_response_time
    test_results["performance"]["p99_response_time"] = p99_response_time
    test_results["performance"]["min_response_time"] = min_response_time
    test_results["performance"]["max_response_time"] = max_response_time
    
    # Calculate resource usage
    if cpu_usage:
        test_results["system_resources"]["peak_cpu"] = max(cpu_usage)
        test_results["system_resources"]["avg_cpu"] = sum(cpu_usage) / len(cpu_usage)
    
    if memory_usage:
        test_results["system_resources"]["peak_memory"] = max(memory_usage)
        test_results["system_resources"]["avg_memory"] = sum(memory_usage) / len(memory_usage)
    
    # Calculate recommendation quality metrics
    if all_recommendations:
        # Diversity - ratio of unique recommendations to total recommendations
        unique_recs = len(set(all_recommendations))
        total_recs = len(all_recommendations)
        test_results["recommendation_quality"]["diversity"] = unique_recs / total_recs if total_recs > 0 else 0
        
        # Coverage - percentage of product catalog covered by recommendations
        product_ids = [p["id"] for p in products]
        test_results["recommendation_quality"]["coverage"] = unique_recs / len(product_ids) if product_ids else 0
        
        # Personalization - how different recommendations are between users
        # Higher value means more personalized recommendations
        user_similarity_scores = []
        user_ids = list(user_recommendations.keys())
        
        if len(user_ids) > 1:
            for i in range(len(user_ids)):
                for j in range(i+1, len(user_ids)):
                    user1_recs = set(user_recommendations[user_ids[i]])
                    user2_recs = set(user_recommendations[user_ids[j]])
                    
                    if user1_recs and user2_recs:
                        # Jaccard similarity - intersection over union
                        similarity = len(user1_recs.intersection(user2_recs)) / len(user1_recs.union(user2_recs))
                        user_similarity_scores.append(similarity)
            
            # Personalization is 1 - average similarity (higher is better)
            if user_similarity_scores:
                test_results["recommendation_quality"]["personalization"] = 1 - np.mean(user_similarity_scores)
            else:
                test_results["recommendation_quality"]["personalization"] = 0
        else:
            test_results["recommendation_quality"]["personalization"] = 0
    
    return response_times, total_requests, successful_requests

def generate_performance_charts(response_times):
    """Generate charts for performance analysis."""
    logger.info("Generating performance charts...")
    
    try:
        plt.figure(figsize=(15, 10))
        
        # Response time distribution
        plt.subplot(2, 2, 1)
        sns.histplot(response_times, kde=True)
        plt.title('Response Time Distribution')
        plt.xlabel('Response Time (seconds)')
        plt.ylabel('Frequency')
        
        # Response time percentiles
        plt.subplot(2, 2, 2)
        percentiles = [50, 75, 90, 95, 99]
        percentile_values = [np.percentile(response_times, p) for p in percentiles]
        
        plt.bar([str(p) + "%" for p in percentiles], percentile_values)
        plt.title('Response Time Percentiles')
        plt.xlabel('Percentile')
        plt.ylabel('Response Time (seconds)')
        
        # Recommendation quality metrics
        plt.subplot(2, 2, 3)
        quality_metrics = [
            test_results["recommendation_quality"]["diversity"],
            test_results["recommendation_quality"]["coverage"],
            test_results["recommendation_quality"]["personalization"]
        ]
        metric_names = ["Diversity", "Coverage", "Personalization"]
        
        plt.bar(metric_names, quality_metrics)
        plt.title('Recommendation Quality Metrics')
        plt.xlabel('Metric')
        plt.ylabel('Score (0-1)')
        plt.ylim(0, 1)
        
        # Component health
        plt.subplot(2, 2, 4)
        component_status = [int(status) for status in test_results["component_health"].values()]
        component_names = list(test_results["component_health"].keys())
        
        plt.bar(component_names, component_status)
        plt.title('System Component Status')
        plt.xlabel('Component')
        plt.ylabel('Status (1=Healthy)')
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'load_test_performance.png'))
        logger.info(f"Performance charts saved to {os.path.join(args.output_dir, 'load_test_performance.png')}")
    
    except Exception as e:
        logger.error(f"Error generating performance charts: {str(e)}")

def export_results_to_csv():
    """Export test results to CSV files."""
    try:
        # Save summary metrics
        summary = {
            "metric": [],
            "value": []
        }
        
        # Performance metrics
        for key, value in test_results["performance"].items():
            summary["metric"].append(f"performance_{key}")
            summary["value"].append(value)
        
        # Recommendation quality metrics
        for key, value in test_results["recommendation_quality"].items():
            summary["metric"].append(f"quality_{key}")
            summary["value"].append(value)
        
        # Component health metrics
        for key, value in test_results["component_health"].items():
            summary["metric"].append(f"component_{key}")
            summary["value"].append(int(value))
        
        # System resources metrics
        for key, value in test_results["system_resources"].items():
            summary["metric"].append(f"resource_{key}")
            summary["value"].append(value)
        
        # Save summary
        df_summary = pd.DataFrame(summary)
        df_summary.to_csv(os.path.join(args.output_dir, "test_summary.csv"), index=False)
        
        logger.info(f"Test results exported to CSV files in {args.output_dir}")
    
    except Exception as e:
        logger.error(f"Error exporting results to CSV: {str(e)}")
        logger.debug(traceback.format_exc())

def send_events_to_kafka(interactions):
    """Send interaction events to Kafka."""
    if not KAFKA_AVAILABLE:
        logger.warning("Kafka client not available - skipping sending events to Kafka")
        return

    try:
        logger.info(f"Sending {len(interactions)} events to Kafka...")
        
        # Create a custom JSON serializer that handles NumPy types
        def json_serializer(data):
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (np.integer, np.int64)):
                        return int(obj)
                    if isinstance(obj, (np.floating, np.float64)):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super().default(obj)
            
            return json.dumps(data, cls=NumpyEncoder).encode('utf-8')
        
        # Create a Kafka producer with the custom serializer
        producer = KafkaProducer(
            bootstrap_servers=[args.kafka_broker],
            value_serializer=json_serializer
        )
        
        # Send a sample of interactions to various topics
        sample_size = min(1000, len(interactions))
        sample_interactions = random.sample(interactions, sample_size)
        
        events_sent = 0
        topics = {
            "view": "amazon-product-views",
            "click": "amazon-user-events",
            "add_to_cart": "amazon-user-events",
            "purchase": "amazon-purchases",
            "rate": "amazon-reviews"
        }
        
        for interaction in sample_interactions:
            event_type = interaction.get("event_type")
            topic = topics.get(event_type, "amazon-user-events")
            
            # Send message to appropriate topic
            future = producer.send(topic, interaction)
            events_sent += 1
            
            # Don't slow down the test too much, just send the events
            if events_sent % 100 == 0:
                producer.flush()  # Ensure messages are sent
                logger.info(f"Sent {events_sent}/{sample_size} events to Kafka")
        
        # Make sure all messages are sent before continuing
        producer.flush()
        logger.info(f"Successfully sent {events_sent} events to Kafka")
        
        # Close the producer
        producer.close()
        
    except Exception as e:
        logger.error(f"Error sending events to Kafka: {str(e)}")
        logger.debug(traceback.format_exc())

# -------------------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------------------

def fetch_real_users():
    """Fetch real users from the API."""
    logger.info("Fetching real users from the API...")
    try:
        response = requests.get(f"{args.api_url.rstrip('/api')}/api/users", timeout=10)
        response.raise_for_status()
        data = response.json()
        users = [{"id": user_id} for user_id in data.get("users", [])]
        logger.info(f"Fetched {len(users)} real users from the API")
        test_results["synthetic_data"]["users"] = len(users)
        return users
    except Exception as e:
        logger.error(f"Error fetching users from API: {str(e)}")
        return []

def main():
    """Main execution function."""
    logger.info("="*80)
    logger.info("STARTING LARGE-SCALE LOAD TEST FOR RECOMMENDATION ENGINE")
    logger.info(f"Target: {args.api_url}")
    logger.info(f"Users: {args.users}, Products: {args.products}, Interactions: {args.interactions}")
    logger.info(f"Concurrency: {args.concurrency}, Duration: {args.duration} seconds")
    logger.info("="*80)
    
    # Step 1: Check if API is healthy
    if not check_api_health():
        logger.error("API is not healthy. Aborting test.")
        return
    
    # Step 2: Check component health
    check_component_health()
    
    # Step 3: Generate or load data
    # First try to fetch real users from the API to improve success rate
    users = fetch_real_users()
    
    # If we couldn't get real users or got too few, fall back to synthetic
    if not users or len(users) < min(10, args.users):
        # For large-scale testing, use batching to avoid memory issues
        if args.users > 10000:
            logger.info(f"Large-scale test with {args.users} users - using batched processing")
            
        if args.amazon_data and os.path.exists(args.amazon_data):
            users, products, interactions = load_amazon_data(
                args.amazon_data, 
                max_users=args.users,
                max_products=args.products
            )
            if not users:
                # Fall back to synthetic data if amazon data loading failed
                users = generate_users(args.users)
                products = generate_products(args.products)
                interactions = generate_interactions(users, products, args.interactions)
        else:
            users = generate_users(args.users)
            products = generate_products(args.products)
            interactions = generate_interactions(users, products, args.interactions)
    else:
        # We got real users, just generate products and interactions
        products = generate_products(args.products)
        interactions = generate_interactions(users, products, args.interactions)
    
    # Step 4: Test Kafka streaming (if available)
    if KAFKA_AVAILABLE and test_results["component_health"]["kafka"]:
        send_events_to_kafka(interactions)
    
    # Step 5: Test feedback loop with a sample user
    test_user = users[0]
    test_product = products[0]
    check_feedback_loop(test_user["id"], test_product["id"])
    
    # Step 6: Run load test
    response_times, total_requests, successful_requests = run_load_test(users, products, interactions)
    
    # Step 7: Generate charts
    if response_times:
        generate_performance_charts(response_times)
    
    # Step 8: Export results
    export_results_to_csv()
    
    # Step 9: Final summary
    logger.info("="*80)
    logger.info("LOAD TEST COMPLETE")
    logger.info("="*80)
    logger.info(f"Total Requests: {total_requests}")
    logger.info(f"Successful Requests: {successful_requests}")
    logger.info(f"Success Rate: {test_results['performance']['success_rate']:.2%}")
    logger.info(f"Throughput: {test_results['performance']['throughput']:.2f} requests/second")
    
    if response_times:
        logger.info(f"Average Response Time: {np.mean(response_times):.3f} seconds")
        logger.info(f"95th Percentile Response Time: {np.percentile(response_times, 95):.3f} seconds")
    
    component_status = sum(test_results["component_health"].values())
    total_components = len(test_results["component_health"])
    logger.info(f"System Health: {component_status}/{total_components} components healthy")
    
    logger.info(f"Results saved to {args.output_dir}")
    logger.info("="*80)

if __name__ == "__main__":
    main() 