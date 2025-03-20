#!/usr/bin/env python3
"""
Production Test Script for Recommendation Engine - 100,000 User Scale

This script performs comprehensive production testing of the recommendation system at scale:
1. Generates synthetic user data for 100,000 users (or loads real data if available)
2. Tests system performance under high load conditions
3. Measures and reports latency, throughput, and error rates
4. Evaluates recommendation quality across different models
5. Generates detailed performance reports and visualizations

Requirements:
    - requests
    - numpy
    - pandas
    - matplotlib
    - tqdm
    - psutil
"""

import os
import sys
import time
import json
import random
import logging
import argparse
import threading
import traceback
import concurrent.futures
from datetime import datetime, timedelta
from collections import defaultdict, Counter

# Data processing and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# System monitoring
import psutil

# For API requests
import requests

# Set up argument parser
parser = argparse.ArgumentParser(description='Run a production test of the recommendation engine at scale')
parser.add_argument('--api-url', type=str, default='http://localhost:5050/api', help='Base URL for the API')
parser.add_argument('--users', type=int, default=100000, help='Number of users to simulate')
parser.add_argument('--products', type=int, default=10000, help='Number of products to simulate')
parser.add_argument('--concurrency', type=int, default=50, help='Number of concurrent requests')
parser.add_argument('--duration', type=int, default=300, help='Duration of test in seconds')
parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for processing')
parser.add_argument('--data-file', type=str, default='', help='Path to real user data (optional)')
parser.add_argument('--output-dir', type=str, default='production_test_results', help='Output directory')
parser.add_argument('--models', type=str, default='lightgcn,ncf,simple_mf,content,ensemble', 
                   help='Comma-separated list of models to test')
parser.add_argument('--test-new-users', action='store_true', help='Test cold start for new users')
args = parser.parse_args()

# Set up logging
os.makedirs(args.output_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(args.output_dir, 'production_test.log')),
    ]
)
logger = logging.getLogger('production_test')

# Parse models to test
models_to_test = [model.strip() for model in args.models.split(',')]
logger.info(f"Models to test: {models_to_test}")

# Test results dictionary
test_results = {
    "timestamp": datetime.now().isoformat(),
    "api_url": args.api_url,
    "models_tested": models_to_test,
    "api_health": False,
    "synthetic_data": {
        "users": 0,
        "products": 0,
        "interactions": 0
    },
    "model_performance": {},
    "system_resources": {
        "peak_cpu": 0,
        "peak_memory": 0,
        "avg_cpu": 0,
        "avg_memory": 0
    }
}

# Initialize model performance dictionaries
for model in models_to_test:
    test_results["model_performance"][model] = {
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
        "max_response_time": 0,
        "quality": {
            "diversity": 0,
            "coverage": 0,
            "novelty": 0,
            "personalization": 0
        }
    }

# -------------------------------------------------------------------------
# DATA GENERATION FUNCTIONS
# -------------------------------------------------------------------------

def generate_user_id(index):
    """Generate a user ID based on an index."""
    return f"test_user_{index}"

def generate_product_id(index):
    """Generate a product ID based on an index."""
    return f"test_product_{index}"

def generate_users(count):
    """Generate synthetic user data."""
    logger.info(f"Generating {count} synthetic users...")
    
    users = []
    for i in range(1, count + 1):
        user_id = generate_user_id(i)
        users.append({"id": user_id})
    
    logger.info(f"Generated {len(users)} users")
    test_results["synthetic_data"]["users"] = len(users)
    return users

def generate_products(count):
    """Generate synthetic product data."""
    logger.info(f"Generating {count} synthetic products...")
    
    products = []
    # Define product categories
    categories = ["Electronics", "Books", "Clothing", "Home", "Beauty", 
                 "Sports", "Toys", "Grocery", "Automotive", "Health"]
    
    for i in range(1, count + 1):
        product_id = generate_product_id(i)
        product = {
            "id": product_id,
            "category": random.choice(categories),
            "price": round(random.uniform(5.0, 500.0), 2),
            "popularity": random.random()
        }
        products.append(product)
    
    logger.info(f"Generated {len(products)} products")
    test_results["synthetic_data"]["products"] = len(products)
    return products

def load_real_data(filepath, max_users=100000):
    """Load real user data from a file if available."""
    logger.info(f"Loading real user data from {filepath}...")
    
    try:
        df = pd.read_parquet(filepath)
        logger.info(f"Loaded {len(df)} records")
        
        # Extract user and product IDs
        users = df['user_id'].unique()
        products = df['product_id'].unique()
        
        if len(users) > max_users:
            users = users[:max_users]
        
        # Create user and product lists
        user_list = [{"id": user_id} for user_id in users]
        product_list = [{"id": product_id} for product_id in products]
        
        logger.info(f"Processed {len(user_list)} users and {len(product_list)} products")
        
        test_results["synthetic_data"]["users"] = len(user_list)
        test_results["synthetic_data"]["products"] = len(product_list)
        
        return user_list, product_list
    
    except Exception as e:
        logger.error(f"Error loading real data: {str(e)}")
        logger.info("Falling back to synthetic data generation")
        return None, None

def fetch_real_users():
    """Fetch real users from the API."""
    logger.info("Fetching real users from the API...")
    try:
        response = requests.get(f"{args.api_url}/users", timeout=10)
        response.raise_for_status()
        data = response.json()
        users = [{"id": user_id} for user_id in data.get("users", [])]
        logger.info(f"Fetched {len(users)} real users from the API")
        test_results["synthetic_data"]["users"] = len(users)
        return users
    except Exception as e:
        logger.error(f"Error fetching users from API: {str(e)}")
        return []

# -------------------------------------------------------------------------
# API TESTING FUNCTIONS
# -------------------------------------------------------------------------

def check_api_health():
    """Check if the API is healthy and which models are available."""
    try:
        # Try different health endpoint paths
        endpoints = [
            f"{args.api_url}/health",
            f"{args.api_url.rstrip('/api')}/health",
            f"{args.api_url.rstrip('/api')}/api/health",
            "http://localhost:5050/health",
            "http://localhost:5050/api/health"
        ]
        
        logger.info(f"Attempting to connect to API via multiple health endpoints...")
        
        for endpoint in endpoints:
            try:
                logger.info(f"Trying endpoint: {endpoint}")
                response = requests.get(endpoint, timeout=5)
                
                if response.status_code < 400:
                    logger.info(f"Successfully connected to {endpoint}")
                    data = response.json()
                    
                    is_healthy = data.get("status") == "healthy"
                    available_models = data.get("models_loaded", [])
                    
                    logger.info(f"API Health: {'Healthy' if is_healthy else 'Unhealthy'}")
                    logger.info(f"Models loaded: {available_models}")
                    
                    # Update test results
                    test_results["api_health"] = is_healthy
                    
                    # Check if requested models are available
                    missing_models = [model for model in models_to_test if model not in available_models]
                    if missing_models:
                        logger.warning(f"Requested models not available: {missing_models}")
                    
                    return is_healthy, available_models
                else:
                    logger.warning(f"Endpoint {endpoint} returned status code {response.status_code}")
            
            except Exception as e:
                logger.warning(f"Failed to connect to {endpoint}: {str(e)}")
                continue
        
        # If we get here, none of the endpoints worked
        logger.error("Failed to connect to API via any of the attempted endpoints")
        test_results["api_health"] = False
        return False, []
        
    except Exception as e:
        logger.error(f"API health check failed: {str(e)}")
        test_results["api_health"] = False
        return False, []

def make_recommendation_request(user_id, model="lightgcn", n=10):
    """Make a recommendation request for a specific user and model."""
    start_time = time.time()
    success = False
    response_time = None
    recommendations = []
    
    try:
        # Try both possible endpoints
        endpoints = [
            f"{args.api_url}/recommend/{model}/{user_id}?n={n}",
            f"{args.api_url.rstrip('/api')}/api/recommend/{model}/{user_id}?n={n}"
        ]
        
        for endpoint in endpoints:
            try:
                response = requests.get(endpoint, timeout=20)
                if response.status_code < 400:
                    end_time = time.time()
                    response_time = end_time - start_time
                    
                    data = response.json()
                    recommendations = data.get("recommendations", [])
                    success = len(recommendations) > 0
                    break
            except:
                continue
        
        if not success:
            end_time = time.time()
            response_time = end_time - start_time
    
    except Exception as e:
        end_time = time.time()
        response_time = end_time - start_time
        logger.debug(f"Request failed for user {user_id}, model {model}: {str(e)}")
    
    return {
        "user_id": user_id,
        "model": model,
        "success": success,
        "response_time": response_time,
        "recommendations": recommendations
    }

def test_cold_start(model="lightgcn", n=10):
    """Test the system's ability to handle new users (cold start)."""
    logger.info(f"Testing cold start recommendations for model {model}...")
    
    # Generate a completely new user ID
    new_user_id = f"new_user_{int(time.time())}"
    
    # Try to get recommendations for this new user
    result = make_recommendation_request(new_user_id, model, n)
    
    if result["success"]:
        logger.info(f"Cold start test PASSED for model {model}. Got {len(result['recommendations'])} recommendations")
    else:
        logger.warning(f"Cold start test FAILED for model {model}. No recommendations returned.")
    
    return result

# -------------------------------------------------------------------------
# LOAD TESTING FUNCTIONS
# -------------------------------------------------------------------------

def run_model_load_test(model, users, duration, concurrency):
    """Run a load test for a specific model."""
    logger.info(f"Starting load test for model {model} with {concurrency} concurrent users for {duration} seconds")
    
    start_time = time.time()
    end_time = start_time + duration
    
    # Metrics
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
    
    # Start resource monitoring
    resource_thread = threading.Thread(target=monitor_resources)
    resource_thread.daemon = True
    resource_thread.start()
    
    # Prepare user batches for processing
    total_users = len(users)
    user_batches = []
    batch_size = min(args.batch_size, total_users)
    
    for i in range(0, total_users, batch_size):
        user_batches.append(users[i:i+batch_size])
    
    logger.info(f"Created {len(user_batches)} batches of users for testing model {model}")
    
    # Process user batches
    batch_index = 0
    
    # Run load test until duration is reached
    with tqdm(total=duration, desc=f"Testing {model}") as pbar:
        while time.time() < end_time:
            # Update progress bar
            elapsed = time.time() - start_time
            pbar.update(min(1, elapsed - pbar.n))
            
            # Get the current batch of users
            if batch_index >= len(user_batches):
                batch_index = 0  # Wrap around to the first batch
                
            current_batch = user_batches[batch_index]
            batch_index += 1
            
            # Create a set of users for this iteration
            batch_users = random.sample(current_batch, min(concurrency, len(current_batch)))
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                # Submit recommendation requests
                futures = [
                    executor.submit(make_recommendation_request, user["id"], model) 
                    for user in batch_users
                ]
                
                # Process results
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    total_requests += 1
                    
                    if result["success"]:
                        successful_requests += 1
                        response_times.append(result["response_time"])
                        all_recommendations.extend(result["recommendations"])
                        user_recommendations[result["user_id"]].extend(result["recommendations"])
            
            # Small delay to prevent CPU overload
            time.sleep(0.1)
    
    # Calculate metrics
    test_duration = time.time() - start_time
    
    # Response time metrics
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
    test_results["model_performance"][model]["total_requests"] = total_requests
    test_results["model_performance"][model]["successful_requests"] = successful_requests
    test_results["model_performance"][model]["failed_requests"] = total_requests - successful_requests
    test_results["model_performance"][model]["success_rate"] = successful_requests / total_requests if total_requests > 0 else 0
    test_results["model_performance"][model]["error_rate"] = 1 - test_results["model_performance"][model]["success_rate"]
    test_results["model_performance"][model]["throughput"] = successful_requests / test_duration if test_duration > 0 else 0
    test_results["model_performance"][model]["avg_response_time"] = avg_response_time
    test_results["model_performance"][model]["p50_response_time"] = p50_response_time
    test_results["model_performance"][model]["p95_response_time"] = p95_response_time
    test_results["model_performance"][model]["p99_response_time"] = p99_response_time
    test_results["model_performance"][model]["min_response_time"] = min_response_time
    test_results["model_performance"][model]["max_response_time"] = max_response_time
    
    # Calculate resource usage
    if cpu_usage:
        peak_cpu = max(cpu_usage)
        avg_cpu = sum(cpu_usage) / len(cpu_usage)
        test_results["system_resources"]["peak_cpu"] = max(test_results["system_resources"]["peak_cpu"], peak_cpu)
        test_results["system_resources"]["avg_cpu"] = avg_cpu
    
    if memory_usage:
        peak_memory = max(memory_usage)
        avg_memory = sum(memory_usage) / len(memory_usage)
        test_results["system_resources"]["peak_memory"] = max(test_results["system_resources"]["peak_memory"], peak_memory)
        test_results["system_resources"]["avg_memory"] = avg_memory
    
    # Calculate recommendation quality metrics
    if all_recommendations:
        # Diversity - ratio of unique recommendations to total recommendations
        unique_recs = len(set(all_recommendations))
        total_recs = len(all_recommendations)
        diversity = unique_recs / total_recs if total_recs > 0 else 0
        test_results["model_performance"][model]["quality"]["diversity"] = diversity
        
        # Personalization - how different recommendations are between users
        user_similarity_scores = []
        user_ids = list(user_recommendations.keys())
        
        if len(user_ids) > 1:
            for i in range(min(100, len(user_ids))):  # Sample to avoid O(n²) complexity
                for j in range(i+1, min(100, len(user_ids))):
                    user1_recs = set(user_recommendations[user_ids[i]])
                    user2_recs = set(user_recommendations[user_ids[j]])
                    
                    if user1_recs and user2_recs:
                        # Jaccard similarity - intersection over union
                        similarity = len(user1_recs.intersection(user2_recs)) / len(user1_recs.union(user2_recs))
                        user_similarity_scores.append(similarity)
            
            # Personalization is 1 - average similarity (higher is better)
            if user_similarity_scores:
                personalization = 1 - np.mean(user_similarity_scores)
            else:
                personalization = 0
            
            test_results["model_performance"][model]["quality"]["personalization"] = personalization
    
    logger.info(f"Completed load test for model {model}")
    logger.info(f"Requests: {total_requests}, Successful: {successful_requests}, Success Rate: {test_results['model_performance'][model]['success_rate']:.2%}")
    logger.info(f"Avg Response Time: {avg_response_time:.3f}s, P95: {p95_response_time:.3f}s")
    
    return response_times, total_requests, successful_requests

# -------------------------------------------------------------------------
# REPORTING FUNCTIONS
# -------------------------------------------------------------------------

def generate_performance_charts():
    """Generate performance comparison charts for all tested models."""
    logger.info("Generating performance comparison charts...")
    
    try:
        plt.figure(figsize=(20, 15))
        
        # Response time comparison
        plt.subplot(2, 2, 1)
        model_names = []
        avg_times = []
        p95_times = []
        
        for model in models_to_test:
            if model in test_results["model_performance"]:
                model_names.append(model)
                avg_times.append(test_results["model_performance"][model]["avg_response_time"])
                p95_times.append(test_results["model_performance"][model]["p95_response_time"])
        
        x = np.arange(len(model_names))
        width = 0.35
        
        plt.bar(x - width/2, avg_times, width, label='Avg Response Time')
        plt.bar(x + width/2, p95_times, width, label='P95 Response Time')
        plt.xlabel('Model')
        plt.ylabel('Response Time (seconds)')
        plt.title('Response Time Comparison')
        plt.xticks(x, model_names)
        plt.legend()
        
        # Success rate comparison
        plt.subplot(2, 2, 2)
        success_rates = [test_results["model_performance"][model]["success_rate"] * 100 for model in model_names]
        throughputs = [test_results["model_performance"][model]["throughput"] for model in model_names]
        
        plt.bar(x - width/2, success_rates, width, label='Success Rate (%)')
        plt.bar(x + width/2, throughputs, width, label='Throughput (req/s)')
        plt.xlabel('Model')
        plt.ylabel('Metric Value')
        plt.title('Success Rate and Throughput Comparison')
        plt.xticks(x, model_names)
        plt.legend()
        
        # Quality metrics comparison
        plt.subplot(2, 2, 3)
        diversity = [test_results["model_performance"][model]["quality"]["diversity"] for model in model_names]
        personalization = [test_results["model_performance"][model]["quality"]["personalization"] for model in model_names]
        
        plt.bar(x - width/2, diversity, width, label='Diversity')
        plt.bar(x + width/2, personalization, width, label='Personalization')
        plt.xlabel('Model')
        plt.ylabel('Score (0-1)')
        plt.title('Recommendation Quality Comparison')
        plt.xticks(x, model_names)
        plt.ylim(0, 1)
        plt.legend()
        
        # Resource usage over time
        plt.subplot(2, 2, 4)
        plt.plot(range(len(cpu_usage)), cpu_usage, label='CPU Usage (%)')
        plt.plot(range(len(memory_usage)), memory_usage, label='Memory Usage (%)')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Resource Usage (%)')
        plt.title('Resource Usage During Test')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'model_performance_comparison.png'))
        logger.info(f"Performance charts saved to {os.path.join(args.output_dir, 'model_performance_comparison.png')}")
    
    except Exception as e:
        logger.error(f"Error generating performance charts: {str(e)}")
        logger.debug(traceback.format_exc())

def export_results_to_csv():
    """Export test results to CSV files."""
    try:
        # Create a DataFrame for model performance comparison
        model_comparison = {
            "model": [],
            "total_requests": [],
            "success_rate": [],
            "avg_response_time": [],
            "p95_response_time": [],
            "throughput": [],
            "diversity": [],
            "personalization": []
        }
        
        for model in models_to_test:
            if model in test_results["model_performance"]:
                perf = test_results["model_performance"][model]
                model_comparison["model"].append(model)
                model_comparison["total_requests"].append(perf["total_requests"])
                model_comparison["success_rate"].append(perf["success_rate"])
                model_comparison["avg_response_time"].append(perf["avg_response_time"])
                model_comparison["p95_response_time"].append(perf["p95_response_time"])
                model_comparison["throughput"].append(perf["throughput"])
                model_comparison["diversity"].append(perf["quality"]["diversity"])
                model_comparison["personalization"].append(perf["quality"]["personalization"])
        
        # Save model comparison
        df_comparison = pd.DataFrame(model_comparison)
        df_comparison.to_csv(os.path.join(args.output_dir, "model_comparison.csv"), index=False)
        
        # Save detailed results as JSON
        with open(os.path.join(args.output_dir, "detailed_results.json"), 'w') as f:
            json.dump(test_results, f, indent=2)
        
        logger.info(f"Test results exported to CSV and JSON files in {args.output_dir}")
    
    except Exception as e:
        logger.error(f"Error exporting results: {str(e)}")
        logger.debug(traceback.format_exc())

# -------------------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------------------

def main():
    """Main execution function."""
    global cpu_usage, memory_usage
    cpu_usage = []
    memory_usage = []
    
    logger.info("="*80)
    logger.info("STARTING PRODUCTION TEST FOR RECOMMENDATION ENGINE")
    logger.info(f"Target API: {args.api_url}")
    logger.info(f"Testing models: {models_to_test}")
    logger.info(f"Users: {args.users}, Concurrency: {args.concurrency}, Duration: {args.duration}s per model")
    logger.info("="*80)
    
    # Step 1: Check API health and available models
    api_healthy, available_models = check_api_health()
    if not api_healthy:
        logger.error("API is not healthy. Aborting test.")
        return
    
    # Step 2: Get user data - real or synthetic
    users = fetch_real_users()
    
    if not users or len(users) < min(100, args.users):
        if args.data_file and os.path.exists(args.data_file):
            users, products = load_real_data(args.data_file, max_users=args.users)
        
        if not users:
            # Fall back to synthetic data
            users = generate_users(args.users)
            products = generate_products(args.products)
    
    logger.info(f"Using {len(users)} users for testing")
    
    # Step 3: Test each model
    for model in models_to_test:
        if model not in available_models and model != "ensemble":
            logger.warning(f"Model {model} is not available. Skipping.")
            continue
        
        # Run cold start test if requested
        if args.test_new_users:
            cold_start_result = test_cold_start(model)
        
        # Run load test for the model
        response_times, total_requests, successful_requests = run_model_load_test(
            model, users, args.duration, args.concurrency
        )
    
    # Step 4: Generate charts and export results
    generate_performance_charts()
    export_results_to_csv()
    
    # Step 5: Final summary
    logger.info("="*80)
    logger.info("PRODUCTION TEST COMPLETE")
    logger.info("="*80)
    
    for model in models_to_test:
        if model in test_results["model_performance"]:
            perf = test_results["model_performance"][model]
            logger.info(f"Model: {model}")
            logger.info(f"  Requests: {perf['total_requests']}, Success Rate: {perf['success_rate']:.2%}")
            logger.info(f"  Avg Response Time: {perf['avg_response_time']:.3f}s, P95: {perf['p95_response_time']:.3f}s")
            logger.info(f"  Throughput: {perf['throughput']:.2f} req/s")
            logger.info(f"  Diversity: {perf['quality']['diversity']:.3f}, Personalization: {perf['quality']['personalization']:.3f}")
    
    logger.info(f"Peak CPU: {test_results['system_resources']['peak_cpu']:.1f}%, Peak Memory: {test_results['system_resources']['peak_memory']:.1f}%")
    logger.info(f"Results saved to {args.output_dir}")
    logger.info("="*80)

if __name__ == "__main__":
    main() 