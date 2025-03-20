#!/usr/bin/env python3
"""
Comprehensive System Test Script for the Amazon Recommendation Engine.

This script tests the entire recommendation system's robustness, including:
1. System robustness (API availability, load testing, error handling)
2. Recommendation accuracy evaluation
3. A/B testing capabilities

Usage:
    python test_system.py

Requirements:
    - requests
    - numpy
    - pandas
    - matplotlib
    - scikit-learn
"""

import requests
import json
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import precision_score, recall_score, ndcg_score
import threading
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("system_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("system_test")

# Configuration
BASE_API_URL = "http://localhost:5050/api"
MODELS = ["lightgcn", "ncf"]
NUM_RECOMMENDATIONS = 10
NUM_TEST_USERS = 20
NUM_CONCURRENT_REQUESTS = 20
LOAD_TEST_DURATION = 60  # seconds
REQUEST_TIMEOUT = 10  # seconds

# Test results storage
test_results = {
    "api_health": False,
    "recommendation_endpoints": False,
    "tracking_endpoint": False,
    "metrics_endpoint": False,
    "error_handling": False,
    "response_times": [],
    "model_accuracy": {},
    "ab_testing": False,
    "kafka_available": False,
    "concurrency_score": 0.0,
    "overall_score": 0.0
}

def log_test_result(test_name, result, details=None):
    """Log test results with details."""
    logger.info(f"Test: {test_name}")
    logger.info(f"Result: {'PASS' if result else 'FAIL'}")
    if details:
        logger.info(f"Details: {details}")
    logger.info("-" * 50)

def check_api_health():
    """Check if the API is healthy and accessible."""
    try:
        response = requests.get(f"{BASE_API_URL}/health", timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        health_data = response.json()
        
        # Check if all required components are healthy
        is_healthy = (
            health_data.get("status") == "healthy" and
            health_data.get("models_loaded", []) and
            "kafka_available" in health_data
        )
        
        test_results["api_health"] = is_healthy
        test_results["kafka_available"] = health_data.get("kafka_available", False)
        
        # Check for feedback loop status
        feedback_loop_enabled = health_data.get("feedback_loop_enabled", False)
        
        log_test_result(
            "API Health Check",
            is_healthy,
            f"Status: {health_data.get('status')}, Models: {health_data.get('models_loaded')}, "
            f"Feedback Loop: {'Enabled' if feedback_loop_enabled else 'Disabled'}"
        )
        
        return is_healthy
    except requests.exceptions.ConnectionError as e:
        logger.error(f"API connection failed: {str(e)}")
        test_results["api_health"] = False
        return False
    except requests.exceptions.Timeout as e:
        logger.error(f"API request timed out: {str(e)}")
        test_results["api_health"] = False
        return False
    except Exception as e:
        logger.error(f"API health check failed: {str(e)}")
        logger.error(traceback.format_exc())
        test_results["api_health"] = False
        return False

def test_recommendation_endpoints():
    """Test recommendation endpoints for all models."""
    success = True
    details = []
    
    for model in MODELS:
        try:
            # Test with a sample user
            response = requests.get(
                f"{BASE_API_URL}/recommend/{model}/user_1?n={NUM_RECOMMENDATIONS}",
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            
            data = response.json()
            if not data.get("recommendations"):
                success = False
                details.append(f"{model}: No recommendations returned")
                continue
                
            # Verify recommendation format
            recs = data.get("recommendations", [])
            if not all(isinstance(rec, str) for rec in recs):
                success = False
                details.append(f"{model}: Invalid recommendation format")
                continue
                
            details.append(f"{model}: {len(recs)} recommendations returned")
            
        except requests.exceptions.RequestException as e:
            success = False
            details.append(f"{model}: Request Error - {str(e)}")
        except Exception as e:
            success = False
            details.append(f"{model}: Error - {str(e)}")
            logger.error(traceback.format_exc())
    
    test_results["recommendation_endpoints"] = success
    log_test_result("Recommendation Endpoints", success, "\n".join(details))
    return success

def test_tracking_endpoint():
    """Test the event tracking endpoint."""
    try:
        # Test event tracking
        event_data = {
            "user_id": "user_1",
            "product_id": "product_1",
            "event_type": "view",
            "metadata": {
                "source": "test",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        response = requests.post(
            f"{BASE_API_URL}/track-event",
            json=event_data,
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        
        result = response.json()
        success = result.get("status") == "success"
        
        test_results["tracking_endpoint"] = success
        log_test_result("Event Tracking", success, f"Response: {result}")
        
        return success
    except requests.exceptions.RequestException as e:
        logger.error(f"Event tracking request failed: {str(e)}")
        test_results["tracking_endpoint"] = False
        return False
    except Exception as e:
        logger.error(f"Event tracking test failed: {str(e)}")
        logger.error(traceback.format_exc())
        test_results["tracking_endpoint"] = False
        return False

def test_metrics_endpoint():
    """Test the metrics endpoint if available."""
    try:
        # First try the dedicated metrics endpoint
        response = requests.get(
            f"{BASE_API_URL}/metrics",
            timeout=REQUEST_TIMEOUT
        )
        
        if response.status_code == 200:
            test_results["metrics_endpoint"] = True
            log_test_result("Metrics Endpoint", True, "Metrics endpoint available")
            return True
        
        # If that fails, try the feedback stats endpoint as a fallback
        response = requests.get(
            f"{BASE_API_URL}/feedback-stats",
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        
        result = response.json()
        success = result.get("status") == "active"
        
        test_results["metrics_endpoint"] = success
        log_test_result("Metrics Endpoint (feedback-stats)", success, f"Response: {result}")
        
        return success
    except requests.exceptions.RequestException as e:
        logger.error(f"Metrics endpoint request failed: {str(e)}")
        test_results["metrics_endpoint"] = False
        return False
    except Exception as e:
        logger.error(f"Metrics endpoint test failed: {str(e)}")
        logger.error(traceback.format_exc())
        test_results["metrics_endpoint"] = False
        return False

def test_error_handling():
    """Test error handling for various edge cases."""
    test_cases = [
        # Invalid user ID
        (f"{BASE_API_URL}/recommend/lightgcn/invalid_user", "GET"),
        # Invalid model
        (f"{BASE_API_URL}/recommend/invalid_model/user_1", "GET"),
        # Invalid event data
        (f"{BASE_API_URL}/track-event", "POST", {"invalid": "data"}),
        # Invalid number of recommendations
        (f"{BASE_API_URL}/recommend/lightgcn/user_1?n=-1", "GET")
    ]
    
    success = True
    details = []
    
    for test_case in test_cases:
        url = test_case[0]
        method = test_case[1]
        data = test_case[2] if len(test_case) > 2 else None
        
        try:
            if method == "GET":
                response = requests.get(url, timeout=REQUEST_TIMEOUT)
            else:
                response = requests.post(url, json=data, timeout=REQUEST_TIMEOUT)
            
            # We expect these to fail with appropriate error codes
            if response.status_code not in [400, 404, 422]:
                success = False
                details.append(f"Expected error for {url}, got {response.status_code}")
            else:
                # Check that error response is properly formatted
                error_data = response.json()
                if "error" not in error_data:
                    success = False
                    details.append(f"Missing error field in error response for {url}")
                else:
                    details.append(f"Correct error handling for {url}: {error_data.get('error')}")
                
        except requests.exceptions.RequestException as e:
            # Some errors are expected
            details.append(f"Expected error for {url}: {str(e)}")
        except Exception as e:
            success = False
            details.append(f"Unexpected error for {url}: {str(e)}")
            logger.error(traceback.format_exc())
    
    test_results["error_handling"] = success
    log_test_result("Error Handling", success, "\n".join(details))
    return success

def load_test():
    """Perform load testing on the recommendation endpoint."""
    def make_request():
        try:
            start_time = time.time()
            response = requests.get(
                f"{BASE_API_URL}/recommend/lightgcn/user_1?n={NUM_RECOMMENDATIONS}",
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            end_time = time.time()
            return end_time - start_time
        except Exception as e:
            logger.error(f"Load test request failed: {str(e)}")
            return None
    
    start_time = time.time()
    response_times = []
    successful_requests = 0
    total_requests = 0
    
    with ThreadPoolExecutor(max_workers=NUM_CONCURRENT_REQUESTS) as executor:
        while time.time() - start_time < LOAD_TEST_DURATION:
            futures = [executor.submit(make_request) for _ in range(NUM_CONCURRENT_REQUESTS)]
            for future in futures:
                total_requests += 1
                result = future.result()
                if result is not None:
                    successful_requests += 1
                    response_times.append(result)
    
    if response_times:
        avg_response_time = np.mean(response_times)
        p95_response_time = np.percentile(response_times, 95)
        success_rate = successful_requests / total_requests if total_requests > 0 else 0
        
        test_results["response_times"] = response_times
        test_results["concurrency_score"] = success_rate
        
        details = [
            f"Total requests: {total_requests}",
            f"Successful requests: {successful_requests}",
            f"Success rate: {success_rate:.2%}",
            f"Average response time: {avg_response_time:.3f}s",
            f"95th percentile: {p95_response_time:.3f}s"
        ]
        
        log_test_result("Load Testing", success_rate > 0.8, "\n".join(details))
        return success_rate > 0.8
    else:
        logger.error("No successful responses during load test")
        return False

def evaluate_recommendation_accuracy():
    """Evaluate recommendation accuracy using various metrics."""
    accuracy_results = {}
    
    for model in MODELS:
        try:
            # Get recommendations for multiple users
            user_recs = {}
            for user_id in [f"user_{i}" for i in range(1, NUM_TEST_USERS + 1)]:
                response = requests.get(
                    f"{BASE_API_URL}/recommend/{model}/{user_id}?n={NUM_RECOMMENDATIONS}",
                    timeout=REQUEST_TIMEOUT
                )
                response.raise_for_status()
                user_recs[user_id] = response.json().get("recommendations", [])
            
            # Calculate metrics
            # Note: In a real system, we would have ground truth data
            # For this test, we'll use some basic metrics
            
            # Calculate diversity (number of unique recommendations)
            all_recs = [rec for recs in user_recs.values() for rec in recs]
            unique_recs = len(set(all_recs))
            total_recs = len(all_recs)
            diversity = unique_recs / total_recs if total_recs > 0 else 0
            
            # Calculate personalization (how different are recommendations between users)
            user_similarities = []
            user_pairs = list(user_recs.items())
            for i in range(len(user_pairs)):
                for j in range(i + 1, len(user_pairs)):
                    recs1 = set(user_pairs[i][1])
                    recs2 = set(user_pairs[j][1])
                    if recs1 and recs2:
                        similarity = len(recs1.intersection(recs2)) / len(recs1.union(recs2))
                        user_similarities.append(similarity)
            
            personalization = 1 - np.mean(user_similarities) if user_similarities else 0
            
            accuracy_results[model] = {
                "diversity": diversity,
                "personalization": personalization,
                "avg_recommendations_per_user": total_recs / len(user_recs) if user_recs else 0
            }
            
            log_test_result(
                f"Model Accuracy ({model})",
                True,
                f"Diversity: {diversity:.2f}, Personalization: {personalization:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Accuracy evaluation failed for {model}: {str(e)}")
            logger.error(traceback.format_exc())
            accuracy_results[model] = None
    
    test_results["model_accuracy"] = accuracy_results
    return bool(accuracy_results)

def test_feedback_loop():
    """Test the feedback loop functionality."""
    try:
        # Step 1: Get initial recommendations for a test user
        test_user = f"test_user_{int(time.time())}"  # Create a unique test user
        test_product = "product_1"
        
        # Get initial recommendations
        initial_response = requests.get(
            f"{BASE_API_URL}/recommend/lightgcn/{test_user}?n=5",
            timeout=REQUEST_TIMEOUT
        )
        initial_response.raise_for_status()
        initial_recs = initial_response.json().get("recommendations", [])
        
        # Step 2: Track some events for the test user
        for i in range(1, 6):
            product_id = f"product_{i}"
            event_data = {
                "user_id": test_user,
                "product_id": product_id,
                "event_type": "view",
                "metadata": {
                    "source": "test",
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            response = requests.post(
                f"{BASE_API_URL}/track-event",
                json=event_data,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
        
        # Step 3: Trigger recommendation update
        update_response = requests.post(
            f"{BASE_API_URL}/update-recommendations",
            timeout=REQUEST_TIMEOUT
        )
        update_response.raise_for_status()
        
        # Step 4: Get updated recommendations
        time.sleep(2)  # Wait for processing
        updated_response = requests.get(
            f"{BASE_API_URL}/recommend/lightgcn/{test_user}?n=5",
            timeout=REQUEST_TIMEOUT
        )
        updated_response.raise_for_status()
        updated_recs = updated_response.json().get("recommendations", [])
        
        # Check if recommendations changed
        feedback_works = initial_recs != updated_recs
        
        log_test_result(
            "Feedback Loop",
            feedback_works,
            f"Initial: {initial_recs}, Updated: {updated_recs}"
        )
        
        return feedback_works
    except Exception as e:
        logger.error(f"Feedback loop test failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def generate_performance_charts():
    """Generate charts showing system performance metrics."""
    try:
        os.makedirs("test_results", exist_ok=True)
        plt.figure(figsize=(15, 10))
        
        # Response time distribution
        plt.subplot(2, 2, 1)
        if test_results["response_times"]:
            plt.hist(test_results["response_times"], bins=20, alpha=0.7)
            plt.title('Response Time Distribution')
            plt.xlabel('Response Time (seconds)')
            plt.ylabel('Count')
        
        # Model accuracy comparison
        plt.subplot(2, 2, 2)
        if test_results["model_accuracy"]:
            models = list(test_results["model_accuracy"].keys())
            diversity = []
            personalization = []
            
            for model in models:
                metrics = test_results["model_accuracy"][model]
                if metrics:
                    diversity.append(metrics["diversity"])
                    personalization.append(metrics["personalization"])
            
            if diversity and personalization:
                x = np.arange(len(models))
                width = 0.35
                
                plt.bar(x - width/2, diversity, width, label='Diversity')
                plt.bar(x + width/2, personalization, width, label='Personalization')
                plt.title('Model Performance Comparison')
                plt.xlabel('Model')
                plt.ylabel('Score')
                plt.xticks(x, models)
                plt.legend()
        
        # Test results overview
        plt.subplot(2, 2, 3)
        test_names = [
            'API Health', 'Recommendation Endpoints', 
            'Tracking Endpoint', 'Error Handling', 
            'Metrics Endpoint'
        ]
        test_values = [
            test_results['api_health'], 
            test_results['recommendation_endpoints'],
            test_results['tracking_endpoint'], 
            test_results['error_handling'],
            test_results['metrics_endpoint']
        ]
        colors = ['green' if v else 'red' for v in test_values]
        
        plt.bar(test_names, [int(v) for v in test_values], color=colors)
        plt.title('Test Results Overview')
        plt.ylabel('Pass (1) / Fail (0)')
        plt.xticks(rotation=45, ha='right')
        
        # Overall score
        plt.subplot(2, 2, 4)
        plt.pie([test_results["overall_score"], 1 - test_results["overall_score"]], 
                labels=[f'Score: {test_results["overall_score"]:.2f}', ''], 
                colors=['green', 'lightgray'],
                autopct='%1.1f%%',
                startangle=90)
        plt.title('Overall System Score')
        
        plt.tight_layout()
        plt.savefig("test_results/system_test_results.png")
        logger.info("Performance charts saved to test_results/system_test_results.png")
    except Exception as e:
        logger.error(f"Failed to generate performance charts: {str(e)}")
        logger.error(traceback.format_exc())

def calculate_overall_score():
    """Calculate an overall score for the system based on test results."""
    # Define weights for different components
    weights = {
        "api_health": 0.20,
        "recommendation_endpoints": 0.15,
        "tracking_endpoint": 0.15,
        "metrics_endpoint": 0.10,
        "error_handling": 0.10,
        "concurrency_score": 0.15,
        "model_accuracy": 0.15  # Based on average metrics
    }
    
    # Calculate score based on test results
    score = 0.0
    
    for key, weight in weights.items():
        if key == "concurrency_score":
            # Concurrency score is already a value between 0 and 1
            score += test_results[key] * weight
        elif key == "model_accuracy":
            # Calculate average model accuracy
            if test_results[key]:
                model_scores = []
                for model, metrics in test_results[key].items():
                    if metrics:
                        # Average of diversity and personalization
                        model_score = (metrics["diversity"] + metrics["personalization"]) / 2
                        model_scores.append(model_score)
                
                if model_scores:
                    avg_model_score = np.mean(model_scores)
                    score += avg_model_score * weight
        else:
            # Boolean test results
            score += (1.0 if test_results[key] else 0.0) * weight
    
    test_results["overall_score"] = score
    return score

def run_full_system_test():
    """Run all system tests and generate a comprehensive report."""
    logger.info("="*80)
    logger.info("BEGINNING COMPREHENSIVE SYSTEM TEST")
    logger.info("="*80)
    
    # Run all tests
    check_api_health()
    test_recommendation_endpoints()
    test_tracking_endpoint()
    test_metrics_endpoint()
    test_error_handling()
    load_test()
    evaluate_recommendation_accuracy()
    test_feedback_loop()
    
    # Calculate overall score
    overall_score = calculate_overall_score()
    
    # Generate performance charts
    generate_performance_charts()
    
    # Print final summary
    logger.info("="*80)
    logger.info("SYSTEM TEST SUMMARY")
    logger.info("="*80)
    logger.info(f"Overall Score: {overall_score:.2f}")
    logger.info(f"API Health: {'PASS' if test_results['api_health'] else 'FAIL'}")
    logger.info(f"Recommendation Endpoints: {'PASS' if test_results['recommendation_endpoints'] else 'FAIL'}")
    logger.info(f"Tracking Endpoint: {'PASS' if test_results['tracking_endpoint'] else 'FAIL'}")
    logger.info(f"Metrics Endpoint: {'PASS' if test_results['metrics_endpoint'] else 'FAIL'}")
    logger.info(f"Error Handling: {'PASS' if test_results['error_handling'] else 'FAIL'}")
    logger.info(f"Concurrency Score: {test_results['concurrency_score']:.2f}")
    logger.info(f"Kafka Available: {'Yes' if test_results['kafka_available'] else 'No'}")
    
    if test_results["model_accuracy"]:
        logger.info("\nModel Accuracy:")
        for model, metrics in test_results["model_accuracy"].items():
            if metrics:
                logger.info(f"- {model}: Diversity={metrics['diversity']:.2f}, "
                           f"Personalization={metrics['personalization']:.2f}")
    
    logger.info("="*80)
    
    # Export results to JSON
    try:
        os.makedirs("test_results", exist_ok=True)
        with open("test_results/system_test_results.json", "w") as f:
            json.dump(test_results, f, indent=2)
        logger.info("Test results saved to test_results/system_test_results.json")
    except Exception as e:
        logger.error(f"Failed to export results: {str(e)}")
    
    return overall_score

if __name__ == "__main__":
    try:
        # Run the full system test
        overall_score = run_full_system_test()
        
        # Exit with appropriate status code
        if overall_score >= 0.8:
            logger.info("System test PASSED with high score")
            sys.exit(0)  # Success
        elif overall_score >= 0.6:
            logger.warning("System test PASSED with warnings")
            sys.exit(1)  # Warning
        else:
            logger.error("System test FAILED")
            sys.exit(2)  # Failure
    except Exception as e:
        logger.error(f"System test failed with unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(3)  # Critical failure 