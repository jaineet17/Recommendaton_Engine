#!/usr/bin/env python3
"""
A/B Testing Evaluation Script for the Amazon Recommendation Engine.

This script focuses on testing the A/B testing capabilities of the recommendation system:
1. Checks if the system supports experiment groups
2. Compares different models to see if they provide meaningfully different recommendations
3. Evaluates the potential for running controlled experiments

Usage:
    python test_ab_testing.py

Requirements:
    - requests
    - numpy
    - pandas
    - matplotlib
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ab_testing.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ab_testing")

# Configuration
BASE_API_URL = "http://localhost:5050/api"
MODELS = ["lightgcn", "ncf"]  # The models to compare
NUM_RECOMMENDATIONS = 10
NUM_TEST_USERS = 20

def get_test_users():
    """Get a list of users to test with."""
    try:
        response = requests.get(f"{BASE_API_URL}/users", timeout=5)
        response.raise_for_status()
        users = response.json()
        
        if not users:
            logger.error("No users found in the system")
            return []
            
        return users[:NUM_TEST_USERS] if len(users) >= NUM_TEST_USERS else users
    except Exception as e:
        logger.error(f"Failed to get users: {str(e)}")
        return []

def check_experiment_header_support(user_id):
    """Check if the API accepts experiment group headers."""
    experiment_groups = ["control", "test_group_a", "test_group_b"]
    results = {}
    
    for group in experiment_groups:
        try:
            response = requests.get(
                f"{BASE_API_URL}/recommend/lightgcn/{user_id}?n=5",
                headers={"X-Experiment-Group": group},
                timeout=5
            )
            
            results[group] = {
                "status_code": response.status_code,
                "accepts_header": response.status_code == 200,
                "recommendations": response.json().get("recommendations", []) if response.status_code == 200 else []
            }
            
            logger.info(f"Testing experiment group '{group}': Status code {response.status_code}")
        except Exception as e:
            logger.error(f"Error testing experiment group '{group}': {str(e)}")
            results[group] = {
                "status_code": -1,
                "accepts_header": False,
                "recommendations": []
            }
    
    # Check if all groups were accepted
    all_accepted = all(results[group]["accepts_header"] for group in experiment_groups)
    logger.info(f"API accepts experiment headers: {all_accepted}")
    
    # Check if the recommendations differ between groups
    # This is a basic test - in a real system, there should be observable differences
    recommendations_differ = False
    if all_accepted:
        control_recs = set(results["control"]["recommendations"])
        test_a_recs = set(results["test_group_a"]["recommendations"])
        test_b_recs = set(results["test_group_b"]["recommendations"])
        
        # Calculate Jaccard similarity between recommendation sets
        jaccard_control_a = len(control_recs.intersection(test_a_recs)) / len(control_recs.union(test_a_recs)) if control_recs.union(test_a_recs) else 1.0
        jaccard_control_b = len(control_recs.intersection(test_b_recs)) / len(control_recs.union(test_b_recs)) if control_recs.union(test_b_recs) else 1.0
        jaccard_a_b = len(test_a_recs.intersection(test_b_recs)) / len(test_a_recs.union(test_b_recs)) if test_a_recs.union(test_b_recs) else 1.0
        
        avg_similarity = (jaccard_control_a + jaccard_control_b + jaccard_a_b) / 3
        recommendations_differ = avg_similarity < 0.8
        
        logger.info(f"Recommendations differ between experiment groups: {recommendations_differ} (Avg similarity: {avg_similarity:.2f})")
    
    return {
        "accepts_headers": all_accepted,
        "recommendations_differ": recommendations_differ,
        "header_details": results
    }

def compare_model_recommendations(users):
    """Compare recommendations between different models to see if they differ."""
    model_comparisons = {}
    
    # Compare each pair of models
    if len(MODELS) < 2:
        logger.warning("Need at least 2 models to compare")
        return {}
        
    for i in range(len(MODELS)):
        for j in range(i+1, len(MODELS)):
            model_a = MODELS[i]
            model_b = MODELS[j]
            
            comparison_key = f"{model_a}_vs_{model_b}"
            model_comparisons[comparison_key] = {
                "similarity_scores": [],
                "avg_similarity": 0.0,
                "user_scores": {}
            }
            
            logger.info(f"Comparing models: {model_a} vs {model_b}")
            
            for user_id in users:
                try:
                    # Get recommendations from both models
                    response_a = requests.get(
                        f"{BASE_API_URL}/recommend/{model_a}/{user_id}?n={NUM_RECOMMENDATIONS}", 
                        timeout=5
                    )
                    response_a.raise_for_status()
                    recs_a = response_a.json().get("recommendations", [])
                    
                    response_b = requests.get(
                        f"{BASE_API_URL}/recommend/{model_b}/{user_id}?n={NUM_RECOMMENDATIONS}", 
                        timeout=5
                    )
                    response_b.raise_for_status()
                    recs_b = response_b.json().get("recommendations", [])
                    
                    # Calculate similarity
                    set_a = set(recs_a)
                    set_b = set(recs_b)
                    
                    if not set_a or not set_b:
                        logger.warning(f"Empty recommendation sets for user {user_id}")
                        continue
                        
                    intersection = len(set_a.intersection(set_b))
                    union = len(set_a.union(set_b))
                    similarity = intersection / union if union > 0 else 1.0
                    
                    model_comparisons[comparison_key]["similarity_scores"].append(similarity)
                    model_comparisons[comparison_key]["user_scores"][user_id] = similarity
                    
                    logger.debug(f"User {user_id}: {model_a} vs {model_b} similarity: {similarity:.2f}")
                    
                except Exception as e:
                    logger.error(f"Error comparing models for user {user_id}: {str(e)}")
            
            # Calculate average similarity
            if model_comparisons[comparison_key]["similarity_scores"]:
                avg_similarity = np.mean(model_comparisons[comparison_key]["similarity_scores"])
                model_comparisons[comparison_key]["avg_similarity"] = avg_similarity
                logger.info(f"Average similarity between {model_a} and {model_b}: {avg_similarity:.2f}")
            else:
                logger.warning(f"No valid comparisons between {model_a} and {model_b}")
    
    return model_comparisons

def evaluate_recommendation_distribution(users, model="lightgcn"):
    """
    Evaluate how recommendations are distributed across products.
    This helps assess if A/B testing would be effective.
    """
    all_recommendations = []
    
    for user_id in users:
        try:
            response = requests.get(
                f"{BASE_API_URL}/recommend/{model}/{user_id}?n={NUM_RECOMMENDATIONS}", 
                timeout=5
            )
            response.raise_for_status()
            recommendations = response.json().get("recommendations", [])
            all_recommendations.extend(recommendations)
        except Exception as e:
            logger.error(f"Error getting recommendations for user {user_id}: {str(e)}")
    
    # Count frequency of each product
    product_counts = {}
    for product in all_recommendations:
        product_counts[product] = product_counts.get(product, 0) + 1
    
    # Calculate metrics
    if not product_counts:
        logger.warning("No recommendations collected")
        return {}
        
    unique_products = len(product_counts)
    total_recommendations = len(all_recommendations)
    
    # Calculate diversity metrics
    coverage = unique_products / total_recommendations if total_recommendations > 0 else 0
    
    # Gini coefficient for diversity
    counts = sorted(product_counts.values())
    n = len(counts)
    
    if n == 0:
        gini = 0
    else:
        cumul = np.cumsum(counts)
        gini = 1 - 2 * np.sum((cumul - counts) / cumul[-1]) / n if cumul[-1] > 0 else 0
    
    # Top-N concentration (how much of the recommendation volume is in the top 10 products)
    top_products = sorted(product_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    top10_concentration = sum(count for _, count in top_products) / total_recommendations if total_recommendations > 0 else 0
    
    logger.info(f"Model {model} recommendation distribution:")
    logger.info(f"- Unique products recommended: {unique_products}")
    logger.info(f"- Coverage: {coverage:.2f}")
    logger.info(f"- Gini coefficient: {gini:.2f}")
    logger.info(f"- Top-10 concentration: {top10_concentration:.2f}")
    
    return {
        "unique_products": unique_products,
        "total_recommendations": total_recommendations,
        "coverage": coverage,
        "gini_coefficient": gini,
        "top10_concentration": top10_concentration,
        "product_counts": product_counts
    }

def run_ab_testing_evaluation():
    """Run a comprehensive evaluation of A/B testing capabilities."""
    logger.info("="*80)
    logger.info("BEGINNING A/B TESTING CAPABILITY EVALUATION")
    logger.info("="*80)
    
    # Get users to test with
    users = get_test_users()
    
    if not users:
        logger.error("No users found. Aborting tests.")
        return {
            "supports_ab_testing": False,
            "explanation": "No users found in the system"
        }
    
    # Check if the API supports experiment headers
    logger.info("Testing experiment header support...")
    user_id = random.choice(users)
    header_results = check_experiment_header_support(user_id)
    
    # Compare models to see if they give different recommendations
    logger.info("Comparing model recommendations...")
    model_comparison_results = compare_model_recommendations(users)
    
    # Evaluate recommendation distribution
    logger.info("Evaluating recommendation distribution...")
    distribution_results = {}
    for model in MODELS:
        distribution_results[model] = evaluate_recommendation_distribution(users, model)
    
    # Determine if the system can support A/B testing
    models_differ = all(comp["avg_similarity"] < 0.8 for comp in model_comparison_results.values())
    
    supports_ab_testing = header_results["accepts_headers"] and models_differ
    
    # Analyze diversity metrics to assess A/B testing potential
    has_sufficient_diversity = all(dist["gini_coefficient"] < 0.7 for dist in distribution_results.values())
    
    # Generate recommendations for improvement
    recommendations = []
    if not header_results["accepts_headers"]:
        recommendations.append("Implement explicit support for experiment group headers")
    
    if not models_differ:
        recommendations.append("Ensure models provide significantly different recommendations for valid A/B testing")
    
    if not has_sufficient_diversity:
        recommendations.append("Improve recommendation diversity to make A/B testing more effective")
    
    # Final results
    results = {
        "supports_ab_testing": supports_ab_testing,
        "experiment_header_support": header_results,
        "model_comparisons": model_comparison_results,
        "recommendation_distribution": distribution_results,
        "has_sufficient_diversity": has_sufficient_diversity,
        "recommendations": recommendations
    }
    
    # Print summary
    logger.info("="*80)
    logger.info("A/B TESTING EVALUATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Supports A/B testing: {supports_ab_testing}")
    
    if model_comparison_results:
        logger.info("Model comparison results:")
        for comparison, data in model_comparison_results.items():
            logger.info(f"- {comparison}: Average similarity {data.get('avg_similarity', 'N/A'):.2f}")
    
    logger.info("Recommendation distribution summary:")
    for model, data in distribution_results.items():
        logger.info(f"- {model}: Diversity (Gini: {data.get('gini_coefficient', 'N/A'):.2f}, " + 
                    f"Coverage: {data.get('coverage', 'N/A'):.2f})")
    
    if recommendations:
        logger.info("Recommendations for improvement:")
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"{i}. {rec}")
    
    logger.info("="*80)
    
    return results

def generate_ab_testing_charts(results):
    """Generate charts visualizing A/B testing capabilities."""
    try:
        plt.figure(figsize=(15, 10))
        
        # Model similarity chart
        plt.subplot(2, 2, 1)
        comparisons = []
        similarities = []
        
        for comparison, data in results["model_comparisons"].items():
            comparisons.append(comparison)
            similarities.append(data.get("avg_similarity", 0))
        
        plt.bar(comparisons, similarities, color='blue')
        plt.title('Model Recommendation Similarity')
        plt.xlabel('Model Comparison')
        plt.ylabel('Jaccard Similarity')
        plt.ylim(0, 1)
        plt.axhline(y=0.8, color='r', linestyle='--', label='Similarity Threshold')
        plt.legend()
        
        # Recommendation distribution
        plt.subplot(2, 2, 2)
        models = []
        gini_values = []
        coverage_values = []
        
        for model, data in results["recommendation_distribution"].items():
            models.append(model)
            gini_values.append(data.get("gini_coefficient", 0))
            coverage_values.append(data.get("coverage", 0))
        
        x = np.arange(len(models))
        width = 0.35
        
        plt.bar(x - width/2, gini_values, width, label='Gini Coefficient')
        plt.bar(x + width/2, coverage_values, width, label='Coverage')
        plt.axhline(y=0.7, color='r', linestyle='--', label='Gini Threshold')
        plt.title('Recommendation Diversity Metrics')
        plt.xlabel('Model')
        plt.ylabel('Value')
        plt.xticks(x, models)
        plt.legend()
        
        # Product popularity distribution (for the first model)
        plt.subplot(2, 2, 3)
        if results["recommendation_distribution"] and list(results["recommendation_distribution"].keys()):
            first_model = list(results["recommendation_distribution"].keys())[0]
            product_counts = results["recommendation_distribution"][first_model].get("product_counts", {})
            
            if product_counts:
                sorted_counts = sorted(product_counts.items(), key=lambda x: x[1], reverse=True)[:20]
                products, counts = zip(*sorted_counts)
                
                plt.bar(range(len(products)), counts, color='green')
                plt.title(f'Top 20 Recommended Products ({first_model})')
                plt.xlabel('Product Rank')
                plt.ylabel('Recommendation Count')
                plt.xticks([])  # Hide x labels as they would be product IDs
        
        # User-level similarity distribution
        plt.subplot(2, 2, 4)
        if results["model_comparisons"] and list(results["model_comparisons"].keys()):
            first_comparison = list(results["model_comparisons"].keys())[0]
            similarity_scores = results["model_comparisons"][first_comparison].get("similarity_scores", [])
            
            if similarity_scores:
                plt.hist(similarity_scores, bins=10, alpha=0.7, color='purple')
                plt.title(f'User-level Similarity Distribution ({first_comparison})')
                plt.xlabel('Jaccard Similarity')
                plt.ylabel('Number of Users')
                plt.xlim(0, 1)
        
        plt.tight_layout()
        plt.savefig('ab_testing_analysis.png')
        logger.info("A/B testing analysis charts generated: ab_testing_analysis.png")
    except Exception as e:
        logger.error(f"Error generating A/B testing charts: {str(e)}")

if __name__ == "__main__":
    try:
        # Run the A/B testing evaluation
        results = run_ab_testing_evaluation()
        
        # Generate charts
        if results["model_comparisons"] and results["recommendation_distribution"]:
            generate_ab_testing_charts(results)
        
        # Exit with success if A/B testing is supported
        if results["supports_ab_testing"]:
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        logger.error(f"A/B testing evaluation failed with unexpected error: {str(e)}")
        sys.exit(1) 