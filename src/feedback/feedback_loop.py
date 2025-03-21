"""
Shared module for feedback loop storage in the recommendation system.

This module provides storage for user events and product relationships,
as well as integration with the cold-start handler and recommendation cache.
"""

from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Set, Optional, Any, Tuple

import numpy as np

from src.utils.logging_config import get_logger
# Import the ModelFactory class rather than specific functions
from src.models.model_factory import ModelFactory
from src.models.cold_start import ColdStartHandler
from src.caching.recommendation_cache import RecommendationCache

# Set up logging
logger = get_logger('feedback_loop')

# Global storage for feedback events
GLOBAL_USER_EVENTS = defaultdict(list)
GLOBAL_PRODUCT_VIEWS = defaultdict(int)
GLOBAL_USER_PRODUCTS = defaultdict(set)
GLOBAL_RELATED_PRODUCTS = {}

# Initialize caching and cold-start components
recommendation_cache = RecommendationCache()
cold_start_handler = ColdStartHandler()

def get_events_storage():
    """
    Get access to the global events storage.
    
    Returns:
        dict: Dictionary containing all event storage objects
    """
    return {
        'user_events': GLOBAL_USER_EVENTS,
        'product_views': GLOBAL_PRODUCT_VIEWS,
        'user_products': GLOBAL_USER_PRODUCTS,
        'related_products': GLOBAL_RELATED_PRODUCTS
    }

def store_event(event):
    """Store an event in the feedback loop database"""
    user_id = event.get('user_id')
    product_id = event.get('product_id')
    event_type = event.get('event_type')
    timestamp = event.get('timestamp', datetime.now().isoformat())
    session_id = event.get('session_id', 'default')
    
    # Skip if missing required fields
    if not all([user_id, product_id, event_type]):
        logger.warning("Missing required fields in event")
        return False
    
    # Initialize user if new
    if user_id not in GLOBAL_USER_EVENTS:
        logger.info(f"New user: {user_id}")
        GLOBAL_USER_EVENTS[user_id] = []
        GLOBAL_USER_PRODUCTS[user_id] = set()
    
    # Add product to user's viewed products
    GLOBAL_USER_PRODUCTS[user_id].add(product_id)
    
    # Track event
    event_data = {
        'user_id': user_id,
        'product_id': product_id,
        'event_type': event_type,
        'timestamp': timestamp,
        'session_id': session_id
    }
    GLOBAL_USER_EVENTS[user_id].append(event_data)
    
    # Update product views
    if event_type == 'view':
        GLOBAL_PRODUCT_VIEWS[product_id] = GLOBAL_PRODUCT_VIEWS.get(product_id, 0) + 1
    
    # Update product relationships for this user
    # The key insight: Build relationships between ALL pairs of products this user has viewed
    viewed_products = list(GLOBAL_USER_PRODUCTS[user_id])
    
    # Create relationships between all pairs of viewed products
    for i in range(len(viewed_products)):
        for j in range(i+1, len(viewed_products)):
            prod1 = viewed_products[i]
            prod2 = viewed_products[j]
            
            # Initialize if these are new products
            if prod1 not in GLOBAL_RELATED_PRODUCTS:
                GLOBAL_RELATED_PRODUCTS[prod1] = {}
            if prod2 not in GLOBAL_RELATED_PRODUCTS:
                GLOBAL_RELATED_PRODUCTS[prod2] = {}
            
            # Create bidirectional relationship
            # Products viewed in the same session have stronger relationship
            weight = 2.0 if session_id != 'default' else 1.0
            
            # Update relationship from prod1 -> prod2
            curr_score = GLOBAL_RELATED_PRODUCTS[prod1].get(prod2, 0)
            GLOBAL_RELATED_PRODUCTS[prod1][prod2] = curr_score + weight
            
            # Update relationship from prod2 -> prod1
            curr_score = GLOBAL_RELATED_PRODUCTS[prod2].get(prod1, 0)
            GLOBAL_RELATED_PRODUCTS[prod2][prod1] = curr_score + weight
            
            logger.debug(f"Updated relationship: {prod1} <-> {prod2} (weight: {weight})")
    
    # Invalidate cache for this user to force fresh recommendations
    try:
        # Use the recommendation_cache module directly to invalidate cache
        recommendation_cache.invalidate_user_cache(user_id)
        logger.debug(f"Invalidated recommendation cache for user {user_id}")
    except Exception as e:
        logger.error(f"Error invalidating cache: {e}")
    
    logger.debug(f"Stored event: {event_type} for user {user_id} on product {product_id}")
    return True

def get_cold_start_recommendations(user_info=None, model_name=None, top_n=10):
    """
    Get recommendations for a new user with no history using the cold-start handler.
    
    Args:
        user_info: User demographic and preference information (optional)
        model_name: The name of the model (optional)
        top_n: Number of recommendations to return
        
    Returns:
        List of dictionaries with product_id and score
    """
    # First, ensure the cold-start handler has the latest popularity data
    if not hasattr(cold_start_handler, 'popularity_cache') or not cold_start_handler.popularity_cache:
        # Build popularity cache if it doesn't exist yet
        item_counts = GLOBAL_PRODUCT_VIEWS.copy()
        if item_counts:
            cold_start_handler.build_popularity_cache(item_counts)
            logger.info(f"Built cold-start popularity cache with {len(item_counts)} items")
    
    # Use the cold-start handler to get recommendations
    recommended_items = cold_start_handler.get_recommendations(user_info=user_info, count=top_n)
    
    # Format recommendations with scores
    recommendations = []
    for i, item_id in enumerate(recommended_items):
        # Assign decreasing scores starting from 1.0
        score = 1.0 - (i * 0.01)
        recommendations.append({'product_id': item_id, 'score': score})
    
    logger.info(f"Generated {len(recommendations)} cold-start recommendations")
    return recommendations

def handle_new_user(user_id, user_info=None, models=None):
    """
    Process a new user, initializing storage and pre-generating cold-start recommendations.
    
    Args:
        user_id: The user ID to create
        user_info: Optional user profile information (demographics, preferences)
        models: Dictionary of loaded models (optional)
        
    Returns:
        True if user was created, False otherwise
    """
    logger.info(f"Handling new user: {user_id}")
    
    # Check if the user already exists
    if user_id in GLOBAL_USER_EVENTS and len(GLOBAL_USER_EVENTS[user_id]) > 0:
        logger.info(f"User {user_id} already exists with {len(GLOBAL_USER_EVENTS[user_id])} events")
        return True
    
    # Initialize the user with empty events
    GLOBAL_USER_EVENTS[user_id] = []
    GLOBAL_USER_PRODUCTS[user_id] = set()
    
    # Pre-generate and cache cold-start recommendations for available models
    if models:
        for model_name in models.keys():
            try:
                # Generate cold-start recommendations
                recommendations = get_cold_start_recommendations(user_info, model_name, top_n=20)
                
                # Cache these recommendations for quick access
                context = {'is_cold_start': True}
                if user_info:
                    context.update(user_info)
                
                recommendation_cache.cache_recommendations(
                    user_id=user_id,
                    model_name=model_name,
                    recommendations=recommendations,
                    context=context,
                    limit=20,
                    user_activity='new',
                    ttl=1800  # 30-minute TTL for new user recommendations
                )
                
                logger.debug(f"Pre-cached cold-start recommendations for new user {user_id} with model {model_name}")
            except Exception as e:
                logger.error(f"Error pre-caching recommendations for new user {user_id}: {e}")
    
    logger.info(f"Created new user {user_id} in events storage")
    return True

def update_recommendations(models, cache_size=100, rebuild_cold_start=False):
    """
    Update recommendations based on user interactions.
    
    Args:
        models: Dictionary of loaded models
        cache_size: Maximum number of recommendations to generate
        
    Returns:
        int: Number of users with updated recommendations
    """
    import numpy as np
    logger.info("Updating real-time recommendations based on feedback")
    logger.info(f"User products entries: {len(GLOBAL_USER_PRODUCTS)}")
    logger.info(f"Total events: {sum(len(events) for events in GLOBAL_USER_EVENTS.values())}")
    logger.info(f"Related products entries: {len(GLOBAL_RELATED_PRODUCTS)}")
    
    if not models:
        logger.warning("No models available for recommendation updates")
        return 0
    
    # Get popular products from events (recent views)
    sorted_products = sorted(GLOBAL_PRODUCT_VIEWS.items(), key=lambda x: x[1], reverse=True)
    top_popular_from_events = [p[0] for p in sorted_products[:cache_size]] if sorted_products else []
    logger.info(f"Top popular products from events: {top_popular_from_events[:5] if top_popular_from_events else 'None'}")
    
    # Always get model-based popular products
    model_based_popular = []
    if 'lightgcn' in models:
        # Get top popular products from the model's item factors
        model = models['lightgcn']
        item_factors = model['item_factors']
        popularity_scores = {}
        
        # Calculate average popularity score for all products
        for idx, prod_id in enumerate(model['product_map'].keys()):
            popularity_score = np.sum(item_factors[idx])
            popularity_scores[prod_id] = popularity_score
        
        # Sort by popularity score
        sorted_model_products = sorted(popularity_scores.items(), key=lambda x: x[1], reverse=True)
        model_based_popular = [p[0] for p in sorted_model_products[:cache_size]]
        logger.info(f"Model-based popular products: {model_based_popular[:5] if model_based_popular else 'None'}")
    
    # Combine both popular product lists, with events taking precedence
    top_popular = []
    if top_popular_from_events:
        top_popular.extend(top_popular_from_events)
    
    # Add model-based popular products that aren't already in the list
    if model_based_popular:
        for prod_id in model_based_popular:
            if prod_id not in top_popular:
                top_popular.append(prod_id)
                if len(top_popular) >= cache_size * 2:  # Get twice as many for variety
                    break
    
    logger.info(f"Combined popular products: {top_popular[:5] if top_popular else 'None'}")
    
    # Calculate personalized recommendations for each user
    updated_users = 0
    for user_id, viewed_products in GLOBAL_USER_PRODUCTS.items():
        logger.info(f"Processing user {user_id} with {len(viewed_products)} viewed products")
        
        if not viewed_products:
            logger.debug(f"User {user_id} has no viewed products, skipping")
            continue
        
        # Skip if too many products (likely a bot)
        if len(viewed_products) > 1000:
            logger.debug(f"User {user_id} has too many viewed products ({len(viewed_products)}), skipping")
            continue
        
        # Get related products for all products this user has viewed
        candidate_products = defaultdict(float)
        for product_id in viewed_products:
            # Check if the product has any related products
            if product_id in GLOBAL_RELATED_PRODUCTS:
                related_dict = GLOBAL_RELATED_PRODUCTS[product_id]
                logger.debug(f"Product {product_id} has {len(related_dict)} related products")
                
                for related_id, score in related_dict.items():
                    if related_id not in viewed_products:  # Don't recommend already viewed
                        candidate_products[related_id] += score
        
        # Check if we have any candidate products
        top_n = []
        if not candidate_products:
            logger.info(f"No candidate products for user {user_id}, will use popular products")
        else:
            # Sort candidates by score
            recommendations = sorted(candidate_products.items(), key=lambda x: x[1], reverse=True)
            top_n = [p[0] for p in recommendations[:cache_size]]
            logger.info(f"Found {len(top_n)} candidate products for user {user_id}")
        
        # If not enough recommendations (or none), fill with top popular
        current_len = len(top_n)
        num_to_add = min(cache_size - current_len, len(top_popular))
        if current_len < cache_size and top_popular:
            added = 0
            for product_id in top_popular:
                if product_id not in viewed_products and product_id not in top_n:
                    top_n.append(product_id)
                    added += 1
                    if len(top_n) >= cache_size:
                        break
            logger.info(f"Added {added} popular products to recommendations")
        
        # Store recommendations in cache - ALWAYS do this if the user has viewed products
        # even if we're just using popular products
        if len(viewed_products) > 0:
            # If we still have no recommendations, just use all popular products 
            # (including model-based ones)
            if not top_n:
                logger.info(f"No recommendations yet, trying harder with model-based popular products")
                # First try model-based popular products
                for product_id in model_based_popular:
                    if product_id not in viewed_products:
                        top_n.append(product_id)
                        if len(top_n) >= cache_size:
                            break
                
                # If still no recommendations, generate some random ones from the model
                if not top_n and 'lightgcn' in models:
                    logger.info(f"Still no recommendations, generating random ones from model")
                    model = models['lightgcn']
                    all_products = list(model['product_map'].keys())
                    import random
                    random.shuffle(all_products)
                    for product_id in all_products:
                        if product_id not in viewed_products:
                            top_n.append(product_id)
                            if len(top_n) >= cache_size:
                                break
                
                logger.info(f"Final attempt yielded {len(top_n)} recommendations")
            
            # Make sure we have at least some recommendations
            if top_n:
                updated_users += 1
                for model_name in models.keys():
                    # Format recommendations with scores
                    formatted_recs = []
                    for i, pid in enumerate(top_n[:cache_size]):
                        score = 1.0 - (i * 0.01)  # Simple score falloff
                        formatted_recs.append({'product_id': pid, 'score': score})
                    
                    # Cache using our recommendation cache
                    context = {
                        'source': 'feedback_loop',
                        'viewed_products_count': len(viewed_products)
                    }
                    
                    # Determine user activity level for better TTL settings
                    activity_level = 'high' if len(viewed_products) > 20 else 'medium' if len(viewed_products) > 5 else 'low'
                    
                    # Save to the cache system
                    recommendation_cache.cache_recommendations(
                        user_id=user_id,
                        model_name=model_name,
                        recommendations=formatted_recs,
                        context=context,
                        limit=cache_size,
                        user_activity=activity_level
                    )
                
                logger.info(f"Updated recommendations for user {user_id}: {top_n[:5] if len(top_n) >= 5 else top_n}")
            else:
                logger.warning(f"No recommendations could be generated for user {user_id}")
                # Use cold start recommendations as fallback
                try:
                    fallback_recs = get_cold_start_recommendations(top_n=cache_size)
                    context = {
                        'source': 'feedback_loop_fallback',
                        'viewed_products_count': len(viewed_products),
                        'is_cold_start': True
                    }
                    
                    # Cache fallback recommendations
                    for model_name in models.keys():
                        recommendation_cache.cache_recommendations(
                            user_id=user_id,
                            model_name=model_name,
                            recommendations=fallback_recs,
                            context=context,
                            limit=cache_size,
                            ttl=1800  # 30 minute TTL for fallbacks
                        )
                    logger.info(f"Stored fallback cold-start recommendations for user {user_id}")
                except Exception as e:
                    logger.error(f"Error generating fallback recommendations: {e}")
    
    # Update cold-start data with new information if needed
    if rebuild_cold_start or not hasattr(cold_start_handler, 'popularity_cache'):
        try:
            # Build item counts from views
            item_counts = GLOBAL_PRODUCT_VIEWS.copy()
            cold_start_handler.build_popularity_cache(item_counts)
            
            logger.info(f"Refreshed cold-start popularity data with {len(item_counts)} items")
        except Exception as e:
            logger.error(f"Error updating cold-start data: {e}")
    
    # Get cache statistics
    try:
        cache_stats = recommendation_cache.get_cache_stats()
        logger.info(f"Cache statistics: {cache_stats}")
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
    
    logger.info(f"Updated recommendations for {updated_users} users based on feedback")
    return updated_users
