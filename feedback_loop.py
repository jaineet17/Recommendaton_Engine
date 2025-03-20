"""
Shared module for feedback loop storage in the Amazon Recommendation System.

This module provides global storage for events and recommendations that can be 
shared between the API and other components.
"""

import logging
from collections import defaultdict, deque
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

# Global storage for feedback events
GLOBAL_USER_EVENTS = defaultdict(list)
GLOBAL_PRODUCT_VIEWS = defaultdict(int)
GLOBAL_USER_PRODUCTS = defaultdict(set)
GLOBAL_RELATED_PRODUCTS = {}
GLOBAL_RECOMMENDATION_CACHE = {}

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
        'related_products': GLOBAL_RELATED_PRODUCTS,
        'recommendation_cache': GLOBAL_RECOMMENDATION_CACHE
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
    
    # Clear cache for this user to force fresh recommendations
    for key in list(GLOBAL_RECOMMENDATION_CACHE.keys()):
        if user_id in key:
            del GLOBAL_RECOMMENDATION_CACHE[key]
    
    logger.debug(f"Stored event: {event_type} for user {user_id} on product {product_id}")
    return True

def get_cold_start_recommendations(popular_products, model_name=None, top_n=10):
    """
    Get recommendations for a new user with no history.
    
    Args:
        popular_products: List of popular products to recommend
        model_name: The name of the model (optional)
        top_n: Number of recommendations to return
        
    Returns:
        List of top-N recommended product IDs
    """
    # Use popular products if available
    if popular_products:
        return popular_products[:top_n]
    
    # Fallback: get most viewed products from feedback loop
    if GLOBAL_PRODUCT_VIEWS:
        sorted_products = sorted(GLOBAL_PRODUCT_VIEWS.items(), key=lambda x: x[1], reverse=True)
        top_viewed = [p[0] for p in sorted_products[:top_n]]
        return top_viewed
    
    # Last resort: generate fake product IDs
    return [f"product_{i}" for i in range(1, top_n+1)]

def handle_new_user(user_id, models=None):
    """
    Create a placeholder for a new user that doesn't exist in the model.
    
    Args:
        user_id: The user ID to create
        models: Dictionary of loaded models (optional)
        
    Returns:
        True if user was created, False otherwise
    """
    logger.info(f"Handling new user: {user_id}")
    
    # Check if the user already exists
    if user_id in GLOBAL_USER_EVENTS:
        logger.info(f"User {user_id} already exists in events storage")
        return True
    
    # Initialize the user with empty events
    GLOBAL_USER_EVENTS[user_id] = []
    GLOBAL_USER_PRODUCTS[user_id] = set()
    
    # Set a flag in recommendation cache to show this is a new user
    if models:
        for model_name in models.keys():
            cache_key = f"{model_name}_{user_id}_new_user"
            GLOBAL_RECOMMENDATION_CACHE[cache_key] = True
    
    logger.info(f"Created new user {user_id} in events storage")
    return True

def update_recommendations(models, cache_size=100):
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
                    # Cache for top 10
                    cache_key = f"{model_name}_{user_id}_10"
                    GLOBAL_RECOMMENDATION_CACHE[cache_key] = top_n[:min(10, len(top_n))]
                    
                    # Also cache other common sizes
                    cache_key = f"{model_name}_{user_id}_5"
                    GLOBAL_RECOMMENDATION_CACHE[cache_key] = top_n[:min(5, len(top_n))]
                
                logger.info(f"Updated recommendations for user {user_id}: {top_n[:5] if len(top_n) >= 5 else top_n}")
            else:
                logger.warning(f"No recommendations could be generated for user {user_id}")
    
    logger.info(f"Updated recommendations for {updated_users} users based on feedback")
    return updated_users