"""
Cold-start strategies for recommendation engines.

This module provides implementations of various strategies for handling
cold-start problems in recommendation systems.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

# Import common utilities
from src.utils.logging_config import get_logger

logger = get_logger("models.cold_start")

class ColdStartHandler:
    """
    Handler for cold-start problems in recommendation systems.
    
    This class implements multiple strategies for generating recommendations
    when no data is available for a user or item.
    """
    
    def __init__(self, 
                fallback_strategies: List[str] = None,
                popularity_threshold: int = 10,
                max_items_to_cache: int = 1000):
        """
        Initialize cold-start handler.
        
        Args:
            fallback_strategies: List of strategies to try in order
                                 ["popularity", "category", "diversity", "random"]
            popularity_threshold: Minimum popularity count to consider an item popular
            max_items_to_cache: Maximum number of items to cache for each strategy
        """
        self.fallback_strategies = fallback_strategies or ["popularity", "category", "diversity", "random"]
        self.popularity_threshold = popularity_threshold
        self.max_items_to_cache = max_items_to_cache
        
        # Initialize caches for different strategies
        self.popularity_cache = {}
        self.category_cache = defaultdict(list)
        self.diversity_cache = {}
        self.item_metadata = {}
        
        logger.info(f"Initialized cold-start handler with strategies: {self.fallback_strategies}")
    
    def build_popularity_cache(self, 
                              item_counts: Dict[str, int], 
                              rating_df: Optional[pd.DataFrame] = None) -> None:
        """
        Build cache of popular items for cold-start recommendations.
        
        Args:
            item_counts: Dictionary mapping item IDs to their popularity counts
            rating_df: Optional dataframe with user-item interactions
        """
        # Filter items with enough popularity
        popular_items = {
            item_id: count for item_id, count in item_counts.items() 
            if count >= self.popularity_threshold
        }
        
        # Sort by popularity
        sorted_items = sorted(popular_items.items(), key=lambda x: x[1], reverse=True)
        top_items = [item_id for item_id, _ in sorted_items[:self.max_items_to_cache]]
        
        # Cache overall popularity
        self.popularity_cache["overall"] = top_items
        
        # If rating data is available, calculate user demographic-based popularity
        if rating_df is not None and "user_id" in rating_df.columns:
            # Sample user for demographic analysis
            sample_size = min(100000, len(rating_df))
            sample_df = rating_df.sample(n=sample_size) if len(rating_df) > sample_size else rating_df
            
            # Extract user features if available
            if "user_age" in sample_df.columns:
                self._build_demographic_popularity(sample_df, "user_age")
            
            if "user_gender" in sample_df.columns:
                self._build_demographic_popularity(sample_df, "user_gender")
                
            if "user_location" in sample_df.columns:
                self._build_demographic_popularity(sample_df, "user_location")
        
        logger.info(f"Built popularity cache with {len(top_items)} items")
    
    def _build_demographic_popularity(self, df: pd.DataFrame, demographic_col: str) -> None:
        """
        Build popularity cache segmented by demographic attribute.
        
        Args:
            df: DataFrame with user-item interactions
            demographic_col: Column name for demographic attribute
        """
        # Group by demographic and item, count occurrences
        demo_counts = df.groupby([demographic_col, "product_id"]).size().reset_index(name="count")
        
        # For each demographic value, get top items
        for demo_value, group_df in demo_counts.groupby(demographic_col):
            # Sort by popularity within this demographic
            sorted_items = group_df.sort_values("count", ascending=False)
            top_items = sorted_items["product_id"].tolist()[:self.max_items_to_cache]
            
            # Cache for this demographic
            cache_key = f"{demographic_col}:{demo_value}"
            self.popularity_cache[cache_key] = top_items
            
            logger.debug(f"Built popularity cache for {cache_key} with {len(top_items)} items")
    
    def build_category_cache(self, item_metadata: Dict[str, Dict[str, Any]]) -> None:
        """
        Build cache of items by category for diversity-based cold-start.
        
        Args:
            item_metadata: Dictionary mapping item IDs to their metadata
        """
        # Store item metadata
        self.item_metadata = item_metadata
        
        # Group items by category
        categories = defaultdict(list)
        for item_id, metadata in item_metadata.items():
            if "category" in metadata:
                categories[metadata["category"]].append(item_id)
        
        # For each category, sort items by popularity if available
        for category, items in categories.items():
            if hasattr(self, "popularity_cache") and self.popularity_cache:
                # Use global popularity to sort within category
                sorted_items = sorted(
                    items,
                    key=lambda x: self.popularity_cache.get("overall", {}).get(x, 0),
                    reverse=True
                )
                self.category_cache[category] = sorted_items[:self.max_items_to_cache]
            else:
                # Without popularity data, just store the category items
                self.category_cache[category] = items[:self.max_items_to_cache]
        
        logger.info(f"Built category cache with {len(self.category_cache)} categories")
    
    def build_diversity_cache(self, item_factors: np.ndarray, item_map: Dict[str, int]) -> None:
        """
        Build cache of diverse items based on latent factors.
        
        Args:
            item_factors: Item latent factors matrix
            item_map: Mapping from item IDs to indices in factors matrix
        """
        # Create reverse mapping from indices to item IDs
        reverse_map = {idx: item_id for item_id, idx in item_map.items()}
        
        # Compute item clusters using factor space
        try:
            from sklearn.cluster import KMeans
            
            # Determine optimal number of clusters (limited by max_items_to_cache)
            n_clusters = min(20, self.max_items_to_cache // 5)
            
            # Cluster items
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(item_factors)
            
            # For each cluster, find the central items
            for cluster_id in range(n_clusters):
                # Get indices of items in this cluster
                cluster_indices = np.where(clusters == cluster_id)[0]
                
                # Get cluster center
                cluster_center = kmeans.cluster_centers_[cluster_id]
                
                # Calculate distance to center for each item
                distances = np.linalg.norm(item_factors[cluster_indices] - cluster_center, axis=1)
                
                # Sort by closeness to center
                sorted_indices = cluster_indices[np.argsort(distances)]
                
                # Get item IDs for these indices
                central_items = [reverse_map[idx] for idx in sorted_indices 
                                if idx in reverse_map][:self.max_items_to_cache // n_clusters]
                
                # Store in diversity cache
                self.diversity_cache[f"cluster_{cluster_id}"] = central_items
                
            logger.info(f"Built diversity cache with {n_clusters} clusters")
        except ImportError:
            logger.warning("scikit-learn not available, skipping diversity cache")
        except Exception as e:
            logger.error(f"Error building diversity cache: {e}")
    
    def get_recommendations(self, 
                           user_info: Optional[Dict[str, Any]] = None,
                           count: int = 10) -> List[str]:
        """
        Get cold-start recommendations for a new user.
        
        Args:
            user_info: Optional user information for personalized cold-start
            count: Number of recommendations to return
            
        Returns:
            List of recommended item IDs
        """
        recommendations = []
        
        # Try each strategy in order until we have enough recommendations
        for strategy in self.fallback_strategies:
            if len(recommendations) >= count:
                break
                
            strategy_items = self._get_strategy_recommendations(strategy, user_info, count)
            
            # Add new items from this strategy
            for item in strategy_items:
                if item not in recommendations:
                    recommendations.append(item)
                    if len(recommendations) >= count:
                        break
        
        # If we still don't have enough, use random recommendation as last resort
        if len(recommendations) < count:
            random_items = self._get_random_recommendations(count - len(recommendations))
            for item in random_items:
                if item not in recommendations:
                    recommendations.append(item)
        
        return recommendations[:count]
    
    def _get_strategy_recommendations(self, 
                                     strategy: str, 
                                     user_info: Optional[Dict[str, Any]], 
                                     count: int) -> List[str]:
        """
        Get recommendations using a specific strategy.
        
        Args:
            strategy: Strategy name ("popularity", "category", "diversity", "random")
            user_info: Optional user information for personalized cold-start
            count: Number of recommendations to return
            
        Returns:
            List of recommended item IDs
        """
        if strategy == "popularity":
            return self._get_popularity_recommendations(user_info, count)
        elif strategy == "category":
            return self._get_category_recommendations(user_info, count)
        elif strategy == "diversity":
            return self._get_diversity_recommendations(count)
        elif strategy == "random":
            return self._get_random_recommendations(count)
        else:
            logger.warning(f"Unknown strategy: {strategy}")
            return []
    
    def _get_popularity_recommendations(self, 
                                       user_info: Optional[Dict[str, Any]], 
                                       count: int) -> List[str]:
        """
        Get recommendations based on popularity.
        
        Args:
            user_info: Optional user demographic information
            count: Number of recommendations to return
            
        Returns:
            List of popular item IDs
        """
        # If user info is available, try demographic-based popularity
        if user_info:
            # Try each demographic feature
            for feature in ["user_age", "user_gender", "user_location"]:
                if feature in user_info:
                    cache_key = f"{feature}:{user_info[feature]}"
                    if cache_key in self.popularity_cache:
                        return self.popularity_cache[cache_key][:count]
        
        # Fallback to overall popularity
        if "overall" in self.popularity_cache:
            return self.popularity_cache["overall"][:count]
        
        # No popularity data available
        return []
    
    def _get_category_recommendations(self, 
                                     user_info: Optional[Dict[str, Any]], 
                                     count: int) -> List[str]:
        """
        Get recommendations based on category preferences.
        
        Args:
            user_info: Optional user category preferences
            count: Number of recommendations to return
            
        Returns:
            List of category-based item IDs
        """
        recommendations = []
        
        # If user has category preferences, use those
        if user_info and "preferred_categories" in user_info:
            preferred_categories = user_info["preferred_categories"]
            items_per_category = max(1, count // len(preferred_categories))
            
            # Get items from each preferred category
            for category in preferred_categories:
                if category in self.category_cache:
                    category_items = self.category_cache[category][:items_per_category]
                    recommendations.extend(category_items)
        
        # If we don't have user preferences or not enough items, use diverse categories
        if len(recommendations) < count:
            # Get a diverse set of categories
            diverse_categories = list(self.category_cache.keys())[:10]
            items_needed = count - len(recommendations)
            items_per_category = max(1, items_needed // len(diverse_categories))
            
            # Get a few items from each category
            for category in diverse_categories:
                if len(recommendations) >= count:
                    break
                    
                category_items = self.category_cache[category][:items_per_category]
                for item in category_items:
                    if item not in recommendations:
                        recommendations.append(item)
                        if len(recommendations) >= count:
                            break
        
        return recommendations[:count]
    
    def _get_diversity_recommendations(self, count: int) -> List[str]:
        """
        Get recommendations optimized for diversity.
        
        Args:
            count: Number of recommendations to return
            
        Returns:
            List of diverse item IDs
        """
        recommendations = []
        
        if not self.diversity_cache:
            return []
            
        # Get items from each cluster
        clusters = list(self.diversity_cache.keys())
        items_per_cluster = max(1, count // len(clusters))
        
        for cluster in clusters:
            if len(recommendations) >= count:
                break
                
            cluster_items = self.diversity_cache[cluster][:items_per_cluster]
            for item in cluster_items:
                if item not in recommendations:
                    recommendations.append(item)
                    if len(recommendations) >= count:
                        break
        
        return recommendations[:count]
    
    def _get_random_recommendations(self, count: int) -> List[str]:
        """
        Get random recommendations (last resort).
        
        Args:
            count: Number of recommendations to return
            
        Returns:
            List of random item IDs
        """
        # Create pool of available items
        item_pool = []
        
        # First try to use items from metadata
        if self.item_metadata:
            item_pool = list(self.item_metadata.keys())
        
        # If we don't have metadata, try using items from popularity cache
        elif "overall" in self.popularity_cache:
            item_pool = self.popularity_cache["overall"]
        
        # If we don't have popularity data, try using items from any category
        elif self.category_cache:
            for category_items in self.category_cache.values():
                item_pool.extend(category_items)
            item_pool = list(set(item_pool))  # Remove duplicates
        
        # If we don't have any items, generate fake IDs (last resort)
        if not item_pool:
            return [f"item_{i}" for i in range(count)]
        
        # Sample random items
        np.random.seed(42)  # For reproducibility
        indices = np.random.choice(len(item_pool), min(count, len(item_pool)), replace=False)
        return [item_pool[i] for i in indices]
