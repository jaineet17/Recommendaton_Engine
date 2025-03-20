import os
import json
import logging
import redis
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RecommendationCache:
    """Cache system for recommendation results using Redis"""
    
    def __init__(self):
        # Initialize Redis connection
        self.redis_host = os.environ.get('REDIS_HOST', 'redis')
        self.redis_port = int(os.environ.get('REDIS_PORT', 6379))
        self.redis_db = int(os.environ.get('REDIS_CACHE_DB', 1))  # Use DB 1 for caching
        self.ttl = int(os.environ.get('CACHE_TTL_SECONDS', 3600))  # Default 1 hour TTL
        
        try:
            self.redis = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                decode_responses=True
            )
            logger.info(f"Connected to Redis cache at {self.redis_host}:{self.redis_port}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis = None
    
    def get_cache_key(self, user_id, model_name, context=None, limit=10):
        """Generate a cache key for recommendation results"""
        context_str = "".join(f":{k}:{v}" for k, v in sorted(context.items())) if context else ""
        return f"rec:{user_id}:{model_name}{context_str}:{limit}"
    
    def get_recommendations(self, user_id, model_name, context=None, limit=10):
        """Get cached recommendations if available"""
        if not self.redis:
            return None
        
        try:
            cache_key = self.get_cache_key(user_id, model_name, context, limit)
            cached_data = self.redis.get(cache_key)
            
            if cached_data:
                return json.loads(cached_data)
            return None
        except Exception as e:
            logger.error(f"Error retrieving from cache: {e}")
            return None
    
    def cache_recommendations(self, user_id, model_name, recommendations, context=None, limit=10, ttl=None):
        """Cache recommendation results"""
        if not self.redis:
            return False
        
        try:
            cache_key = self.get_cache_key(user_id, model_name, context, limit)
            cache_data = {
                'user_id': user_id,
                'model_name': model_name,
                'recommendations': recommendations,
                'context': context,
                'timestamp': datetime.now().isoformat(),
                'limit': limit
            }
            
            # Cache with expiration time
            expiration = ttl if ttl is not None else self.ttl
            self.redis.setex(
                cache_key,
                expiration,
                json.dumps(cache_data)
            )
            
            logger.debug(f"Cached recommendations for {user_id} using {model_name}")
            return True
        except Exception as e:
            logger.error(f"Error caching recommendations: {e}")
            return False
    
    def invalidate_cache(self, user_id=None, model_name=None):
        """Invalidate cache for a user or model"""
        if not self.redis:
            return False
        
        try:
            if user_id and model_name:
                # Delete specific user-model combinations
                pattern = f"rec:{user_id}:{model_name}:*"
            elif user_id:
                # Delete all for specific user
                pattern = f"rec:{user_id}:*"
            elif model_name:
                # Delete all for specific model
                pattern = f"rec:*:{model_name}:*"
            else:
                # Delete all recommendations
                pattern = "rec:*"
            
            # Find all matching keys
            keys = self.redis.keys(pattern)
            if keys:
                self.redis.delete(*keys)
                logger.info(f"Invalidated {len(keys)} cache entries with pattern {pattern}")
            
            return True
        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")
            return False
    
    def get_cache_stats(self):
        """Get cache statistics"""
        if not self.redis:
            return {}
        
        try:
            total_keys = len(self.redis.keys("rec:*"))
            memory_used = self.redis.info().get('used_memory_human', 'unknown')
            
            # Count keys by model
            model_counts = {}
            models = set()
            for key in self.redis.keys("rec:*"):
                parts = key.split(':')
                if len(parts) >= 3:
                    model = parts[2]
                    models.add(model)
            
            for model in models:
                model_keys = len(self.redis.keys(f"rec:*:{model}:*"))
                model_counts[model] = model_keys
            
            return {
                'total_cached_entries': total_keys,
                'memory_used': memory_used,
                'model_distribution': model_counts
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {} 