"""
Integration tests for the enhanced recommendation engine components.

This test suite validates:
1. Cold-start recommendation handling
2. Multi-level caching (memory and Redis)
3. Recommendation updates via feedback loop
"""

import os
import sys
import unittest
import time
import random
import json
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set up comprehensive mocking for all external dependencies

# Mock environment variables
os.environ['REDIS_HOST'] = 'localhost'
os.environ['REDIS_PORT'] = '6379'
os.environ['REDIS_PASSWORD'] = ''
os.environ['MEMORY_CACHE_SIZE'] = '1000'
os.environ['CACHE_TTL_SECONDS'] = '3600'

# Mock the database module to avoid the KeyError
sys.modules['src.data.database'] = MagicMock()
sys.modules['src.data.database'].get_connection_string = MagicMock(return_value='postgresql://user:pass@localhost:5432/testdb')
sys.modules['src.data.database'].get_engine = MagicMock()
sys.modules['src.data.database'].load_config = MagicMock(return_value={
    'host': 'localhost', 
    'port': 5432, 
    'database': 'testdb',
    'model': {
        'path': '/tmp/models',
        'default': 'lightgcn'
    },
    'cache': {
        'redis_host': 'localhost',
        'redis_port': 6379,
        'ttl': 3600
    }
})

# Mock Redis to avoid actual connection attempts
redis_mock = MagicMock()
sys.modules['redis'] = MagicMock()
sys.modules['redis'].Redis = MagicMock(return_value=redis_mock)
redis_mock.set = MagicMock(return_value=True)
redis_mock.get = MagicMock(return_value=None)
redis_mock.delete = MagicMock(return_value=True)
redis_mock.keys = MagicMock(return_value=[])

# Now import our modules
from src.models.cold_start import ColdStartHandler
from src.caching.recommendation_cache import RecommendationCache
from src.feedback.feedback_loop import *  # Import directly from the new location

class TestIntegration(unittest.TestCase):
    """Integration test suite for recommendation engine components."""
    
    def setUp(self):
        """Set up test environment."""
        # Clear feedback loop global storages
        feedback_loop.GLOBAL_USER_EVENTS.clear()
        feedback_loop.GLOBAL_PRODUCT_VIEWS.clear()
        feedback_loop.GLOBAL_USER_PRODUCTS.clear()
        feedback_loop.GLOBAL_RELATED_PRODUCTS.clear()
        
        # Initialize test data
        self.user_id = f"test_user_{random.randint(1000, 9999)}"
        self.products = [f"prod_{i}" for i in range(1, 21)]
        
        # Create mock models
        self.mock_models = {
            'lightgcn': {
                'product_map': {prod_id: i for i, prod_id in enumerate(self.products)},
                'item_factors': [[random.random() for _ in range(10)] for _ in range(len(self.products))]
            }
        }
        
        # Initialize the cache
        self.cache = RecommendationCache()
        
        # Clear the cache at start - invalidate cache for all users
        self.cache.invalidate_cache()
        
    def tearDown(self):
        """Clean up after tests."""
        # Clear cache
        self.cache.invalidate_cache()
        
    def generate_mock_event(self, user_id, product_id, event_type="view"):
        """Generate a mock user event."""
        return {
            "user_id": user_id,
            "product_id": product_id,
            "event_type": event_type,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "session_id": f"session_{random.randint(1000, 9999)}"
        }
    
    def test_cold_start_handler(self):
        """Test the cold start handler."""
        cold_start = ColdStartHandler()
        
        # Generate product views for popularity
        product_views = {prod: random.randint(1, 100) for prod in self.products}
        
        # Build popularity cache
        cold_start.build_popularity_cache(product_views)
        
        # Get recommendations for a new user
        recs = cold_start.get_recommendations(count=5)
        
        # Validate
        self.assertEqual(len(recs), 5, "Should return exactly 5 recommendations")
        
        # All recommendations should be from our product list
        for rec in recs:
            self.assertIn(rec, self.products, "Cold start recommendations should be from product list")
        
        # Test with user info
        user_info = {"preferred_category": "electronics"}
        recs_with_info = cold_start.get_recommendations(user_info=user_info, count=5)
        self.assertEqual(len(recs_with_info), 5, "Should return exactly 5 recommendations with user info")
    
    def test_recommendation_cache(self):
        """Test the multi-level caching system."""
        cache = RecommendationCache()
        
        user_id = self.user_id
        model_name = "lightgcn"
        
        # Create sample recommendations
        recommendations = [
            {"product_id": f"prod_{i}", "score": 1.0 - (i * 0.01)} 
            for i in range(10)
        ]
        
        # Cache recommendations
        context = {"source": "test", "is_cold_start": False}
        cache.cache_recommendations(
            user_id=user_id,
            model_name=model_name,
            recommendations=recommendations,
            context=context,
            limit=10
        )
        
        # Test retrieving from cache
        cached_recs = cache.get_recommendations(
            user_id=user_id,
            model_name=model_name,
            count=5
        )
        
        self.assertIsNotNone(cached_recs, "Should retrieve recommendations from cache")
        self.assertEqual(len(cached_recs), 5, "Should return exactly 5 recommendations")
        
        # Test the memory vs Redis caching
        # First make sure the cache has our recommendations
        self.assertIsNotNone(cached_recs, "Cache should contain our recommendations")
        
        # Invalidate user-specific cache but keep other entries
        cache.invalidate_cache(user_id=user_id + "_other")
        
        # Should still be able to retrieve from Redis
        cached_recs = cache.get_recommendations(
            user_id=user_id,
            model_name=model_name,
            count=5
        )
        
        self.assertIsNotNone(cached_recs, "Should retrieve recommendations from Redis after memory cache cleared")
        self.assertEqual(len(cached_recs), 5, "Should return exactly 5 recommendations")
    
    def test_feedback_loop_integration(self):
        """Test feedback loop with cold start and caching."""
        # Register a new user
        feedback_loop.handle_new_user(self.user_id, models=self.mock_models)
        
        # Check if cold start recommendations were generated and cached
        cached_recs = feedback_loop.recommendation_cache.get_recommendations(
            user_id=self.user_id,
            model_name="lightgcn",
            count=10
        )
        
        self.assertIsNotNone(cached_recs, "Cold start recommendations should be cached")
        
        # Generate some user events
        for i in range(5):
            event = self.generate_mock_event(self.user_id, self.products[i])
            feedback_loop.store_event(event)
        
        # Check if the user's viewed products are tracked
        self.assertEqual(len(feedback_loop.GLOBAL_USER_PRODUCTS[self.user_id]), 5, 
                         "Should have 5 products in user's viewed products")
        
        # Update recommendations based on feedback
        updated_users = feedback_loop.update_recommendations(self.mock_models, cache_size=10)
        
        self.assertEqual(updated_users, 1, "Should update recommendations for 1 user")
        
        # Verify the updated recommendations are different from cold start
        updated_recs = feedback_loop.recommendation_cache.get_recommendations(
            user_id=self.user_id,
            model_name="lightgcn",
            count=10
        )
        
        self.assertIsNotNone(updated_recs, "Updated recommendations should be cached")
        
        # Test cache invalidation
        feedback_loop.recommendation_cache.invalidate_user_cache(self.user_id)
        
        # After invalidation, getting from cache should return None
        invalidated_recs = feedback_loop.recommendation_cache.get_recommendations(
            user_id=self.user_id,
            model_name="lightgcn", 
            count=10,
            skip_fallback=True
        )
        
        self.assertIsNone(invalidated_recs, "Cache should be invalidated")

if __name__ == '__main__':
    unittest.main()
