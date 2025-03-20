#!/usr/bin/env python
"""
Wrapper script to run the API with the correct Python path.
"""

import os
import sys
import logging
import threading
import time
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='api.log',
    filemode='a'
)
logger = logging.getLogger(__name__)

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set environment variables
os.environ["KAFKA_BOOTSTRAP_SERVERS"] = "localhost:29092"

# Import the shared feedback loop module
try:
    import feedback_loop
    logger.info("Successfully imported feedback_loop module")
    USE_FEEDBACK_LOOP = True
except ImportError as e:
    logger.warning(f"Unable to import feedback_loop module: {e}")
    USE_FEEDBACK_LOOP = False

# Create mock Kafka modules if needed
try:
    from src.kafka.producer import get_producer
    from src.kafka.stream_processor import RealtimeRecommender
    logger.info("Successfully imported Kafka modules")
except ImportError as e:
    logger.warning(f"Unable to import Kafka modules: {e}")
    logger.info("Creating mock implementations for Kafka modules")
    
    # Create mock modules
    import types
    import numpy as np
    
    # Mock producer module
    mock_producer_module = types.ModuleType('src.kafka.producer')
    sys.modules['src.kafka.producer'] = mock_producer_module
    
    class MockProducer:
        def __init__(self, *args, **kwargs):
            logger.info(f"Initialized MockProducer with {args}, {kwargs}")
            self.producer_type = kwargs.get('producer_type', 'unknown')
            
        def send_product_event(self, user_id, product_id, event_type, session_id, metadata=None):
            """Store events in global storage for real-time feedback processing"""
            logger.info(f"MOCK: Sending product event: user={user_id}, product={product_id}, event={event_type}")
            
            # Store the event in the feedback loop
            if USE_FEEDBACK_LOOP:
                feedback_loop.store_event(user_id, product_id, event_type, session_id, metadata)
            
            return True
            
        def send_metric(self, metric_name, metric_value, tags=None):
            logger.info(f"MOCK: Would send metric: {metric_name}={metric_value}, tags={tags}")
            return True
            
        def close(self):
            logger.info("MOCK: Producer closed")
    
    def mock_get_producer(*args, **kwargs):
        logger.info(f"MOCK: Creating producer with args: {args}, {kwargs}")
        return MockProducer(*args, **kwargs)
    
    mock_producer_module.get_producer = mock_get_producer
    
    # Mock stream processor module
    mock_stream_module = types.ModuleType('src.kafka.stream_processor')
    sys.modules['src.kafka.stream_processor'] = mock_stream_module
    
    class MockRealtimeRecommender:
        def __init__(self, update_interval_seconds=30, cache_size=1000, **kwargs):
            logger.info(f"MOCK: Initialized RealtimeRecommender with update_interval={update_interval_seconds}s, cache_size={cache_size}")
            self.consumer_type = "product_event"
            self.update_interval_seconds = update_interval_seconds
            self.cache_size = cache_size
            self.running = False
            self.last_update_time = datetime.now()
            self.update_thread = None
            
        def update_recommendations(self):
            """Update recommendations based on events"""
            if not USE_FEEDBACK_LOOP:
                logger.warning("Cannot update recommendations: feedback loop module not available")
                return
                
            # Get models from app module
            from src.api.app import models
            return feedback_loop.update_recommendations(models, self.cache_size)
            
        def update_loop(self):
            """Background thread to periodically update recommendations"""
            logger.info("Starting recommendation update loop")
            
            while self.running:
                try:
                    # Update recommendations periodically
                    now = datetime.now()
                    if (now - self.last_update_time).total_seconds() >= self.update_interval_seconds:
                        self.update_recommendations()
                        self.last_update_time = now
                    
                    # Sleep a bit to avoid CPU spinning
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"Error in recommendation update loop: {e}")
                    time.sleep(5)  # Sleep longer on error
            
            logger.info("Recommendation update loop stopped")
            
        def start(self, blocking=True):
            """Start the recommender in background or blocking mode"""
            if self.running:
                logger.warning("RealtimeRecommender is already running")
                return
            
            self.running = True
            logger.info(f"MOCK: Started RealtimeRecommender (blocking={blocking})")
            
            if blocking:
                # In blocking mode, run directly
                self.update_loop()
            else:
                # In non-blocking mode, start background thread
                self.update_thread = threading.Thread(target=self.update_loop)
                self.update_thread.daemon = True
                self.update_thread.start()
            
        def stop(self):
            """Stop the recommender"""
            logger.info("MOCK: Stopping RealtimeRecommender")
            self.running = False
            
            if self.update_thread and self.update_thread.is_alive():
                self.update_thread.join(timeout=10)
                if self.update_thread.is_alive():
                    logger.warning("Update thread did not terminate gracefully")
    
    mock_stream_module.RealtimeRecommender = MockRealtimeRecommender
    
    logger.info("Mock Kafka modules created successfully")

# Run the API
from src.api.app import app

if __name__ == "__main__":
    # Run the Flask application
    app.run(host='0.0.0.0', port=5050, debug=True) 