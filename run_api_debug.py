#!/usr/bin/env python
"""
Simplified wrapper script to run the API with debug logging and error handling.
"""

import os
import sys
import logging
import traceback
import threading
import time
from datetime import datetime

# Set up logging to both console and file
logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG level for more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_debug.log', mode='w'),  # Overwrite with each run
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Log startup information
logger.info("="*80)
logger.info("STARTING API IN DEBUG MODE")
logger.info("Python version: %s", sys.version)
logger.info("Current directory: %s", os.getcwd())
logger.info("="*80)

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
logger.info("Python path: %s", sys.path)

# Set environment variables
os.environ["KAFKA_BOOTSTRAP_SERVERS"] = "localhost:29092"
logger.info("Set KAFKA_BOOTSTRAP_SERVERS=localhost:29092")

# Clean exit function for later use
def exit_handler():
    logger.info("API process shutting down")
    sys.exit(0)

# Import the shared feedback loop module with error handling
try:
    import feedback_loop
    logger.info("Successfully imported feedback_loop module")
    USE_FEEDBACK_LOOP = True
except ImportError as e:
    logger.warning(f"Unable to import feedback_loop module: {e}")
    logger.warning(traceback.format_exc())
    USE_FEEDBACK_LOOP = False

# Create mock Kafka modules if needed
try:
    logger.info("Attempting to import Kafka modules")
    from src.kafka.producer import get_producer
    from src.kafka.stream_processor import RealtimeRecommender
    logger.info("Successfully imported Kafka modules")
    USE_MOCK_KAFKA = False
except ImportError as e:
    logger.warning(f"Unable to import Kafka modules: {e}")
    logger.warning(traceback.format_exc())
    logger.info("Creating mock implementations for Kafka modules")
    USE_MOCK_KAFKA = True
    
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
            logger.debug(f"MOCK: Sending product event: user={user_id}, product={product_id}, event={event_type}")
            
            # Store the event in the feedback loop
            if USE_FEEDBACK_LOOP:
                event_data = {
                    'user_id': user_id,
                    'product_id': product_id,
                    'event_type': event_type,
                    'session_id': session_id,
                    'metadata': metadata or {}
                }
                feedback_loop.store_event(event_data)
            
            return True
            
        def send_metric(self, metric_name, metric_value, tags=None):
            logger.debug(f"MOCK: Would send metric: {metric_name}={metric_value}, tags={tags}")
            return True
            
        def close(self):
            logger.debug("MOCK: Producer closed")
    
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
            # Add the proper consumer type
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
            try:
                from src.api.app import models
                return feedback_loop.update_recommendations(models, self.cache_size)
            except Exception as e:
                logger.error(f"Error in update_recommendations: {e}")
                logger.error(traceback.format_exc())
                return 0
            
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
                    logger.error(traceback.format_exc())
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

# Attempt to import and run the API with comprehensive error handling
try:
    logger.info("Importing Flask app")
    from src.api.app import app
    logger.info("Flask app imported successfully")
    
    if __name__ == "__main__":
        logger.info("Starting Flask application on port 5050")
        
        try:
            # Run the Flask application without Flask's internal reloader
            # which can cause problems with mock modules
            app.run(host='0.0.0.0', port=5050, debug=True, use_reloader=False)
        except Exception as e:
            logger.critical(f"Failed to start Flask application: {e}")
            logger.critical(traceback.format_exc())
            exit_handler()
            
except Exception as e:
    logger.critical(f"Failed to import Flask app: {e}")
    logger.critical(traceback.format_exc())
    exit_handler() 