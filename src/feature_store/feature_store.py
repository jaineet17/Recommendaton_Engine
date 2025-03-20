import os
import logging
import redis
import json
import time
import numpy as np
from datetime import datetime, timedelta
import threading
import queue
from src.kafka.consumer import BaseConsumer
from src.kafka.producer import get_producer

logger = logging.getLogger(__name__)

class FeatureStore:
    """
    Feature store using Redis for fast feature retrieval and real-time updates
    """
    
    def __init__(self, prefix="features"):
        """Initialize the feature store"""
        # Connect to Redis
        self.redis_host = os.environ.get("REDIS_HOST", "redis")
        self.redis_port = int(os.environ.get("REDIS_PORT", 6379))
        self.redis_db = int(os.environ.get("REDIS_FEATURE_DB", 2))  # Use DB 2 for features
        self.prefix = prefix
        
        try:
            self.redis = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                decode_responses=True
            )
            logger.info(f"Connected to Redis feature store at {self.redis_host}:{self.redis_port}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis = None
    
    def _get_key(self, feature_type, entity_id, feature_name=None):
        """Generate a Redis key for a feature"""
        if feature_name:
            return f"{self.prefix}:{feature_type}:{entity_id}:{feature_name}"
        else:
            return f"{self.prefix}:{feature_type}:{entity_id}"
    
    def get_feature(self, feature_type, entity_id, feature_name):
        """Get a specific feature value"""
        if not self.redis:
            return None
        
        try:
            key = self._get_key(feature_type, entity_id, feature_name)
            value = self.redis.get(key)
            
            if value is None:
                return None
            
            # Try to parse as JSON, fall back to string
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                try:
                    return float(value)
                except ValueError:
                    return value
        except Exception as e:
            logger.error(f"Error getting feature {feature_name}: {e}")
            return None
    
    def get_features(self, feature_type, entity_id, feature_names=None):
        """Get multiple features for an entity"""
        if not self.redis:
            return {}
        
        try:
            if feature_names:
                # Get specific features
                features = {}
                pipeline = self.redis.pipeline()
                
                for name in feature_names:
                    key = self._get_key(feature_type, entity_id, name)
                    pipeline.get(key)
                
                values = pipeline.execute()
                
                for i, name in enumerate(feature_names):
                    if values[i] is not None:
                        try:
                            features[name] = json.loads(values[i])
                        except json.JSONDecodeError:
                            try:
                                features[name] = float(values[i])
                            except ValueError:
                                features[name] = values[i]
                
                return features
            else:
                # Get all features using pattern
                features = {}
                pattern = self._get_key(feature_type, entity_id, "*")
                
                for key in self.redis.scan_iter(match=pattern):
                    name = key.split(":")[-1]
                    value = self.redis.get(key)
                    
                    if value is not None:
                        try:
                            features[name] = json.loads(value)
                        except json.JSONDecodeError:
                            try:
                                features[name] = float(value)
                            except ValueError:
                                features[name] = value
                
                return features
        except Exception as e:
            logger.error(f"Error getting features for {entity_id}: {e}")
            return {}
    
    def set_feature(self, feature_type, entity_id, feature_name, value, ttl=None):
        """Set a feature value"""
        if not self.redis:
            return False
        
        try:
            key = self._get_key(feature_type, entity_id, feature_name)
            
            # Convert to JSON if not a string
            if not isinstance(value, str):
                value = json.dumps(value)
            
            if ttl:
                self.redis.setex(key, ttl, value)
            else:
                self.redis.set(key, value)
            
            return True
        except Exception as e:
            logger.error(f"Error setting feature {feature_name}: {e}")
            return False
    
    def set_features(self, feature_type, entity_id, features, ttl=None):
        """Set multiple feature values for an entity"""
        if not self.redis or not features:
            return False
        
        try:
            pipeline = self.redis.pipeline()
            
            for name, value in features.items():
                key = self._get_key(feature_type, entity_id, name)
                
                # Convert to JSON if not a string
                if not isinstance(value, str):
                    value = json.dumps(value)
                
                if ttl:
                    pipeline.setex(key, ttl, value)
                else:
                    pipeline.set(key, value)
            
            pipeline.execute()
            return True
        except Exception as e:
            logger.error(f"Error setting features for {entity_id}: {e}")
            return False
    
    def get_vector(self, feature_type, entity_id, vector_name="embedding"):
        """Get embedding vector for an entity"""
        result = self.get_feature(feature_type, entity_id, vector_name)
        
        if result and isinstance(result, list):
            return np.array(result, dtype=np.float32)
        
        return None
    
    def set_vector(self, feature_type, entity_id, vector, vector_name="embedding", ttl=None):
        """Set embedding vector for an entity"""
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()
        
        return self.set_feature(feature_type, entity_id, vector_name, vector, ttl)
    
    def increment_feature(self, feature_type, entity_id, feature_name, amount=1):
        """Increment a numeric feature value"""
        if not self.redis:
            return None
        
        try:
            key = self._get_key(feature_type, entity_id, feature_name)
            new_value = self.redis.incrbyfloat(key, amount)
            return float(new_value)
        except Exception as e:
            logger.error(f"Error incrementing feature {feature_name}: {e}")
            return None
    
    def delete_feature(self, feature_type, entity_id, feature_name):
        """Delete a feature"""
        if not self.redis:
            return False
        
        try:
            key = self._get_key(feature_type, entity_id, feature_name)
            self.redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error deleting feature {feature_name}: {e}")
            return False
    
    def delete_all_features(self, feature_type, entity_id):
        """Delete all features for an entity"""
        if not self.redis:
            return False
        
        try:
            pattern = self._get_key(feature_type, entity_id, "*")
            keys = list(self.redis.scan_iter(match=pattern))
            
            if keys:
                self.redis.delete(*keys)
            
            return True
        except Exception as e:
            logger.error(f"Error deleting features for {entity_id}: {e}")
            return False
    
    def get_feature_stats(self, feature_type=None):
        """Get statistics about stored features"""
        if not self.redis:
            return {}
        
        try:
            stats = {}
            if feature_type:
                pattern = f"{self.prefix}:{feature_type}:*"
            else:
                pattern = f"{self.prefix}:*"
            
            # Get key count
            stats["feature_count"] = len(list(self.redis.scan_iter(match=pattern)))
            
            # Get memory usage
            memory_info = self.redis.info("memory")
            stats["memory_used"] = memory_info.get("used_memory_human")
            
            return stats
        except Exception as e:
            logger.error(f"Error getting feature stats: {e}")
            return {}
    
    def get_entities(self, feature_type, pattern="*"):
        """Get all entity IDs of a specific type"""
        if not self.redis:
            return []
        
        try:
            key_pattern = f"{self.prefix}:{feature_type}:{pattern}:*"
            entity_ids = set()
            
            for key in self.redis.scan_iter(match=key_pattern):
                parts = key.split(":")
                if len(parts) >= 4:
                    entity_ids.add(parts[2])
            
            return list(entity_ids)
        except Exception as e:
            logger.error(f"Error getting entities: {e}")
            return []

class FeatureProcessor(BaseConsumer):
    """
    Process events from Kafka and update features in real-time
    """
    
    def __init__(self, topics=None, batch_size=100, update_interval_seconds=10):
        # Initialize Kafka consumer
        topics = topics or ["amazon-user-events", "amazon-product-views", "amazon-purchases"]
        super().__init__(topics=topics)
        
        # Initialize feature store
        self.feature_store = FeatureStore()
        
        # Configure batching
        self.batch_size = batch_size
        self.update_interval_seconds = update_interval_seconds
        self.event_queue = queue.Queue()
        
        # Initialize processing thread
        self.last_update_time = datetime.now()
        self.processing_thread = threading.Thread(target=self._process_events)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Initialize metrics producer
        self.metrics_producer = get_producer("metrics", metrics_type="features")
    
    def process_message(self, message):
        """Process incoming event message from Kafka"""
        try:
            # Validate message
            if not message.get('user_id') or not message.get('event_type'):
                logger.warning(f"Incomplete event message: {message}")
                return
            
            # Add to processing queue
            self.event_queue.put(message)
            
        except Exception as e:
            logger.error(f"Error processing event message: {e}")
    
    def _process_events(self):
        """Background thread to process events in batches"""
        while True:
            try:
                # Check if we should process a batch
                current_time = datetime.now()
                time_since_update = (current_time - self.last_update_time).total_seconds()
                
                # Process if we have enough events or enough time has passed
                if (self.event_queue.qsize() >= self.batch_size or 
                    time_since_update >= self.update_interval_seconds):
                    
                    # Collect events from queue
                    events = []
                    try:
                        while len(events) < self.batch_size and not self.event_queue.empty():
                            events.append(self.event_queue.get_nowait())
                            self.event_queue.task_done()
                    except queue.Empty:
                        pass
                    
                    if events:
                        # Process batch of events
                        self._process_event_batch(events)
                        self.last_update_time = current_time
                
                # Sleep to avoid high CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in event processing thread: {e}")
                time.sleep(5)  # Sleep longer on error
    
    def _process_event_batch(self, events):
        """Process a batch of events and update features"""
        try:
            logger.info(f"Processing batch of {len(events)} events for feature updates")
            
            # Group events by user and product
            user_events = {}
            product_events = {}
            
            # Process each event
            for event in events:
                user_id = event.get('user_id')
                product_id = event.get('product_id')
                event_type = event.get('event_type')
                timestamp = event.get('timestamp', int(time.time() * 1000))
                
                # Group by user
                if user_id:
                    if user_id not in user_events:
                        user_events[user_id] = []
                    user_events[user_id].append(event)
                
                # Group by product
                if product_id:
                    if product_id not in product_events:
                        product_events[product_id] = []
                    product_events[product_id].append(event)
            
            # Update user features
            for user_id, user_event_list in user_events.items():
                self._update_user_features(user_id, user_event_list)
            
            # Update product features
            for product_id, product_event_list in product_events.items():
                self._update_product_features(product_id, product_event_list)
            
            # Track metrics
            self._track_batch_metrics(events)
            
            return True
        except Exception as e:
            logger.error(f"Error processing event batch: {e}")
            return False
    
    def _update_user_features(self, user_id, events):
        """Update features for a user based on events"""
        try:
            # Get existing features
            existing_features = self.feature_store.get_features('user', user_id)
            
            # Initialize new features
            new_features = {}
            
            # Process each event
            for event in events:
                event_type = event.get('event_type')
                product_id = event.get('product_id')
                timestamp = event.get('timestamp', int(time.time() * 1000))
                
                # Update features based on event type
                if event_type == 'view':
                    # Increment view count
                    view_count = existing_features.get('view_count', 0) + 1
                    new_features['view_count'] = view_count
                    
                    # Update recently viewed products
                    recent_views = existing_features.get('recent_views', [])
                    if isinstance(recent_views, list) and product_id not in recent_views:
                        recent_views = [product_id] + recent_views
                        recent_views = recent_views[:20]  # Keep only most recent 20
                        new_features['recent_views'] = recent_views
                
                elif event_type == 'purchase':
                    # Increment purchase count
                    purchase_count = existing_features.get('purchase_count', 0) + 1
                    new_features['purchase_count'] = purchase_count
                    
                    # Update purchased products
                    purchased = existing_features.get('purchased_products', [])
                    if isinstance(purchased, list) and product_id not in purchased:
                        purchased = [product_id] + purchased
                        purchased = purchased[:50]  # Keep only most recent 50
                        new_features['purchased_products'] = purchased
                
                elif event_type == 'rating':
                    # Update rating count and average
                    rating_count = existing_features.get('rating_count', 0) + 1
                    new_features['rating_count'] = rating_count
                    
                    # Calculate new average rating
                    rating = float(event.get('rating', 0))
                    prev_avg = existing_features.get('avg_rating', 0)
                    new_avg = ((prev_avg * (rating_count - 1)) + rating) / rating_count
                    new_features['avg_rating'] = new_avg
                
                # Update last active timestamp
                new_features['last_active'] = timestamp
            
            # Set all updated features
            if new_features:
                self.feature_store.set_features('user', user_id, new_features)
                logger.debug(f"Updated features for user {user_id}: {list(new_features.keys())}")
            
            return True
        except Exception as e:
            logger.error(f"Error updating user features for {user_id}: {e}")
            return False
    
    def _update_product_features(self, product_id, events):
        """Update features for a product based on events"""
        try:
            # Get existing features
            existing_features = self.feature_store.get_features('product', product_id)
            
            # Initialize new features
            new_features = {}
            
            # Count events by type
            view_count = 0
            purchase_count = 0
            ratings = []
            
            # Process each event
            for event in events:
                event_type = event.get('event_type')
                user_id = event.get('user_id')
                timestamp = event.get('timestamp', int(time.time() * 1000))
                
                # Count by event type
                if event_type == 'view':
                    view_count += 1
                elif event_type == 'purchase':
                    purchase_count += 1
                elif event_type == 'rating':
                    rating = float(event.get('rating', 0))
                    ratings.append(rating)
            
            # Update view count
            if view_count > 0:
                total_views = existing_features.get('view_count', 0) + view_count
                new_features['view_count'] = total_views
            
            # Update purchase count
            if purchase_count > 0:
                total_purchases = existing_features.get('purchase_count', 0) + purchase_count
                new_features['purchase_count'] = total_purchases
            
            # Update ratings
            if ratings:
                rating_count = existing_features.get('rating_count', 0) + len(ratings)
                new_features['rating_count'] = rating_count
                
                # Calculate new average rating
                prev_avg = existing_features.get('avg_rating', 0)
                prev_count = rating_count - len(ratings)
                new_avg = ((prev_avg * prev_count) + sum(ratings)) / rating_count
                new_features['avg_rating'] = new_avg
            
            # Calculate popularity score
            if 'view_count' in new_features or 'purchase_count' in new_features or 'rating_count' in new_features:
                total_views = new_features.get('view_count', existing_features.get('view_count', 0))
                total_purchases = new_features.get('purchase_count', existing_features.get('purchase_count', 0))
                rating_score = new_features.get('avg_rating', existing_features.get('avg_rating', 0)) * new_features.get('rating_count', existing_features.get('rating_count', 0))
                
                # Simple popularity formula: views + purchases*10 + rating_score*5
                popularity = total_views + (total_purchases * 10) + (rating_score * 5)
                new_features['popularity'] = popularity
            
            # Set all updated features
            if new_features:
                self.feature_store.set_features('product', product_id, new_features)
                logger.debug(f"Updated features for product {product_id}: {list(new_features.keys())}")
            
            return True
        except Exception as e:
            logger.error(f"Error updating product features for {product_id}: {e}")
            return False
    
    def _track_batch_metrics(self, events):
        """Track metrics about the batch processing"""
        try:
            # Count event types
            event_types = {}
            for event in events:
                event_type = event.get('event_type', 'unknown')
                if event_type not in event_types:
                    event_types[event_type] = 0
                event_types[event_type] += 1
            
            # Send metrics
            for event_type, count in event_types.items():
                self.metrics_producer.send_metric(
                    metric_name=f"feature_events_{event_type}",
                    metric_value=count,
                    tags={
                        'batch_size': len(events),
                        'timestamp': datetime.now().isoformat()
                    }
                )
            
            # Track feature store stats
            stats = self.feature_store.get_feature_stats()
            if 'feature_count' in stats:
                self.metrics_producer.send_metric(
                    metric_name="feature_store_count",
                    metric_value=stats['feature_count']
                )
            
            return True
        except Exception as e:
            logger.error(f"Error tracking feature metrics: {e}")
            return False 