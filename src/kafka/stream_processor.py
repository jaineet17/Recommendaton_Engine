"""
Stream processor for real-time Kafka message processing in the Amazon recommendation system.
"""

import logging
import signal
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.data.database import load_config
from src.kafka.consumer import get_consumer
from src.kafka.producer import get_producer

# Set up logging
logger = logging.getLogger(__name__)

# Load configuration
config = load_config()


class StreamProcessor(ABC):
    """
    Base class for stream processors that process Kafka messages in real-time.
    """
    
    def __init__(self, consumer_type: str, producer_type: Optional[str] = None, 
                run_interval_seconds: Optional[float] = None, **kwargs):
        """
        Initialize stream processor.
        
        Args:
            consumer_type: Type of consumer to use
            producer_type: Type of producer to use (optional)
            run_interval_seconds: Interval between processing runs (None for continuous)
            **kwargs: Additional arguments for the consumer/producer
        """
        self.consumer_type = consumer_type
        self.producer_type = producer_type
        self.run_interval_seconds = run_interval_seconds
        self.kafka_kwargs = kwargs
        
        self.consumer = None
        self.producer = None
        self.running = False
        self.thread = None
        
        # Statistics
        self.stats = {
            'processed_messages': 0,
            'errors': 0,
            'start_time': None,
            'last_processed_time': None
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_exit)
        signal.signal(signal.SIGTERM, self._handle_exit)
    
    def initialize(self) -> None:
        """Initialize the stream processor."""
        logger.info(f"Initializing {self.__class__.__name__}")
        
        # Create consumer
        self.consumer = get_consumer(
            self.consumer_type, 
            handler=self._message_handler,
            **self.kafka_kwargs
        )
        
        # Create producer if needed
        if self.producer_type:
            self.producer = get_producer(self.producer_type, **self.kafka_kwargs)
        
        # Additional initialization (can be overridden)
        self._initialize()
        
        logger.info(f"{self.__class__.__name__} initialized")
    
    def _initialize(self) -> None:
        """
        Additional initialization logic.
        This method can be overridden by subclasses.
        """
        pass
    
    def start(self, blocking: bool = True) -> None:
        """
        Start the stream processor.
        
        Args:
            blocking: Whether to run in blocking mode
        """
        if self.running:
            logger.warning(f"{self.__class__.__name__} is already running")
            return
        
        # Initialize if not already
        if not self.consumer:
            self.initialize()
        
        self.running = True
        self.stats['start_time'] = datetime.now()
        logger.info(f"Starting {self.__class__.__name__}")
        
        if blocking:
            self._run()
        else:
            self.thread = threading.Thread(target=self._run)
            self.thread.daemon = True
            self.thread.start()
    
    def stop(self) -> None:
        """Stop the stream processor."""
        logger.info(f"Stopping {self.__class__.__name__}")
        self.running = False
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=10)
            if self.thread.is_alive():
                logger.warning(f"Thread for {self.__class__.__name__} did not terminate gracefully")
        
        # Close Kafka connections
        if self.consumer:
            self.consumer.close()
        
        if self.producer:
            self.producer.close()
        
        # Log statistics
        duration = datetime.now() - self.stats['start_time'] if self.stats['start_time'] else timedelta(0)
        logger.info(f"{self.__class__.__name__} stopped. Stats: {self.stats['processed_messages']} messages processed, "
                    f"{self.stats['errors']} errors in {duration.total_seconds():.1f} seconds")
    
    def _run(self) -> None:
        """Main processing loop."""
        try:
            if self.run_interval_seconds is None:
                # Continuous processing
                logger.info(f"{self.__class__.__name__} running in continuous mode")
                self.consumer.consume()
            else:
                # Interval-based processing
                logger.info(f"{self.__class__.__name__} running with {self.run_interval_seconds}s interval")
                while self.running:
                    start_time = time.time()
                    
                    # Process a batch of messages
                    try:
                        self.consumer.consume(timeout_ms=int(self.run_interval_seconds * 500), max_records=100)
                    except Exception as e:
                        logger.error(f"Error in consume cycle: {e}")
                        self.stats['errors'] += 1
                    
                    # Sleep if needed to maintain interval
                    elapsed = time.time() - start_time
                    sleep_time = max(0, self.run_interval_seconds - elapsed)
                    if sleep_time > 0 and self.running:
                        time.sleep(sleep_time)
        
        except Exception as e:
            logger.error(f"Error in stream processor: {e}")
            self.stats['errors'] += 1
        
        finally:
            if self.running:
                self.stop()
    
    def _message_handler(self, message: Dict[str, Any]) -> None:
        """
        Handle incoming messages from Kafka.
        
        Args:
            message: Message from Kafka
        """
        try:
            self.process_message(message)
            self.stats['processed_messages'] += 1
            self.stats['last_processed_time'] = datetime.now()
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.stats['errors'] += 1
    
    def _handle_exit(self, signum, frame) -> None:
        """
        Handle exit signals gracefully.
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        logger.info(f"Received signal {signum}, shutting down")
        self.stop()
    
    @abstractmethod
    def process_message(self, message: Dict[str, Any]) -> None:
        """
        Process a message from Kafka.
        This method must be implemented by subclasses.
        
        Args:
            message: Message from Kafka
        """
        pass


class UserEventProcessor(StreamProcessor):
    """
    Processor for user events, analyzing user behavior in real-time.
    """
    
    def __init__(self, window_size_minutes: int = 30, output_interval_seconds: int = 60, **kwargs):
        """
        Initialize user event processor.
        
        Args:
            window_size_minutes: Size of the sliding window for events in minutes
            output_interval_seconds: Interval for outputting statistics
            **kwargs: Additional arguments for the base processor
        """
        super().__init__(
            consumer_type='product_event',
            producer_type='metrics',
            run_interval_seconds=1.0,
            **kwargs
        )
        
        self.window_size_minutes = window_size_minutes
        self.output_interval_seconds = output_interval_seconds
        
        # Event storage
        self.events = {}  # user_id -> list of events
        self.session_data = {}  # session_id -> session data
        
        # Last output time
        self.last_output_time = datetime.now()
    
    def _initialize(self) -> None:
        """Initialize processor-specific resources."""
        logger.info(f"Initializing user event processor with {self.window_size_minutes}m window")
        
        # Create in-memory event storage with sliding window
        self.events = defaultdict(lambda: deque(maxlen=1000))
        self.session_data = {}
    
    def process_message(self, message: Dict[str, Any]) -> None:
        """
        Process a user event message.
        
        Args:
            message: User event message from Kafka
        """
        # Extract fields
        user_id = message.get('user_id')
        product_id = message.get('product_id')
        event_type = message.get('event_type')
        session_id = message.get('session_id')
        timestamp = message.get('timestamp')
        
        if not all([user_id, product_id, event_type, session_id]):
            logger.warning(f"Incomplete message: {message}")
            return
        
        # Add event to user's queue
        event_time = datetime.fromisoformat(timestamp) if timestamp else datetime.now()
        event = {
            'user_id': user_id,
            'product_id': product_id,
            'event_type': event_type,
            'session_id': session_id,
            'timestamp': event_time
        }
        
        self.events[user_id].append(event)
        
        # Update session data
        if session_id not in self.session_data:
            self.session_data[session_id] = {
                'user_id': user_id,
                'start_time': event_time,
                'last_time': event_time,
                'events': 1,
                'products': {product_id},
                'views': 1 if event_type == 'view' else 0,
                'add_to_carts': 1 if event_type == 'add_to_cart' else 0,
                'purchases': 1 if event_type == 'purchase' else 0,
                'recommend_clicks': 1 if event_type == 'recommend_click' else 0
            }
        else:
            session = self.session_data[session_id]
            session['last_time'] = event_time
            session['events'] += 1
            session['products'].add(product_id)
            
            if event_type == 'view':
                session['views'] += 1
            elif event_type == 'add_to_cart':
                session['add_to_carts'] += 1
            elif event_type == 'purchase':
                session['purchases'] += 1
            elif event_type == 'recommend_click':
                session['recommend_clicks'] += 1
        
        # Output statistics periodically
        now = datetime.now()
        if (now - self.last_output_time).total_seconds() >= self.output_interval_seconds:
            self._output_statistics()
            self.last_output_time = now
    
    def _output_statistics(self) -> None:
        """Output user behavior statistics."""
        now = datetime.now()
        window_start = now - timedelta(minutes=self.window_size_minutes)
        
        # Clean up old events
        for user_id, events in list(self.events.items()):
            while events and events[0]['timestamp'] < window_start:
                events.popleft()
        
        # Clean up old sessions
        for session_id in list(self.session_data.keys()):
            if self.session_data[session_id]['last_time'] < window_start:
                del self.session_data[session_id]
        
        # Compute statistics
        active_users = len(self.events)
        active_sessions = len(self.session_data)
        
        if active_sessions == 0:
            logger.info("No active sessions in current window")
            return
        
        # Session statistics
        total_events = sum(s['events'] for s in self.session_data.values())
        total_views = sum(s['views'] for s in self.session_data.values())
        total_carts = sum(s['add_to_carts'] for s in self.session_data.values())
        total_purchases = sum(s['purchases'] for s in self.session_data.values())
        total_rec_clicks = sum(s['recommend_clicks'] for s in self.session_data.values())
        
        # Average session statistics
        avg_session_length = total_events / active_sessions
        avg_session_products = sum(len(s['products']) for s in self.session_data.values()) / active_sessions
        
        # Conversion rates
        view_to_cart = total_carts / total_views if total_views > 0 else 0
        cart_to_purchase = total_purchases / total_carts if total_carts > 0 else 0
        rec_click_rate = total_rec_clicks / total_views if total_views > 0 else 0
        
        # Log statistics
        logger.info(f"Window statistics ({self.window_size_minutes}m): "
                    f"{active_users} users, {active_sessions} sessions, "
                    f"{total_events} events, {total_views} views, {total_purchases} purchases")
        
        # Send metrics to Kafka
        if self.producer:
            metrics = {
                'active_users': active_users,
                'active_sessions': active_sessions,
                'total_events': total_events,
                'total_views': total_views,
                'total_add_to_carts': total_carts,
                'total_purchases': total_purchases,
                'total_recommend_clicks': total_rec_clicks,
                'avg_session_length': avg_session_length,
                'avg_session_products': avg_session_products,
                'view_to_cart_rate': view_to_cart,
                'cart_to_purchase_rate': cart_to_purchase,
                'recommendation_click_rate': rec_click_rate
            }
            
            for name, value in metrics.items():
                self.producer.send_metric(
                    metric_name=f"user_behavior.{name}",
                    metric_value=value,
                    tags={
                        'window_minutes': str(self.window_size_minutes),
                        'source': 'stream_processor'
                    }
                )


class RealtimeRecommender(StreamProcessor):
    """
    Processor that updates real-time recommendations based on user behavior.
    """
    
    def __init__(self, update_interval_seconds: int = 30, cache_size: int = 1000, **kwargs):
        """
        Initialize real-time recommender.
        
        Args:
            update_interval_seconds: Interval for updating recommendations
            cache_size: Size of recommendation cache per user
            **kwargs: Additional arguments for the base processor
        """
        super().__init__(
            consumer_type='product_event',
            producer_type=None,  # No direct producer, we'll use the database
            run_interval_seconds=1.0,
            **kwargs
        )
        
        self.update_interval_seconds = update_interval_seconds
        self.cache_size = cache_size
        
        # User behavior storage
        self.user_products = {}  # user_id -> set of product_ids
        self.product_views = {}  # product_id -> count
        self.related_products = {}  # product_id -> {related_product_id: score}
        
        # Last update time
        self.last_update_time = datetime.now()
    
    def _initialize(self) -> None:
        """Initialize recommender-specific resources."""
        logger.info(f"Initializing real-time recommender with {self.update_interval_seconds}s update interval")
        
        # Initialize storage
        self.user_products = defaultdict(set)
        self.product_views = defaultdict(int)
        self.related_products = defaultdict(lambda: defaultdict(float))
    
    def process_message(self, message: Dict[str, Any]) -> None:
        """
        Process a user event message for real-time recommendations.
        
        Args:
            message: User event message from Kafka
        """
        # Extract fields
        user_id = message.get('user_id')
        product_id = message.get('product_id')
        event_type = message.get('event_type')
        
        if not all([user_id, product_id, event_type]):
            logger.warning(f"Incomplete message: {message}")
            return
        
        # Update user-product interactions
        self.user_products[user_id].add(product_id)
        
        # Update product view counts
        if event_type == 'view':
            self.product_views[product_id] += 1
        
        # Update product relationships (items viewed in same session)
        session_id = message.get('session_id')
        if session_id:
            # Find other products viewed in this session
            for other_user_id, products in self.user_products.items():
                if other_user_id != user_id:
                    continue
                
                for other_product in products:
                    if other_product != product_id:
                        # Increase relationship score
                        self.related_products[product_id][other_product] += 1.0
                        self.related_products[other_product][product_id] += 1.0
        
        # Update recommendations periodically
        now = datetime.now()
        if (now - self.last_update_time).total_seconds() >= self.update_interval_seconds:
            self._update_recommendations()
            self.last_update_time = now
    
    def _update_recommendations(self) -> None:
        """Update real-time recommendations."""
        logger.info("Updating real-time recommendations")
        
        # Calculate top products overall (by views)
        sorted_products = sorted(self.product_views.items(), key=lambda x: x[1], reverse=True)
        top_overall = [p[0] for p in sorted_products[:self.cache_size]]
        
        # Calculate personalized recommendations for each user
        for user_id, viewed_products in self.user_products.items():
            if not viewed_products:
                continue
            
            # Skip if too many products (likely a bot)
            if len(viewed_products) > 1000:
                continue
            
            # Get related products for all products this user has viewed
            candidate_products = defaultdict(float)
            for product_id in viewed_products:
                for related_id, score in self.related_products[product_id].items():
                    if related_id not in viewed_products:  # Don't recommend already viewed products
                        candidate_products[related_id] += score
            
            # Sort candidates by score
            recommendations = sorted(candidate_products.items(), key=lambda x: x[1], reverse=True)
            top_n = [p[0] for p in recommendations[:self.cache_size]]
            
            # If not enough recommendations, fill with top overall
            if len(top_n) < self.cache_size:
                for product_id in top_overall:
                    if product_id not in viewed_products and product_id not in top_n:
                        top_n.append(product_id)
                        if len(top_n) >= self.cache_size:
                            break
            
            # Store recommendations (in a real system, this would update a database or cache)
            logger.debug(f"Updated recommendations for user {user_id}: {len(top_n)} items")
        
        logger.info(f"Updated recommendations for {len(self.user_products)} users")


class DataDriftDetector(StreamProcessor):
    """
    Processor that detects data drift in real-time.
    """
    
    def __init__(self, window_size_minutes: int = 60, alert_threshold: float = 0.2, **kwargs):
        """
        Initialize data drift detector.
        
        Args:
            window_size_minutes: Size of the sliding window for analysis
            alert_threshold: Threshold for alerting on drift
            **kwargs: Additional arguments for the base processor
        """
        super().__init__(
            consumer_type='product_event',
            producer_type='metrics',
            run_interval_seconds=5.0,
            **kwargs
        )
        
        self.window_size_minutes = window_size_minutes
        self.alert_threshold = alert_threshold
        
        # Reference distributions (baseline)
        self.reference_event_dist = {'view': 0, 'add_to_cart': 0, 'purchase': 0, 'recommend_click': 0}
        self.reference_total = 0
        
        # Current window data
        self.current_events = deque(maxlen=100000)  # Limit memory usage
        
        # Last check time
        self.last_check_time = datetime.now()
    
    def _initialize(self) -> None:
        """Initialize drift detector resources."""
        logger.info(f"Initializing data drift detector with {self.window_size_minutes}m window")
    
    def process_message(self, message: Dict[str, Any]) -> None:
        """
        Process a message for data drift detection.
        
        Args:
            message: Message from Kafka
        """
        # Extract fields
        event_type = message.get('event_type')
        timestamp = message.get('timestamp')
        
        if not event_type:
            return
        
        # Add to current window
        event_time = datetime.fromisoformat(timestamp) if timestamp else datetime.now()
        self.current_events.append({
            'event_type': event_type,
            'timestamp': event_time
        })
        
        # Update reference distribution if not set
        if self.reference_total == 0:
            self.reference_event_dist[event_type] += 1
            self.reference_total += 1
        
        # Check for drift periodically
        now = datetime.now()
        if (now - self.last_check_time).total_seconds() >= 60:  # Check every minute
            self._check_for_drift()
            self.last_check_time = now
    
    def _check_for_drift(self) -> None:
        """Check for data drift in the current window."""
        now = datetime.now()
        window_start = now - timedelta(minutes=self.window_size_minutes)
        
        # Filter events in the current window
        window_events = [e for e in self.current_events if e['timestamp'] >= window_start]
        
        if not window_events:
            logger.info("No events in current window for drift detection")
            return
        
        # Calculate current distribution
        current_dist = {'view': 0, 'add_to_cart': 0, 'purchase': 0, 'recommend_click': 0}
        for event in window_events:
            event_type = event['event_type']
            if event_type in current_dist:
                current_dist[event_type] += 1
        
        current_total = sum(current_dist.values())
        
        # If reference is not set, use this as reference
        if self.reference_total < 100:
            logger.info(f"Setting reference distribution with {current_total} events")
            self.reference_event_dist = current_dist.copy()
            self.reference_total = current_total
            return
        
        # Calculate drift score using Jensen-Shannon divergence approximation
        drift_score = 0
        for event_type in current_dist:
            ref_prob = self.reference_event_dist[event_type] / max(1, self.reference_total)
            cur_prob = current_dist[event_type] / max(1, current_total)
            
            # Simple difference in probabilities
            type_drift = abs(ref_prob - cur_prob)
            drift_score += type_drift
        
        drift_score /= len(current_dist)  # Average across event types
        
        # Log and send metrics
        logger.info(f"Drift score: {drift_score:.4f} (threshold: {self.alert_threshold:.4f})")
        
        if self.producer:
            self.producer.send_metric(
                metric_name="data_drift.score",
                metric_value=drift_score,
                tags={
                    'window_minutes': str(self.window_size_minutes),
                    'threshold': str(self.alert_threshold)
                }
            )
        
        # Alert if drift exceeds threshold
        if drift_score > self.alert_threshold:
            logger.warning(f"DATA DRIFT DETECTED: Score {drift_score:.4f} exceeds threshold {self.alert_threshold:.4f}")
            
            if self.producer:
                self.producer.send_metric(
                    metric_name="data_drift.alert",
                    metric_value=1,
                    tags={
                        'score': f"{drift_score:.4f}",
                        'threshold': f"{self.alert_threshold:.4f}"
                    }
                )


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Example usage
    logger.info("Starting User Event Processor")
    processor = UserEventProcessor(window_size_minutes=5, output_interval_seconds=30)
    
    try:
        processor.start(blocking=True)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        processor.stop()
        logger.info("Processor stopped") 