"""
Kafka consumers for the Amazon recommendation system.
"""

import json
import logging
import time
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

from kafka import KafkaConsumer
from kafka.errors import KafkaError, NoBrokersAvailable, KafkaConnectionError

from src.data.database import load_config

# Set up logging
logger = logging.getLogger(__name__)

# Load configuration
config = load_config()
kafka_config = config.get('kafka', {})
BOOTSTRAP_SERVERS = kafka_config.get('bootstrap_servers', ['localhost:9092'])
CONSUMER_GROUP_ID = kafka_config.get('consumer_group_id', 'amazon-recommendation-service')
AUTO_OFFSET_RESET = kafka_config.get('auto_offset_reset', 'earliest')
ENABLE_AUTO_COMMIT = kafka_config.get('enable_auto_commit', True)
TOPICS = kafka_config.get('topics', {
    'amazon_reviews': 'amazon-reviews',
    'product_events': 'product-events',
    'recommendation_requests': 'recommendation-requests',
    'model_metrics': 'model-metrics',
    'system_metrics': 'system-metrics'
})


class BaseConsumer(ABC):
    """Base class for Kafka consumers in the recommendation system."""
    
    def __init__(self, 
                topics: List[str], 
                group_id: str = CONSUMER_GROUP_ID,
                bootstrap_servers: List[str] = None,
                auto_offset_reset: str = AUTO_OFFSET_RESET,
                enable_auto_commit: bool = ENABLE_AUTO_COMMIT,
                retry_count: int = 3, 
                retry_delay: int = 5):
        """
        Initialize Kafka consumer.
        
        Args:
            topics: List of Kafka topics to consume from
            group_id: Consumer group ID
            bootstrap_servers: List of Kafka broker addresses
            auto_offset_reset: Where to start consuming from if no offset is stored
            enable_auto_commit: Whether to auto-commit offsets
            retry_count: Number of retries on connection failure
            retry_delay: Delay between retries in seconds
        """
        self.topics = topics
        self.group_id = group_id
        self.bootstrap_servers = bootstrap_servers or BOOTSTRAP_SERVERS
        self.auto_offset_reset = auto_offset_reset
        self.enable_auto_commit = enable_auto_commit
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.consumer = None
        self._connect()
    
    def _connect(self) -> None:
        """Connect to Kafka and create consumer."""
        # Check for test mode to skip actual Kafka connection
        if os.environ.get('TEST_MODE', 'false').lower() == 'true':
            logger.info("Running in TEST_MODE, skipping Kafka connection")
            self.consumer = None
            return

        # Get configuration
        bootstrap_servers = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
        group_id = os.environ.get('KAFKA_CONSUMER_GROUP_ID', 'amazon-recommendation-service')
        auto_offset_reset = os.environ.get('KAFKA_AUTO_OFFSET_RESET', 'earliest')
        
        # Number of connection attempts
        max_attempts = int(os.environ.get('KAFKA_MAX_CONNECTION_ATTEMPTS', 3))
        
        # Try to connect with retries
        for attempt in range(1, max_attempts + 1):
            try:
                self.consumer = KafkaConsumer(
                    *self.topics,
                    bootstrap_servers=bootstrap_servers,
                    group_id=f"{group_id}-{self.consumer_type}",
                    auto_offset_reset=auto_offset_reset,
                    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                    consumer_timeout_ms=1000,  # 1 second timeout to allow for clean shutdown
                    enable_auto_commit=False
                )
                logger.info(f"Connected to Kafka brokers at {bootstrap_servers}")
                return
            except (NoBrokersAvailable, KafkaConnectionError) as e:
                if attempt < max_attempts:
                    logger.warning(f"Failed to connect to Kafka brokers. Retrying in 5 seconds... ({attempt}/{max_attempts})")
                    time.sleep(5)
                else:
                    logger.error(f"Failed to connect to Kafka brokers after {max_attempts} attempts: {str(e)}")
                    # Don't raise, just set consumer to None
                    self.consumer = None
    
    def close(self) -> None:
        """Close the Kafka consumer."""
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka consumer closed")
    
    @abstractmethod
    def process_message(self, message: Dict[str, Any]) -> None:
        """
        Process a message from Kafka.
        
        Args:
            message: Message received from Kafka
        """
        pass
    
    def consume(self, timeout_ms: Optional[int] = None, max_records: Optional[int] = None) -> None:
        """
        Consume messages from Kafka.
        
        Args:
            timeout_ms: Timeout in milliseconds for polling
            max_records: Maximum number of records to consume
        """
        if not self.consumer:
            raise RuntimeError("Not connected to Kafka broker")
        
        try:
            # If max_records is set, consume exactly that many records
            if max_records is not None:
                record_count = 0
                
                while record_count < max_records:
                    records = self.consumer.poll(timeout_ms=timeout_ms or 1000, max_records=max_records - record_count)
                    
                    if not records:
                        logger.debug("No records received in poll")
                        continue
                    
                    for topic_partition, messages in records.items():
                        for message in messages:
                            try:
                                logger.debug(f"Received message: {message.value}")
                                self.process_message(message.value)
                                record_count += 1
                            except Exception as e:
                                logger.error(f"Error processing message: {e}")
                
                logger.info(f"Consumed {record_count} messages")
            
            # Otherwise, consume indefinitely
            else:
                for message in self.consumer:
                    try:
                        logger.debug(f"Received message: {message.value}")
                        self.process_message(message.value)
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
        
        except KafkaError as e:
            logger.error(f"Error consuming from Kafka: {e}")
            raise


class ReviewConsumer(BaseConsumer):
    """Consumer for Amazon review events."""
    
    def __init__(self, handler: Optional[Callable[[Dict[str, Any]], None]] = None, **kwargs):
        """
        Initialize review consumer.
        
        Args:
            handler: Function to handle review messages
            **kwargs: Additional arguments for the base consumer
        """
        super().__init__([TOPICS['amazon_reviews']], **kwargs)
        self.handler = handler
    
    def process_message(self, message: Dict[str, Any]) -> None:
        """
        Process a review message.
        
        Args:
            message: Review message from Kafka
        """
        # Validate message
        required_fields = ['user_id', 'product_id', 'rating', 'review_text']
        for field in required_fields:
            if field not in message:
                logger.warning(f"Review message missing required field: {field}")
                return
        
        # Process with custom handler if provided
        if self.handler:
            self.handler(message)
        else:
            # Default processing: log the review
            logger.info(f"Received review: user={message['user_id']} product={message['product_id']} rating={message['rating']}")


class ProductEventConsumer(BaseConsumer):
    """Consumer for product interaction events."""
    
    def __init__(self, handler: Optional[Callable[[Dict[str, Any]], None]] = None, **kwargs):
        """
        Initialize product event consumer.
        
        Args:
            handler: Function to handle product event messages
            **kwargs: Additional arguments for the base consumer
        """
        super().__init__([TOPICS['product_events']], **kwargs)
        self.handler = handler
    
    def process_message(self, message: Dict[str, Any]) -> None:
        """
        Process a product event message.
        
        Args:
            message: Product event message from Kafka
        """
        # Validate message
        required_fields = ['user_id', 'product_id', 'event_type']
        for field in required_fields:
            if field not in message:
                logger.warning(f"Product event message missing required field: {field}")
                return
        
        # Process with custom handler if provided
        if self.handler:
            self.handler(message)
        else:
            # Default processing: log the event
            logger.info(f"Received product event: user={message['user_id']} product={message['product_id']} type={message['event_type']}")


class RecommendationRequestConsumer(BaseConsumer):
    """Consumer for recommendation request events."""
    
    def __init__(self, handler: Optional[Callable[[Dict[str, Any]], None]] = None, **kwargs):
        """
        Initialize recommendation request consumer.
        
        Args:
            handler: Function to handle recommendation request messages
            **kwargs: Additional arguments for the base consumer
        """
        super().__init__([TOPICS['recommendation_requests']], **kwargs)
        self.handler = handler
    
    def process_message(self, message: Dict[str, Any]) -> None:
        """
        Process a recommendation request message.
        
        Args:
            message: Recommendation request message from Kafka
        """
        # Validate message
        required_fields = ['user_id', 'request_id', 'model_version']
        for field in required_fields:
            if field not in message:
                logger.warning(f"Recommendation request message missing required field: {field}")
                return
        
        # Process with custom handler if provided
        if self.handler:
            self.handler(message)
        else:
            # Default processing: log the request
            logger.info(f"Received recommendation request: user={message['user_id']} request_id={message['request_id']}")


class MetricsConsumer(BaseConsumer):
    """Consumer for metrics events."""
    
    def __init__(self, metrics_type: str = 'model', handler: Optional[Callable[[Dict[str, Any]], None]] = None, **kwargs):
        """
        Initialize metrics consumer.
        
        Args:
            metrics_type: Type of metrics to consume ('model' or 'system')
            handler: Function to handle metrics messages
            **kwargs: Additional arguments for the base consumer
        """
        if metrics_type == 'model':
            topics = [TOPICS['model_metrics']]
        elif metrics_type == 'system':
            topics = [TOPICS['system_metrics']]
        elif metrics_type == 'all':
            topics = [TOPICS['model_metrics'], TOPICS['system_metrics']]
        else:
            raise ValueError(f"Invalid metrics type: {metrics_type}. Must be 'model', 'system', or 'all'.")
        
        super().__init__(topics, **kwargs)
        self.handler = handler
    
    def process_message(self, message: Dict[str, Any]) -> None:
        """
        Process a metrics message.
        
        Args:
            message: Metrics message from Kafka
        """
        # Validate message
        required_fields = ['metric_name', 'metric_value']
        for field in required_fields:
            if field not in message:
                logger.warning(f"Metrics message missing required field: {field}")
                return
        
        # Process with custom handler if provided
        if self.handler:
            self.handler(message)
        else:
            # Default processing: log the metric
            logger.info(f"Received metric: {message['metric_name']}={message['metric_value']}")


# Factory function to get appropriate consumer
def get_consumer(consumer_type: str, handler: Optional[Callable[[Dict[str, Any]], None]] = None, **kwargs) -> BaseConsumer:
    """
    Get a Kafka consumer of the specified type.
    
    Args:
        consumer_type: Type of consumer ('review', 'product_event', 'recommendation_request', 'metrics')
        handler: Function to handle messages
        **kwargs: Additional arguments for the consumer
    
    Returns:
        Appropriate Kafka consumer instance
    
    Raises:
        ValueError: If consumer_type is invalid
    """
    if consumer_type == 'review':
        return ReviewConsumer(handler=handler, **kwargs)
    elif consumer_type == 'product_event':
        return ProductEventConsumer(handler=handler, **kwargs)
    elif consumer_type == 'recommendation_request':
        return RecommendationRequestConsumer(handler=handler, **kwargs)
    elif consumer_type == 'metrics':
        return MetricsConsumer(handler=handler, **kwargs)
    else:
        raise ValueError(f"Invalid consumer type: {consumer_type}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Define a custom handler
    def custom_handler(message):
        logger.info(f"Custom handling: {message}")
    
    # Test consuming product events
    try:
        consumer = get_consumer('product_event', handler=custom_handler)
        logger.info("Starting to consume messages (press Ctrl+C to stop)...")
        
        # Consume for a while, then exit
        consumer.consume(timeout_ms=5000, max_records=10)
        
        consumer.close()
    except KeyboardInterrupt:
        logger.info("Stopping consumer...")
    except Exception as e:
        logger.error(f"Error consuming messages: {e}") 