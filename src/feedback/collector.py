import os
import logging
import json
from datetime import datetime
import threading
import queue
import time

logger = logging.getLogger(__name__)

class FeedbackCollector:
    """Collects and manages user feedback for recommendations."""
    
    def __init__(self, kafka_producer=None, batch_size=50, flush_interval_seconds=60):
        """Initialize feedback collector.
        
        Args:
            kafka_producer: Optional Kafka producer for sending feedback events
            batch_size: Maximum size of feedback batch before flush
            flush_interval_seconds: Maximum time between flushes
        """
        self.kafka_producer = kafka_producer
        self.batch_size = batch_size
        self.flush_interval_seconds = flush_interval_seconds
        
        # Feedback buffer
        self.feedback_buffer = queue.Queue()
        
        # Last flush time
        self.last_flush_time = time.time()
        
        # Start background thread for flushing feedback
        self.is_running = True
        self.flush_thread = threading.Thread(target=self._flush_periodically, daemon=True)
        self.flush_thread.start()
        
        logger.info(f"Feedback collector initialized (batch_size={batch_size}, flush_interval={flush_interval_seconds}s)")
    
    def record_feedback(self, user_id, item_id, event_type, rating=None, context=None, metadata=None):
        """Record user feedback.
        
        Args:
            user_id: User ID
            item_id: Item ID
            event_type: Type of event (click, purchase, view)
            rating: Optional rating value (1-5)
            context: Optional context (e.g., page, source)
            metadata: Additional metadata about the event
            
        Returns:
            bool: Success
        """
        try:
            # Create feedback event
            feedback_event = {
                'user_id': user_id,
                'item_id': item_id,
                'event_type': event_type,
                'timestamp': datetime.now().isoformat(),
                'context': context or {}
            }
            
            # Add optional fields
            if rating is not None:
                feedback_event['rating'] = rating
                
            if metadata:
                feedback_event['metadata'] = metadata
            
            # Add to buffer
            self.feedback_buffer.put(feedback_event)
            
            # Check if we should flush
            if self.feedback_buffer.qsize() >= self.batch_size:
                self.flush()
                
            return True
        except Exception as e:
            logger.error(f"Error recording feedback: {e}")
            return False
            
    def flush(self):
        """Flush feedback buffer to storage/Kafka."""
        try:
            events = []
            while not self.feedback_buffer.empty():
                try:
                    events.append(self.feedback_buffer.get_nowait())
                    self.feedback_buffer.task_done()
                except queue.Empty:
                    break
                
            if not events:
                return True
                
            # Send to Kafka if available
            if self.kafka_producer:
                for event in events:
                    self.kafka_producer.send_message(event)
                logger.debug(f"Sent {len(events)} feedback events to Kafka")
            else:
                # Log to file as fallback
                self._log_to_file(events)
                
            self.last_flush_time = time.time()
            return True
        except Exception as e:
            logger.error(f"Error flushing feedback: {e}")
            return False
    
    def _flush_periodically(self):
        """Background thread to flush feedback periodically."""
        while self.is_running:
            time.sleep(1)  # Check every second
            
            # Check if enough time has passed since last flush
            if (time.time() - self.last_flush_time) > self.flush_interval_seconds:
                if not self.feedback_buffer.empty():
                    self.flush()
    
    def _log_to_file(self, events):
        """Log feedback events to file as fallback."""
        try:
            os.makedirs('data/feedback', exist_ok=True)
            
            filename = f"data/feedback/feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(events, f, indent=2)
                
            logger.info(f"Logged {len(events)} feedback events to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error logging feedback to file: {e}")
            return False
    
    def shutdown(self):
        """Shutdown the feedback collector gracefully."""
        logger.info("Shutting down feedback collector...")
        self.is_running = False
        
        # Join the thread with timeout
        if self.flush_thread.is_alive():
            self.flush_thread.join(timeout=5)
            
        # Final flush
        self.flush()
        logger.info("Feedback collector shutdown complete") 