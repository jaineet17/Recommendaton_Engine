import os
import logging
import numpy as np
import torch
import time
from datetime import datetime, timedelta
import threading
import queue
import json
from src.kafka.consumer import BaseConsumer
from src.kafka.producer import get_producer
from src.config.mlflow_config import MLFlowConfig

logger = logging.getLogger(__name__)

class OnlineLearningProcessor(BaseConsumer):
    """Processes user feedback events from Kafka and updates models incrementally"""
    
    def __init__(self, models=None, batch_size=100, update_interval_minutes=15):
        # Initialize Kafka consumer for feedback events
        super().__init__(topics=["amazon-user-feedback"])
        
        # Models to update (can be passed in or loaded later)
        self.models = models or {}
        
        # Batching parameters
        self.batch_size = batch_size
        self.update_interval_minutes = update_interval_minutes
        
        # Initialize MLFlow for model tracking
        self.mlflow = MLFlowConfig()
        self.mlflow.initialize()
        
        # Initialize feedback buffer
        self.feedback_buffer = []
        
        # Initialize event queue for batch processing
        self.event_queue = queue.Queue()
        
        # Initialize last update time
        self.last_update_time = datetime.now()
        
        # Initialize background processing thread
        self.processing_thread = threading.Thread(target=self._process_events)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Initialize metrics producer
        self.metrics_producer = get_producer("metrics", metrics_type="online_learning")
    
    def process_message(self, message):
        """Process incoming feedback message from Kafka"""
        try:
            # Validate message
            required_fields = ['user_id', 'item_id', 'feedback_type', 'timestamp']
            for field in required_fields:
                if field not in message:
                    logger.warning(f"Missing required field {field} in feedback message")
                    return
            
            # Add to event queue for batch processing
            self.event_queue.put(message)
            
        except Exception as e:
            logger.error(f"Error processing feedback message: {e}")
    
    def _process_events(self):
        """Background thread to process events in batches"""
        while True:
            try:
                # Check if we should process a batch
                current_time = datetime.now()
                time_since_update = (current_time - self.last_update_time).total_seconds() / 60
                
                # Process if we have enough events or enough time has passed
                if (self.event_queue.qsize() >= self.batch_size or 
                    time_since_update >= self.update_interval_minutes):
                    
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
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in event processing thread: {e}")
                time.sleep(5)  # Sleep longer on error
    
    def _process_event_batch(self, events):
        """Process a batch of feedback events"""
        try:
            logger.info(f"Processing batch of {len(events)} feedback events")
            
            # Group events by model
            model_events = {}
            
            for event in events:
                model_name = event.get('model_name')
                
                # If no model specified, assign to all models
                if not model_name:
                    for model in self.models:
                        if model not in model_events:
                            model_events[model] = []
                        model_events[model].append(event)
                else:
                    if model_name not in model_events:
                        model_events[model_name] = []
                    model_events[model_name].append(event)
            
            # Update each model with its events
            for model_name, model_events_list in model_events.items():
                if model_name in self.models:
                    self._update_model(model_name, model_events_list)
                else:
                    logger.warning(f"Model {model_name} not loaded, skipping update")
            
            # Track metrics
            self._track_batch_metrics(events)
            
            return True
        except Exception as e:
            logger.error(f"Error processing event batch: {e}")
            return False
    
    def _update_model(self, model_name, events):
        """Update a specific model with events"""
        try:
            logger.info(f"Updating model {model_name} with {len(events)} events")
            
            model = self.models.get(model_name)
            if not model:
                logger.warning(f"Model {model_name} not available for update")
                return False
            
            # Implementation depends on model type
            if hasattr(model, 'update_incremental'):
                # Convert events to update format
                update_data = self._prepare_update_data(events, model_name)
                
                # Update model
                update_result = model.update_incremental(update_data)
                
                # Log update to MLFlow
                if update_result:
                    self._log_model_update(model_name, len(events), update_result)
                
                return update_result
            else:
                logger.warning(f"Model {model_name} doesn't support incremental updates")
                return False
            
        except Exception as e:
            logger.error(f"Error updating model {model_name}: {e}")
            return False
    
    def _prepare_update_data(self, events, model_name):
        """Prepare events data for model update"""
        # This implementation depends on the model's expected format
        try:
            # Default implementation for matrix factorization models
            users = []
            items = []
            ratings = []
            weights = []
            
            for event in events:
                user_id = event.get('user_id')
                item_id = event.get('item_id')
                feedback_type = event.get('feedback_type')
                timestamp = event.get('timestamp')
                
                # Convert event to rating and weight
                rating = 0.0
                weight = 1.0
                
                if feedback_type == 'click':
                    rating = 1.0
                    weight = 0.5
                elif feedback_type == 'purchase':
                    rating = 1.0
                    weight = 1.0
                elif feedback_type == 'rating':
                    rating = float(event.get('rating', 0)) / 5.0  # Normalize to 0-1
                    weight = 1.0
                elif feedback_type == 'view_time':
                    # Convert view time to a rating
                    view_time_seconds = float(event.get('view_time_seconds', 0))
                    # Normalize view time, e.g., 0-5 minutes to 0-1
                    rating = min(view_time_seconds / 300.0, 1.0)
                    weight = 0.7
                
                users.append(user_id)
                items.append(item_id)
                ratings.append(rating)
                weights.append(weight)
            
            return {
                'users': users,
                'items': items,
                'ratings': ratings,
                'weights': weights
            }
            
        except Exception as e:
            logger.error(f"Error preparing update data: {e}")
            return {}
    
    def _log_model_update(self, model_name, num_events, update_result):
        """Log model update to MLFlow"""
        try:
            metrics = {
                'num_events': num_events,
                'update_time': update_result.get('update_time', 0),
                'loss': update_result.get('loss', 0),
                'timestamp': datetime.now().isoformat()
            }
            
            # Log as a metric
            self.mlflow.log_metrics(model_name, metrics)
            
            # Also send to Kafka for monitoring
            self.metrics_producer.send_metric(
                metric_name=f"{model_name}_update",
                metric_value=update_result.get('loss', 0),
                tags={
                    'num_events': num_events,
                    'update_time': update_result.get('update_time', 0)
                }
            )
            
            return True
        except Exception as e:
            logger.error(f"Error logging model update: {e}")
            return False
    
    def _track_batch_metrics(self, events):
        """Track metrics about the batch processing"""
        try:
            # Count event types
            event_types = {}
            for event in events:
                event_type = event.get('feedback_type', 'unknown')
                if event_type not in event_types:
                    event_types[event_type] = 0
                event_types[event_type] += 1
            
            # Send metrics
            for event_type, count in event_types.items():
                self.metrics_producer.send_metric(
                    metric_name=f"feedback_{event_type}",
                    metric_value=count,
                    tags={
                        'batch_size': len(events),
                        'timestamp': datetime.now().isoformat()
                    }
                )
            
            return True
        except Exception as e:
            logger.error(f"Error tracking batch metrics: {e}")
            return False
    
    def load_model(self, model_name):
        """Load a model for online updates"""
        try:
            # Try to load from MLFlow
            model = self.mlflow.load_model(model_name)
            
            if model:
                self.models[model_name] = model
                logger.info(f"Loaded model {model_name} for online learning")
                return True
            else:
                logger.warning(f"Failed to load model {model_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False
    
    def save_model(self, model_name):
        """Save updated model to MLFlow"""
        try:
            model = self.models.get(model_name)
            if not model:
                logger.warning(f"Model {model_name} not available to save")
                return False
            
            # Get model parameters for logging
            params = {}
            if hasattr(model, 'get_params'):
                params = model.get_params()
            
            # Get metrics
            metrics = {
                'last_update': datetime.now().isoformat(),
                'num_updates': getattr(model, 'num_updates', 0)
            }
            
            # Log model to MLFlow
            run_id = self.mlflow.log_model(
                model=model,
                model_name=model_name,
                params=params,
                metrics=metrics
            )
            
            # Register as new version
            if run_id:
                version = self.mlflow.register_model(
                    run_id=run_id,
                    model_name=model_name,
                    stage="Production"
                )
                
                logger.info(f"Saved model {model_name} as version {version}")
                return True
            else:
                logger.warning(f"Failed to log model {model_name} to MLFlow")
                return False
                
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}")
            return False

class ContextualBandit:
    """Implements a contextual multi-armed bandit for exploration"""
    
    def __init__(self, num_arms=5, context_dim=10, alpha=0.1):
        self.num_arms = num_arms
        self.context_dim = context_dim
        self.alpha = alpha  # Exploration parameter
        
        # Initialize weights
        self.theta = np.zeros((num_arms, context_dim))
        
        # Initialize statistics
        self.A_inv = [np.eye(context_dim) for _ in range(num_arms)]
        self.b = [np.zeros((context_dim, 1)) for _ in range(num_arms)]
        
        # Track rewards
        self.rewards = [[] for _ in range(num_arms)]
    
    def select_arm(self, context):
        """Select arm based on UCB policy"""
        try:
            context = np.array(context).reshape(-1, 1)  # Ensure column vector
            
            # Calculate UCB for each arm
            ucb_values = np.zeros(self.num_arms)
            
            for arm in range(self.num_arms):
                # Calculate expected reward
                theta = self.theta[arm].reshape(-1, 1)
                expected_reward = theta.T.dot(context)
                
                # Calculate confidence bound
                cb = self.alpha * np.sqrt(context.T.dot(self.A_inv[arm]).dot(context))
                
                # UCB = expected reward + confidence bound
                ucb_values[arm] = expected_reward + cb
            
            # Return arm with highest UCB
            return np.argmax(ucb_values)
        except Exception as e:
            logger.error(f"Error selecting arm: {e}")
            return np.random.randint(0, self.num_arms)
    
    def update(self, arm, context, reward):
        """Update model based on observed reward"""
        try:
            context = np.array(context).reshape(-1, 1)  # Ensure column vector
            
            # Update statistics
            self.A_inv[arm] = self.A_inv[arm] - (self.A_inv[arm].dot(context).dot(context.T).dot(self.A_inv[arm])) / (1 + context.T.dot(self.A_inv[arm]).dot(context))
            self.b[arm] = self.b[arm] + context * reward
            
            # Update weights
            self.theta[arm] = self.A_inv[arm].dot(self.b[arm]).flatten()
            
            # Track reward
            self.rewards[arm].append(reward)
            
            return True
        except Exception as e:
            logger.error(f"Error updating bandit: {e}")
            return False
    
    def get_arm_stats(self):
        """Get statistics for each arm"""
        stats = []
        
        for arm in range(self.num_arms):
            rewards = self.rewards[arm]
            stats.append({
                'arm': arm,
                'num_pulls': len(rewards),
                'avg_reward': np.mean(rewards) if rewards else 0,
                'confidence': 1.96 * (np.std(rewards) / np.sqrt(len(rewards))) if len(rewards) > 1 else float('inf')
            })
        
        return stats 