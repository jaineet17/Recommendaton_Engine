import os
import numpy as np
import logging
import json
import pickle
from datetime import datetime
from collections import defaultdict
import scipy.stats as stats
from src.kafka.consumer import BaseConsumer
from src.kafka.producer import get_producer
from src.caching.recommendation_cache import RecommendationCache
from src.feedback.collector import FeedbackCollector

logger = logging.getLogger(__name__)

class HybridEnsemble(BaseConsumer):
    """Combines predictions from multiple models using hybrid approaches"""
    
    def __init__(self):
        # Initialize Kafka consumer
        super().__init__(topics=["ranked_candidates"])
        
        # Initialize weights for different models
        self.model_weights = self._load_model_weights()
        
        # User-specific adaptation
        self.user_weights = defaultdict(lambda: self.model_weights.copy())
        
        # Recent feedback
        self.recent_feedback = {}
        
        # Initialize recommendation cache
        self.cache = RecommendationCache()
        
        # Initialize Kafka producer for sending final results
        self.final_producer = get_producer("final_recommendations")
        
        # Initialize feedback metrics for weight updates
        self.feedback_metrics = defaultdict(lambda: defaultdict(list))
        
        # Load stacking model if available
        self.stacking_model = self._load_stacking_model()
    
    def _load_model_weights(self):
        """Load model weights from configuration"""
        try:
            # Load from config or environment
            weights_file = os.environ.get("MODEL_WEIGHTS_PATH", "config/model_weights.json")
            
            if os.path.exists(weights_file):
                with open(weights_file, 'r') as f:
                    weights = json.load(f)
                logger.info(f"Loaded model weights from {weights_file}")
                return weights
            else:
                # Default weights
                logger.warning(f"Model weights file not found, using defaults")
                return {
                    "lightgcn": 0.5,
                    "product_transformer": 0.8,
                    "product_transformer_onnx": 0.8,
                    "user_history": 0.3,
                    "popularity": 0.1,
                    "diversity": 0.2
                }
        except Exception as e:
            logger.error(f"Error loading model weights: {e}")
            # Fallback default weights
            return {
                "lightgcn": 0.5,
                "product_transformer": 0.8,
                "product_transformer_onnx": 0.8,
                "user_history": 0.3,
                "popularity": 0.1,
                "diversity": 0.2
            }
    
    def _load_stacking_model(self):
        """Load stacking model"""
        try:
            model_path = os.environ.get("STACKING_MODEL_PATH", "models/stacking.pkl")
            
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                logger.info(f"Loaded stacking model from {model_path}")
                return model
            else:
                logger.warning("Stacking model not found, will use weighted average")
                return None
        except Exception as e:
            logger.error(f"Error loading stacking model: {e}")
            return None
    
    def save_model_weights(self):
        """Save model weights to file"""
        try:
            weights_file = os.environ.get("MODEL_WEIGHTS_PATH", "config/model_weights.json")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(weights_file), exist_ok=True)
            
            with open(weights_file, 'w') as f:
                json.dump(self.model_weights, f, indent=2)
            
            logger.info(f"Saved model weights to {weights_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving model weights: {e}")
            return False
    
    def process_message(self, message):
        """Process incoming ranking message from Kafka"""
        try:
            user_id = message.get('user_id')
            ranked_items = message.get('ranked_items', [])
            context = message.get('context')
            source_model = message.get('source_model')
            
            if not user_id or not ranked_items:
                logger.warning(f"Missing data in message: user_id={user_id}, items={len(ranked_items) if ranked_items else 0}")
                return
            
            # Store recent model predictions keyed by user_id
            if user_id not in self.recent_feedback:
                self.recent_feedback[user_id] = {}
            
            # Store ranked items from this model
            self.recent_feedback[user_id][source_model] = {
                'ranked_items': ranked_items,
                'timestamp': datetime.now().isoformat(),
                'context': context
            }
            
            # If we have enough recommendations to ensemble, proceed
            if len(self.recent_feedback[user_id]) >= 1:
                # Create ensemble recommendations
                final_recommendations = self.create_ensemble(user_id, context)
                
                # Send to final recommendations topic
                self._send_final_recommendations(user_id, final_recommendations, context)
                
                # Clean up predictions older than 1 hour
                self._cleanup_old_predictions()
            
        except Exception as e:
            logger.error(f"Error processing ranking message: {e}")
    
    def _get_all_ranked_items(self, user_id):
        """Get all ranked items for a user from different models"""
        if user_id not in self.recent_feedback:
            return {}
        
        all_items = {}
        for model, data in self.recent_feedback[user_id].items():
            all_items[model] = data['ranked_items']
        
        return all_items
    
    def create_ensemble(self, user_id, context=None):
        """Create ensemble recommendations using hybrid approach"""
        try:
            # Check cache first
            cached_results = self.cache.get_recommendations(
                user_id=user_id, 
                model_name="hybrid_ensemble",
                context=context
            )
            
            if cached_results:
                logger.debug(f"Using cached ensemble for user {user_id}")
                return cached_results['recommendations']
            
            # Get all ranked items from different models
            all_ranked_items = self._get_all_ranked_items(user_id)
            
            if not all_ranked_items:
                logger.warning(f"No ranked items available for user {user_id}")
                return []
            
            # Use stacking if available, otherwise weighted average
            if self.stacking_model:
                final_recommendations = self._apply_stacking(all_ranked_items, user_id)
            else:
                final_recommendations = self._apply_weighted_average(all_ranked_items, user_id)
            
            # Apply diversity optimization
            final_recommendations = self._optimize_diversity(final_recommendations)
            
            # Cache results
            self.cache.cache_recommendations(
                user_id=user_id,
                model_name="hybrid_ensemble",
                recommendations=final_recommendations,
                context=context
            )
            
            return final_recommendations
        except Exception as e:
            logger.error(f"Error creating ensemble: {e}")
            # Fall back to first available model's predictions
            for model, items in all_ranked_items.items():
                return items
            return []
    
    def _apply_weighted_average(self, all_ranked_items, user_id):
        """Apply weighted average ensemble"""
        try:
            # Get user-specific weights or default weights
            weights = self.user_weights[user_id]
            
            # Collect all unique items
            all_items = set()
            for model, items in all_ranked_items.items():
                for item in items:
                    all_items.add(item['item_id'])
            
            # Calculate ensemble score for each item
            item_scores = defaultdict(float)
            item_count = defaultdict(int)
            item_details = {}
            
            for model, items in all_ranked_items.items():
                model_weight = weights.get(model, 0.5)  # Default weight if not found
                
                for rank, item in enumerate(items):
                    item_id = item['item_id']
                    normalized_score = item['score'] * model_weight
                    
                    # Rank-based boost (higher ranks get higher weight)
                    rank_weight = 1.0 / (rank + 1)
                    final_score = normalized_score * rank_weight
                    
                    item_scores[item_id] += final_score
                    item_count[item_id] += 1
                    
                    # Keep track of item details for final output
                    if item_id not in item_details:
                        item_details[item_id] = item.copy()
            
            # Normalize by count
            for item_id in item_scores:
                if item_count[item_id] > 0:
                    item_scores[item_id] /= item_count[item_id]
            
            # Create final sorted recommendations
            final_recommendations = []
            
            for item_id, score in sorted(item_scores.items(), key=lambda x: x[1], reverse=True):
                if item_id in item_details:
                    item = item_details[item_id].copy()
                    item['score'] = float(score)
                    item['stage'] = 'ensemble'
                    item['ensemble_type'] = 'weighted_average'
                    final_recommendations.append(item)
            
            return final_recommendations
        except Exception as e:
            logger.error(f"Error applying weighted average: {e}")
            return []
    
    def _apply_stacking(self, all_ranked_items, user_id):
        """Apply stacking ensemble"""
        try:
            # Implementation for stacking model ensemble
            # This would use the pre-trained stacking_model to combine predictions
            # For now, fallback to weighted average
            return self._apply_weighted_average(all_ranked_items, user_id)
        except Exception as e:
            logger.error(f"Error applying stacking: {e}")
            return self._apply_weighted_average(all_ranked_items, user_id)
    
    def _optimize_diversity(self, recommendations, diversity_weight=0.2):
        """Optimize diversity of recommendations"""
        try:
            if not recommendations or len(recommendations) <= 1:
                return recommendations
            
            # Simple diversity optimization: penalize similar items that are close in the list
            # For a full implementation, you would need item metadata/embeddings
            
            # For this example, assume we're diversifying by some feature like category
            # In a real implementation, use actual item features
            result = [recommendations[0]]
            
            for i in range(1, len(recommendations)):
                # Add diversity penalty based on position
                recommendations[i]['score'] = recommendations[i]['score'] * (1 - diversity_weight * (1.0 / (i + 1)))
                result.append(recommendations[i])
            
            # Re-sort after diversity adjustment
            return sorted(result, key=lambda x: x['score'], reverse=True)
        except Exception as e:
            logger.error(f"Error optimizing diversity: {e}")
            return recommendations
    
    def process_feedback(self, user_id, item_id, event_type, value=1.0):
        """Process user feedback to update model weights"""
        try:
            # Get the model that recommended this item
            recommending_models = []
            
            if user_id in self.recent_feedback:
                for model, data in self.recent_feedback[user_id].items():
                    for item in data['ranked_items']:
                        if item['item_id'] == item_id:
                            recommending_models.append((model, item['score']))
            
            if not recommending_models:
                logger.warning(f"No model found for feedback on item {item_id} from user {user_id}")
                return
            
            # Update feedback metrics
            for model, score in recommending_models:
                if event_type == 'click':
                    self.feedback_metrics[model]['clicks'].append((user_id, item_id, score, value))
                elif event_type == 'purchase':
                    self.feedback_metrics[model]['purchases'].append((user_id, item_id, score, value))
                elif event_type == 'rating':
                    self.feedback_metrics[model]['ratings'].append((user_id, item_id, score, value))
                elif event_type == 'view_time':
                    self.feedback_metrics[model]['view_times'].append((user_id, item_id, score, value))
            
            # Update weights based on feedback
            self._update_weights()
            
            # Update user-specific weights
            self._update_user_weights(user_id, recommending_models, event_type, value)
            
            logger.debug(f"Processed feedback for user {user_id}, item {item_id}, event {event_type}")
            return True
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
            return False
    
    def _update_weights(self):
        """Update global model weights based on all collected feedback"""
        try:
            # Only update if we have enough feedback
            min_feedback = int(os.environ.get("MIN_FEEDBACK_FOR_UPDATE", 100))
            
            for model in self.feedback_metrics:
                total_feedback = sum(len(data) for data in self.feedback_metrics[model].values())
                
                if total_feedback < min_feedback:
                    continue
                
                # Calculate model success metrics
                click_rate = len(self.feedback_metrics[model]['clicks']) / total_feedback if total_feedback > 0 else 0
                purchase_rate = len(self.feedback_metrics[model]['purchases']) / total_feedback if total_feedback > 0 else 0
                avg_rating = np.mean([r[3] for r in self.feedback_metrics[model]['ratings']]) if self.feedback_metrics[model]['ratings'] else 0
                
                # Combined success metric
                success_metric = (click_rate * 0.3) + (purchase_rate * 0.5) + (avg_rating / 5.0 * 0.2)
                
                # Update weight with some inertia (0.8 old weight, 0.2 new information)
                old_weight = self.model_weights.get(model, 0.5)
                new_weight = (old_weight * 0.8) + (success_metric * 0.2)
                
                # Ensure weight stays in reasonable range
                new_weight = max(0.1, min(1.0, new_weight))
                
                # Update model weight
                self.model_weights[model] = new_weight
            
            # Save updated weights
            self.save_model_weights()
            
            logger.info(f"Updated model weights: {self.model_weights}")
            return True
        except Exception as e:
            logger.error(f"Error updating weights: {e}")
            return False
    
    def _update_user_weights(self, user_id, recommending_models, event_type, value):
        """Update weights specifically for this user"""
        try:
            # Update user-specific model weights based on feedback
            if event_type in ['click', 'purchase', 'rating']:
                for model, score in recommending_models:
                    # Get current weight
                    current_weight = self.user_weights[user_id].get(model, self.model_weights.get(model, 0.5))
                    
                    # Determine adjustment factor based on event type
                    adjust_factor = 0.0
                    if event_type == 'click':
                        adjust_factor = 0.01  # Small increase for clicks
                    elif event_type == 'purchase':
                        adjust_factor = 0.03  # Larger increase for purchases
                    elif event_type == 'rating':
                        # For ratings, adjust based on rating value (assuming 1-5 scale)
                        if isinstance(value, (int, float)):
                            normalized_rating = (float(value) - 3) / 2  # Map 1-5 to -1 to 1
                            adjust_factor = normalized_rating * 0.02
                    
                    # Apply adjustment
                    new_weight = current_weight + adjust_factor
                    
                    # Ensure weight stays in reasonable range
                    new_weight = max(0.1, min(1.0, new_weight))
                    
                    # Update user-specific weight
                    self.user_weights[user_id][model] = new_weight
            
            return True
        except Exception as e:
            logger.error(f"Error updating user weights: {e}")
            return False
    
    def _send_final_recommendations(self, user_id, recommendations, context=None):
        """Send final recommendations to Kafka"""
        try:
            message = {
                'user_id': user_id,
                'recommendations': recommendations,
                'context': context,
                'timestamp': datetime.now().isoformat(),
                'model': 'hybrid_ensemble'
            }
            
            self.final_producer.send_message(message)
            logger.debug(f"Sent {len(recommendations)} final recommendations for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error sending final recommendations: {e}")
            return False
    
    def _cleanup_old_predictions(self, max_age_hours=1):
        """Clean up old predictions"""
        try:
            current_time = datetime.now()
            users_to_remove = []
            
            for user_id, models in self.recent_feedback.items():
                models_to_remove = []
                
                for model, data in models.items():
                    timestamp = datetime.fromisoformat(data['timestamp'])
                    age = (current_time - timestamp).total_seconds() / 3600
                    
                    if age > max_age_hours:
                        models_to_remove.append(model)
                
                for model in models_to_remove:
                    del self.recent_feedback[user_id][model]
                
                if not self.recent_feedback[user_id]:
                    users_to_remove.append(user_id)
            
            for user_id in users_to_remove:
                del self.recent_feedback[user_id]
            
            return True
        except Exception as e:
            logger.error(f"Error cleaning up old predictions: {e}")
            return False 