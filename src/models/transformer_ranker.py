import os
import torch
import torch.nn as nn
import numpy as np
import logging
from datetime import datetime
import json
from transformers import AutoModel, AutoTokenizer
import onnxruntime as ort
from src.kafka.consumer import BaseConsumer
from src.kafka.producer import get_producer
from src.utils.config.mlflow_config import MLFlowConfig
from src.caching.recommendation_cache import RecommendationCache
from src.models.onnx_utils import load_onnx_model

logger = logging.getLogger(__name__)

class TransformerRanker(BaseConsumer):
    """Ranks candidates using transformer-based models"""
    
    def __init__(self, model_name="product_transformer", onnx_enabled=True):
        # Initialize Kafka consumer
        super().__init__(topics=["candidate_ranking"])
        
        self.model_name = model_name
        self.onnx_enabled = onnx_enabled
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize MLFlow
        self.mlflow = MLFlowConfig()
        self.mlflow.initialize()
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model()
        
        # Load ONNX model if enabled
        self.onnx_model = None
        self.onnx_session = None
        if self.onnx_enabled:
            self.onnx_model = self._load_onnx_model()
        
        # Initialize feature store client for additional features
        # self.feature_store = FeatureStore()
        
        # Initialize recommendation cache
        self.cache = RecommendationCache()
        
        # Initialize Kafka producer for sending ranked results to ensemble
        self.ensemble_producer = get_producer("ranked_candidates")
    
    def _load_model(self):
        """Load transformer model and tokenizer"""
        try:
            # Try to load from MLFlow first
            model = self.mlflow.load_model(self.model_name)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                os.environ.get("TRANSFORMER_TOKENIZER", "bert-base-uncased")
            )
            
            logger.info(f"Loaded {self.model_name} model from MLFlow")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading model from MLFlow: {e}")
            
            # Fallback to local model
            try:
                model_path = os.environ.get("TRANSFORMER_MODEL_PATH", "models/transformer")
                model = AutoModel.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                
                logger.info(f"Loaded {self.model_name} model from local path")
                return model, tokenizer
            except Exception as e:
                logger.error(f"Error loading local model: {e}")
                return None, None
    
    def _load_onnx_model(self):
        """Load ONNX model for faster inference"""
        try:
            onnx_path = os.environ.get("ONNX_MODEL_PATH", f"models/{self.model_name}.onnx")
            
            # Create ONNX runtime session
            session_options = ort.SessionOptions()
            
            # Set graph optimization level
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Set execution providers
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
            
            # Create session
            session = ort.InferenceSession(onnx_path, session_options, providers=providers)
            
            logger.info(f"Loaded ONNX model from {onnx_path}")
            return session
        except Exception as e:
            logger.error(f"Error loading ONNX model: {e}")
            return None
    
    def process_message(self, message):
        """Process incoming candidate ranking message from Kafka"""
        try:
            user_id = message.get('user_id')
            candidates = message.get('candidates', [])
            context = message.get('context')
            
            if not user_id or not candidates:
                logger.warning(f"Missing data in message: user_id={user_id}, candidates={len(candidates) if candidates else 0}")
                return
            
            # Rank candidates
            ranked_items = self.rank_candidates(user_id, candidates, context)
            
            # Send ranked results to ensemble
            self._send_to_ensemble(user_id, ranked_items, context)
            
        except Exception as e:
            logger.error(f"Error processing ranking message: {e}")
    
    def _prepare_inputs(self, user_id, candidates, context=None):
        """Prepare model inputs for the transformer model"""
        try:
            # Get item details
            item_ids = [item['item_id'] for item in candidates]
            
            # Get item descriptions or features
            item_texts = self._get_item_texts(item_ids)
            
            # Get user profile or history
            user_text = self._get_user_profile(user_id)
            
            # Add context if available
            if context:
                context_text = self._format_context(context)
                user_text = f"{user_text} {context_text}"
            
            # Prepare inputs for each item
            model_inputs = []
            for i, item_text in enumerate(item_texts):
                # Combine user and item text
                input_text = f"User: {user_text} Item: {item_text}"
                
                # Tokenize input
                encoded_input = self.tokenizer(
                    input_text,
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                
                # Add to batch
                model_inputs.append({
                    'input_ids': encoded_input['input_ids'],
                    'attention_mask': encoded_input['attention_mask'],
                    'token_type_ids': encoded_input.get('token_type_ids', None),
                    'item_id': item_ids[i],
                    'original_score': candidates[i].get('score', 0.0)
                })
            
            return model_inputs
        except Exception as e:
            logger.error(f"Error preparing inputs: {e}")
            return []
    
    def _get_item_texts(self, item_ids):
        """Get item descriptions or features"""
        # Implementation to retrieve item details
        # This could be from a database, feature store, etc.
        return [f"Product {item_id}" for item_id in item_ids]
    
    def _get_user_profile(self, user_id):
        """Get user profile or history"""
        # Implementation to retrieve user profile
        return f"User {user_id}"
    
    def _format_context(self, context):
        """Format context information as text"""
        if not context:
            return ""
        
        context_str = " ".join([f"{k}: {v}" for k, v in context.items()])
        return f"Context: {context_str}"
    
    def _run_inference(self, model_inputs):
        """Run inference using transformer model or ONNX"""
        # Use ONNX if available
        if self.onnx_enabled and self.onnx_session:
            return self._run_onnx_inference(model_inputs)
        
        # Fallback to PyTorch
        return self._run_pytorch_inference(model_inputs)
    
    def _run_pytorch_inference(self, model_inputs):
        """Run inference using PyTorch model"""
        try:
            results = []
            
            # Process in batches
            batch_size = 16
            for i in range(0, len(model_inputs), batch_size):
                batch = model_inputs[i:i+batch_size]
                
                # Concatenate tensors
                input_ids = torch.cat([x['input_ids'] for x in batch]).to(self.device)
                attention_mask = torch.cat([x['attention_mask'] for x in batch]).to(self.device)
                
                token_type_ids = None
                if batch[0]['token_type_ids'] is not None:
                    token_type_ids = torch.cat([x['token_type_ids'] for x in batch]).to(self.device)
                
                # Run inference
                with torch.no_grad():
                    if token_type_ids is not None:
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids
                        )
                    else:
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                
                # Extract relevance scores
                if hasattr(outputs, 'logits'):
                    scores = outputs.logits.cpu().numpy()
                else:
                    # Use the [CLS] representation
                    scores = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    # Compute a relevance score (simplified)
                    scores = np.mean(scores, axis=1)
                
                # Combine results
                for j, score in enumerate(scores):
                    item_index = i + j
                    if item_index < len(model_inputs):
                        results.append({
                            'item_id': model_inputs[item_index]['item_id'],
                            'score': float(score if isinstance(score, (int, float)) else score[0]),
                            'original_score': model_inputs[item_index]['original_score'],
                            'stage': 'ranking',
                            'model': self.model_name
                        })
            
            return results
        except Exception as e:
            logger.error(f"Error during PyTorch inference: {e}")
            return []
    
    def _run_onnx_inference(self, model_inputs):
        """Run inference using ONNX runtime"""
        try:
            results = []
            
            # Process in batches
            batch_size = 16
            for i in range(0, len(model_inputs), batch_size):
                batch = model_inputs[i:i+batch_size]
                
                # Prepare ONNX inputs
                onnx_inputs = {
                    'input_ids': np.concatenate([x['input_ids'].numpy() for x in batch]),
                    'attention_mask': np.concatenate([x['attention_mask'].numpy() for x in batch])
                }
                
                if batch[0]['token_type_ids'] is not None:
                    onnx_inputs['token_type_ids'] = np.concatenate([x['token_type_ids'].numpy() for x in batch])
                
                # Run ONNX inference
                onnx_outputs = self.onnx_session.run(None, onnx_inputs)
                
                # Extract scores from output
                scores = onnx_outputs[0]  # Assuming first output is the score
                
                # Combine results
                for j, score in enumerate(scores):
                    item_index = i + j
                    if item_index < len(model_inputs):
                        results.append({
                            'item_id': model_inputs[item_index]['item_id'],
                            'score': float(score if isinstance(score, (int, float)) else score[0]),
                            'original_score': model_inputs[item_index]['original_score'],
                            'stage': 'ranking',
                            'model': f"{self.model_name}_onnx"
                        })
            
            return results
        except Exception as e:
            logger.error(f"Error during ONNX inference: {e}")
            return []
    
    def rank_candidates(self, user_id, candidates, context=None):
        """Rank candidates using the transformer model"""
        try:
            # Check cache first
            cache_key = f"{self.model_name}_ranked"
            cached_results = self.cache.get_recommendations(
                user_id=user_id, 
                model_name=cache_key,
                context=context,
                limit=len(candidates)
            )
            
            if cached_results:
                logger.debug(f"Using cached ranking for user {user_id}")
                return cached_results['recommendations']
            
            # Prepare inputs
            model_inputs = self._prepare_inputs(user_id, candidates, context)
            
            # Run inference
            results = self._run_inference(model_inputs)
            
            # Sort by score
            ranked_items = sorted(results, key=lambda x: x['score'], reverse=True)
            
            # Cache results
            self.cache.cache_recommendations(
                user_id=user_id,
                model_name=cache_key,
                recommendations=ranked_items,
                context=context,
                limit=len(candidates)
            )
            
            return ranked_items
        except Exception as e:
            logger.error(f"Error ranking candidates: {e}")
            # Fall back to original ordering
            return candidates
    
    def _send_to_ensemble(self, user_id, ranked_items, context=None):
        """Send ranked items to ensemble service via Kafka"""
        try:
            message = {
                'user_id': user_id,
                'ranked_items': ranked_items,
                'context': context,
                'timestamp': datetime.now().isoformat(),
                'source_model': self.model_name
            }
            
            self.ensemble_producer.send_message(message)
            logger.debug(f"Sent {len(ranked_items)} ranked items to ensemble for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error sending ranked items to ensemble: {e}")
            return False 