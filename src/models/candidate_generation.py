import os
import numpy as np
import torch
import logging
import faiss
import json
from datetime import datetime
from src.kafka.producer import get_producer
from src.utils.config.mlflow_config import MLFlowConfig
from src.caching.recommendation_cache import RecommendationCache

logger = logging.getLogger(__name__)

class CandidateGenerator:
    """Generates recommendation candidates using lightweight models like LightGCN"""
    
    def __init__(self, model_name="lightgcn", candidate_pool_size=1000, min_candidates=100):
        self.model_name = model_name
        self.candidate_pool_size = candidate_pool_size
        self.min_candidates = min_candidates
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize MLFlow
        self.mlflow = MLFlowConfig()
        self.mlflow.initialize()
        
        # Load model and embeddings
        self.model = self._load_model()
        self.user_embeddings = None
        self.item_embeddings = None
        self.item_id_mapping = None
        self.user_id_mapping = None
        
        # Initialize recommendation cache
        self.cache = RecommendationCache()
        
        # Initialize Kafka producer for sending candidates to ranking service
        self.ranking_producer = get_producer("candidate_ranking")
        
        # Initialize FAISS index for fast similarity search
        self._initialize_faiss_index()
    
    def _load_model(self):
        """Load model from MLFlow registry"""
        try:
            model = self.mlflow.load_model(self.model_name)
            logger.info(f"Loaded {self.model_name} model from MLFlow")
            
            # Load embeddings and mappings
            self._load_embeddings()
            return model
        except Exception as e:
            logger.error(f"Error loading {self.model_name} model: {e}")
            return None
    
    def _load_embeddings(self):
        """Load user and item embeddings"""
        try:
            # Load embeddings from files
            embedding_dir = os.environ.get("EMBEDDING_DIR", "data/embeddings")
            
            self.user_embeddings = np.load(f"{embedding_dir}/{self.model_name}_user_embeddings.npy")
            self.item_embeddings = np.load(f"{embedding_dir}/{self.model_name}_item_embeddings.npy")
            
            # Load mappings
            with open(f"{embedding_dir}/{self.model_name}_user_mapping.json", 'r') as f:
                self.user_id_mapping = json.load(f)
            
            with open(f"{embedding_dir}/{self.model_name}_item_mapping.json", 'r') as f:
                self.item_id_mapping = json.load(f)
                
            # Create reverse mappings
            self.user_idx_mapping = {idx: user_id for user_id, idx in self.user_id_mapping.items()}
            self.item_idx_mapping = {idx: item_id for item_id, idx in self.item_id_mapping.items()}
            
            logger.info(f"Loaded embeddings: Users={self.user_embeddings.shape}, Items={self.item_embeddings.shape}")
            return True
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return False
    
    def _initialize_faiss_index(self):
        """Initialize FAISS index for fast similarity search"""
        try:
            if self.item_embeddings is not None:
                dim = self.item_embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
                
                # Normalize vectors to make inner product equivalent to cosine similarity
                faiss.normalize_L2(self.item_embeddings)
                
                # Add items to index
                self.index.add(self.item_embeddings)
                logger.info(f"Initialized FAISS index with {self.index.ntotal} items")
                return True
            else:
                logger.warning("Item embeddings not available, FAISS index not initialized")
                return False
        except Exception as e:
            logger.error(f"Error initializing FAISS index: {e}")
            return False
    
    def get_user_embedding(self, user_id):
        """Get embedding vector for a user"""
        try:
            if self.user_id_mapping and user_id in self.user_id_mapping:
                user_idx = self.user_id_mapping[user_id]
                return self.user_embeddings[user_idx]
            else:
                logger.warning(f"User {user_id} not found in embedding mapping")
                # Return average user embedding as fallback
                return np.mean(self.user_embeddings, axis=0)
        except Exception as e:
            logger.error(f"Error getting user embedding: {e}")
            return None
    
    def fast_similarity_search(self, query_vector, k=100):
        """Perform fast similarity search using FAISS"""
        try:
            # Normalize query vector for cosine similarity
            query_vector = query_vector.reshape(1, -1)
            faiss.normalize_L2(query_vector)
            
            # Search FAISS index
            scores, indices = self.index.search(query_vector, k)
            return indices[0], scores[0]
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return [], []
    
    def select_top_candidates(self, item_indices, scores, filters=None):
        """Select and filter top candidates"""
        try:
            candidates = []
            
            for idx, score in zip(item_indices, scores):
                item_id = self.item_idx_mapping.get(int(idx))
                
                # Apply filters if provided
                if filters and not self._apply_filters(item_id, filters):
                    continue
                
                candidates.append({
                    'item_id': item_id,
                    'score': float(score),
                    'stage': 'candidate_generation',
                    'model': self.model_name
                })
            
            # Sort by score
            candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
            return candidates
        except Exception as e:
            logger.error(f"Error selecting top candidates: {e}")
            return []
    
    def _apply_filters(self, item_id, filters):
        """Apply filters to candidates"""
        # Implementation for filtering candidates
        # Examples: category, price range, etc.
        return True
    
    def generate_candidates(self, user_id, context=None, filters=None, limit=100):
        """Generate recommendation candidates for a user"""
        try:
            # Check cache first
            cached_results = self.cache.get_recommendations(
                user_id=user_id, 
                model_name=f"{self.model_name}_candidates",
                context=context,
                limit=limit
            )
            
            if cached_results:
                logger.debug(f"Using cached candidates for user {user_id}")
                return cached_results['recommendations']
            
            # Get user embedding
            user_embedding = self.get_user_embedding(user_id)
            if user_embedding is None:
                logger.error(f"Could not generate embedding for user {user_id}")
                return []
            
            # Find similar items using FAISS
            item_indices, scores = self.fast_similarity_search(
                user_embedding, 
                k=self.candidate_pool_size
            )
            
            # Select and filter candidates
            candidates = self.select_top_candidates(item_indices, scores, filters)
            
            # Ensure minimum number of candidates
            if len(candidates) < self.min_candidates:
                logger.warning(f"Not enough candidates for user {user_id}, adding popular items")
                candidates.extend(self._get_popular_items(
                    limit=self.min_candidates-len(candidates), 
                    exclude=[c['item_id'] for c in candidates]
                ))
            
            # Limit results
            candidates = candidates[:limit]
            
            # Cache results
            self.cache.cache_recommendations(
                user_id=user_id,
                model_name=f"{self.model_name}_candidates",
                recommendations=candidates,
                context=context,
                limit=limit
            )
            
            # Send to ranking service
            self._send_to_ranking(user_id, candidates, context)
            
            return candidates
        except Exception as e:
            logger.error(f"Error generating candidates: {e}")
            return []
    
    def _get_popular_items(self, limit=10, exclude=None):
        """Get popular items as fallback"""
        # Implementation for retrieving popular items
        return []
    
    def _send_to_ranking(self, user_id, candidates, context=None):
        """Send candidates to the ranking service via Kafka"""
        try:
            message = {
                'user_id': user_id,
                'candidates': candidates,
                'context': context,
                'timestamp': datetime.now().isoformat(),
                'source_model': self.model_name
            }
            
            self.ranking_producer.send_message(message)
            logger.debug(f"Sent {len(candidates)} candidates to ranking service for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error sending candidates to ranking service: {e}")
            return False 