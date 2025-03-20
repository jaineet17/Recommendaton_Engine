"""
Base classes for recommendation models in the Amazon recommendation system.
"""

import logging
import os
import pickle
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.data.database import load_config

# Set up logging
logger = logging.getLogger(__name__)

# Load configuration
config = load_config()
model_config = config.get('model', {})
data_config = config.get('data', {})


class RecommendationModel(ABC):
    """
    Abstract base class for recommendation models.
    
    All recommendation models should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, name: str, version: str = "0.1.0"):
        """
        Initialize recommendation model.
        
        Args:
            name: Name of the model
            version: Version of the model
        """
        self.name = name
        self.version = version
        self.metadata = {
            'name': name,
            'version': version,
            'created_at': datetime.now().isoformat(),
            'trained': False
        }
        self.hyperparameters = {}
        self.metrics = {}
    
    @abstractmethod
    def train(self, train_data: Any, validation_data: Optional[Any] = None) -> Dict[str, float]:
        """
        Train the recommendation model.
        
        Args:
            train_data: Training data
            validation_data: Validation data (optional)
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def predict(self, user_id: str, n: int = 10) -> List[Tuple[str, float]]:
        """
        Generate recommendations for a user.
        
        Args:
            user_id: ID of the user
            n: Number of recommendations to generate
            
        Returns:
            List of tuples containing (product_id, score)
        """
        pass
    
    @abstractmethod
    def batch_predict(self, user_ids: List[str], n: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """
        Generate recommendations for multiple users.
        
        Args:
            user_ids: List of user IDs
            n: Number of recommendations to generate per user
            
        Returns:
            Dictionary mapping user IDs to lists of (product_id, score) tuples
        """
        pass
    
    @abstractmethod
    def evaluate(self, test_data: Any) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Test data
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass
    
    def save(self, path: str) -> str:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model to
            
        Returns:
            Path to the saved model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Update metadata
        self.metadata['saved_at'] = datetime.now().isoformat()
        
        # Prepare data to save
        data = {
            'metadata': self.metadata,
            'hyperparameters': self.hyperparameters,
            'metrics': self.metrics,
            'model': self._get_model_data()
        }
        
        # Save to disk
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Model saved to {path}")
        return path
    
    def load(self, path: str) -> 'RecommendationModel':
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded model
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        # Load metadata and metrics
        self.metadata = data['metadata']
        self.hyperparameters = data['hyperparameters']
        self.metrics = data['metrics']
        
        # Load model-specific data
        self._set_model_data(data['model'])
        
        logger.info(f"Model loaded from {path}")
        return self
    
    @abstractmethod
    def _get_model_data(self) -> Any:
        """
        Get model-specific data for saving.
        
        Returns:
            Model data
        """
        pass
    
    @abstractmethod
    def _set_model_data(self, data: Any) -> None:
        """
        Set model-specific data after loading.
        
        Args:
            data: Model data
        """
        pass
    
    def set_hyperparameters(self, **kwargs) -> None:
        """
        Set model hyperparameters.
        
        Args:
            **kwargs: Hyperparameters to set
        """
        self.hyperparameters.update(kwargs)
        logger.info(f"Hyperparameters set: {kwargs}")
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get model metadata.
        
        Returns:
            Dictionary of metadata
        """
        return {
            **self.metadata,
            'hyperparameters': self.hyperparameters,
            'metrics': self.metrics
        }


class MatrixFactorizationBase(RecommendationModel):
    """
    Base class for matrix factorization recommendation models.
    
    Provides common functionality for collaborative filtering models.
    """
    
    def __init__(self, name: str, version: str = "0.1.0"):
        """
        Initialize matrix factorization model.
        
        Args:
            name: Name of the model
            version: Version of the model
        """
        super().__init__(name, version)
        
        # Internal data structures
        self.user_map = {}  # Mapping from user_id to internal index
        self.product_map = {}  # Mapping from product_id to internal index
        self.reverse_user_map = {}  # Mapping from internal index to user_id
        self.reverse_product_map = {}  # Mapping from internal index to product_id
        
        # Model parameters (to be set by subclasses)
        self.user_factors = None
        self.item_factors = None
    
    def _create_id_mappings(self, ratings_df: pd.DataFrame) -> None:
        """
        Create mappings between external IDs and internal indices.
        
        Args:
            ratings_df: DataFrame with user_id and product_id columns
        """
        # Create user ID mapping
        unique_users = ratings_df['user_id'].unique()
        self.user_map = {user_id: i for i, user_id in enumerate(unique_users)}
        self.reverse_user_map = {i: user_id for user_id, i in self.user_map.items()}
        
        # Create product ID mapping
        unique_products = ratings_df['product_id'].unique()
        self.product_map = {product_id: i for i, product_id in enumerate(unique_products)}
        self.reverse_product_map = {i: product_id for product_id, i in self.product_map.items()}
        
        logger.info(f"Created ID mappings for {len(self.user_map)} users and {len(self.product_map)} products")
    
    def _map_user_id(self, user_id: str) -> Optional[int]:
        """
        Map external user ID to internal index.
        
        Args:
            user_id: External user ID
            
        Returns:
            Internal user index or None if not found
        """
        return self.user_map.get(user_id)
    
    def _map_product_id(self, product_id: str) -> Optional[int]:
        """
        Map external product ID to internal index.
        
        Args:
            product_id: External product ID
            
        Returns:
            Internal product index or None if not found
        """
        return self.product_map.get(product_id)
    
    def _get_user_vector(self, user_idx: int) -> np.ndarray:
        """
        Get the latent factor vector for a user.
        
        Args:
            user_idx: Internal user index
            
        Returns:
            User latent factor vector
        """
        if self.user_factors is None:
            raise RuntimeError("Model has not been trained yet")
        
        return self.user_factors[user_idx]
    
    def _get_item_vector(self, item_idx: int) -> np.ndarray:
        """
        Get the latent factor vector for a product.
        
        Args:
            item_idx: Internal product index
            
        Returns:
            Product latent factor vector
        """
        if self.item_factors is None:
            raise RuntimeError("Model has not been trained yet")
        
        return self.item_factors[item_idx]
    
    def _predict_score(self, user_idx: int, item_idx: int) -> float:
        """
        Predict the rating for a user-item pair.
        
        Args:
            user_idx: Internal user index
            item_idx: Internal product index
            
        Returns:
            Predicted rating
        """
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("Model has not been trained yet")
        
        user_vector = self._get_user_vector(user_idx)
        item_vector = self._get_item_vector(item_idx)
        
        return float(np.dot(user_vector, item_vector))
    
    def predict(self, user_id: str, n: int = 10) -> List[Tuple[str, float]]:
        """
        Generate recommendations for a user.
        
        Args:
            user_id: ID of the user
            n: Number of recommendations to generate
            
        Returns:
            List of tuples containing (product_id, score)
        """
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("Model has not been trained yet")
        
        user_idx = self._map_user_id(user_id)
        if user_idx is None:
            logger.warning(f"User {user_id} not found in model")
            return []
        
        # Calculate scores for all items
        user_vector = self._get_user_vector(user_idx)
        scores = np.dot(self.item_factors, user_vector)
        
        # Get top N items
        top_indices = np.argsort(scores)[::-1][:n]
        
        # Map back to product IDs and scores
        recommendations = [
            (self.reverse_product_map[idx], float(scores[idx]))
            for idx in top_indices
        ]
        
        return recommendations
    
    def batch_predict(self, user_ids: List[str], n: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """
        Generate recommendations for multiple users.
        
        Args:
            user_ids: List of user IDs
            n: Number of recommendations to generate per user
            
        Returns:
            Dictionary mapping user IDs to lists of (product_id, score) tuples
        """
        results = {}
        
        for user_id in user_ids:
            results[user_id] = self.predict(user_id, n)
        
        return results
    
    def _get_model_data(self) -> Dict[str, Any]:
        """
        Get model-specific data for saving.
        
        Returns:
            Dictionary containing model data
        """
        return {
            'user_map': self.user_map,
            'product_map': self.product_map,
            'reverse_user_map': self.reverse_user_map,
            'reverse_product_map': self.reverse_product_map,
            'user_factors': self.user_factors,
            'item_factors': self.item_factors
        }
    
    def _set_model_data(self, data: Dict[str, Any]) -> None:
        """
        Set model-specific data after loading.
        
        Args:
            data: Dictionary containing model data
        """
        self.user_map = data['user_map']
        self.product_map = data['product_map']
        self.reverse_user_map = data['reverse_user_map']
        self.reverse_product_map = data['reverse_product_map']
        self.user_factors = data['user_factors']
        self.item_factors = data['item_factors']


class ContentBasedBase(RecommendationModel):
    """
    Base class for content-based recommendation models.
    
    Provides common functionality for content-based filtering models.
    """
    
    def __init__(self, name: str, version: str = "0.1.0"):
        """
        Initialize content-based model.
        
        Args:
            name: Name of the model
            version: Version of the model
        """
        super().__init__(name, version)
        
        # Product features
        self.product_features = None
        self.product_ids = None
        
        # User profiles
        self.user_profiles = {}
    
    def _build_product_features(self, products_df: pd.DataFrame) -> None:
        """
        Build product feature matrix.
        
        Args:
            products_df: DataFrame with product features
        """
        # To be implemented by subclasses
        pass
    
    def _build_user_profiles(self, ratings_df: pd.DataFrame) -> None:
        """
        Build user profiles based on their rated items.
        
        Args:
            ratings_df: DataFrame with user-item interactions
        """
        # To be implemented by subclasses
        pass
    
    def predict(self, user_id: str, n: int = 10) -> List[Tuple[str, float]]:
        """
        Generate recommendations for a user.
        
        Args:
            user_id: ID of the user
            n: Number of recommendations to generate
            
        Returns:
            List of tuples containing (product_id, score)
        """
        if user_id not in self.user_profiles:
            logger.warning(f"User {user_id} not found in model")
            return []
        
        user_profile = self.user_profiles[user_id]
        
        # Calculate similarity scores
        scores = self._calculate_similarity(user_profile)
        
        # Get top N items
        top_indices = np.argsort(scores)[::-1][:n]
        
        # Map back to product IDs and scores
        recommendations = [
            (self.product_ids[idx], float(scores[idx]))
            for idx in top_indices
        ]
        
        return recommendations
    
    def _calculate_similarity(self, user_profile: np.ndarray) -> np.ndarray:
        """
        Calculate similarity between user profile and all products.
        
        Args:
            user_profile: User profile vector
            
        Returns:
            Array of similarity scores
        """
        # To be implemented by subclasses
        raise NotImplementedError
    
    def batch_predict(self, user_ids: List[str], n: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """
        Generate recommendations for multiple users.
        
        Args:
            user_ids: List of user IDs
            n: Number of recommendations to generate per user
            
        Returns:
            Dictionary mapping user IDs to lists of (product_id, score) tuples
        """
        results = {}
        
        for user_id in user_ids:
            results[user_id] = self.predict(user_id, n)
        
        return results
    
    def _get_model_data(self) -> Dict[str, Any]:
        """
        Get model-specific data for saving.
        
        Returns:
            Dictionary containing model data
        """
        return {
            'product_features': self.product_features,
            'product_ids': self.product_ids,
            'user_profiles': self.user_profiles
        }
    
    def _set_model_data(self, data: Dict[str, Any]) -> None:
        """
        Set model-specific data after loading.
        
        Args:
            data: Dictionary containing model data
        """
        self.product_features = data['product_features']
        self.product_ids = data['product_ids']
        self.user_profiles = data['user_profiles']


class HybridBase(RecommendationModel):
    """
    Base class for hybrid recommendation models.
    
    Provides common functionality for hybrid recommendation models that combine
    collaborative filtering and content-based approaches.
    """
    
    def __init__(self, name: str, collab_model: RecommendationModel, content_model: RecommendationModel, 
                collab_weight: float = 0.7, version: str = "0.1.0"):
        """
        Initialize hybrid model.
        
        Args:
            name: Name of the model
            collab_model: Collaborative filtering model
            content_model: Content-based model
            collab_weight: Weight for collaborative filtering recommendations (0-1)
            version: Version of the model
        """
        super().__init__(name, version)
        
        self.collab_model = collab_model
        self.content_model = content_model
        self.collab_weight = collab_weight
        self.content_weight = 1.0 - collab_weight
        
        # Update hyperparameters
        self.hyperparameters.update({
            'collab_weight': collab_weight,
            'content_weight': self.content_weight,
            'collab_model': collab_model.name,
            'content_model': content_model.name
        })
    
    def train(self, train_data: Any, validation_data: Optional[Any] = None) -> Dict[str, float]:
        """
        Train the hybrid model.
        
        Args:
            train_data: Training data
            validation_data: Validation data (optional)
            
        Returns:
            Dictionary of training metrics
        """
        # Train component models
        collab_metrics = self.collab_model.train(train_data, validation_data)
        content_metrics = self.content_model.train(train_data, validation_data)
        
        # Combine metrics
        metrics = {
            'collab_' + k: v for k, v in collab_metrics.items()
        }
        metrics.update({
            'content_' + k: v for k, v in content_metrics.items()
        })
        
        self.metrics = metrics
        self.metadata['trained'] = True
        
        return metrics
    
    def predict(self, user_id: str, n: int = 10) -> List[Tuple[str, float]]:
        """
        Generate recommendations for a user using hybrid approach.
        
        Args:
            user_id: ID of the user
            n: Number of recommendations to generate
            
        Returns:
            List of tuples containing (product_id, score)
        """
        # Get recommendations from both models
        collab_recs = self.collab_model.predict(user_id, n=n*2)  # Get more to allow for overlap
        content_recs = self.content_model.predict(user_id, n=n*2)
        
        # Combine recommendations
        product_scores = {}
        
        # Add collaborative filtering recommendations
        for product_id, score in collab_recs:
            product_scores[product_id] = score * self.collab_weight
        
        # Add content-based recommendations
        for product_id, score in content_recs:
            if product_id in product_scores:
                product_scores[product_id] += score * self.content_weight
            else:
                product_scores[product_id] = score * self.content_weight
        
        # Sort by score and take top N
        sorted_recs = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        
        return sorted_recs
    
    def batch_predict(self, user_ids: List[str], n: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """
        Generate recommendations for multiple users.
        
        Args:
            user_ids: List of user IDs
            n: Number of recommendations to generate per user
            
        Returns:
            Dictionary mapping user IDs to lists of (product_id, score) tuples
        """
        results = {}
        
        for user_id in user_ids:
            results[user_id] = self.predict(user_id, n)
        
        return results
    
    def evaluate(self, test_data: Any) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Test data
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Evaluate component models
        collab_metrics = self.collab_model.evaluate(test_data)
        content_metrics = self.content_model.evaluate(test_data)
        
        # Combine metrics
        metrics = {
            'collab_' + k: v for k, v in collab_metrics.items()
        }
        metrics.update({
            'content_' + k: v for k, v in content_metrics.items()
        })
        
        # Custom evaluation for hybrid model
        # This would typically involve predicting recommendations for test users
        # and calculating metrics like precision, recall, etc.
        
        return metrics
    
    def _get_model_data(self) -> Dict[str, Any]:
        """
        Get model-specific data for saving.
        
        Returns:
            Dictionary containing model data
        """
        return {
            'collab_model': self.collab_model._get_model_data(),
            'content_model': self.content_model._get_model_data(),
            'collab_weight': self.collab_weight,
            'content_weight': self.content_weight
        }
    
    def _set_model_data(self, data: Dict[str, Any]) -> None:
        """
        Set model-specific data after loading.
        
        Args:
            data: Dictionary containing model data
        """
        self.collab_model._set_model_data(data['collab_model'])
        self.content_model._set_model_data(data['content_model'])
        self.collab_weight = data['collab_weight']
        self.content_weight = data['content_weight'] 