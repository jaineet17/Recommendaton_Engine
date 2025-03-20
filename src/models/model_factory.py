"""
Model factory for the Amazon recommendation system.

This module provides a factory for creating and managing recommendation models.
It simplifies the instantiation of different model types and handles loading and
saving models.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import pickle
from datetime import datetime

import pandas as pd

from src.data.database import load_config
from src.models.base_model import RecommendationModel

# Set up logging
logger = logging.getLogger(__name__)

# Load configuration
config = load_config()
model_config = config.get('model', {})


class ModelFactory:
    """
    Factory for creating and managing recommendation models.
    
    This class provides methods to instantiate different recommendation models and
    to load pre-trained models from disk.
    """
    
    def __init__(self):
        """Initialize model factory."""
        self.models = {}
        self.default_model = model_config.get('default_model', 'hybrid')
        self.model_paths = {}
    
    def create_model(self, model_type: str, **kwargs) -> RecommendationModel:
        """
        Create a recommendation model of the specified type.
        
        Args:
            model_type: Type of model to create
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Instantiated recommendation model
            
        Raises:
            ValueError: If model_type is not recognized
        """
        # Import models here to avoid circular imports
        try:
            if model_type.lower() == 'ncf' or model_type.lower() == 'neural_collaborative_filtering':
                from src.models.neural_collaborative_filtering import NeuralCollaborativeFiltering
                
                # Get default configuration for NCF
                ncf_config = model_config.get('neural_collaborative_filtering', {})
                
                # Update with any provided kwargs
                model_args = {**ncf_config, **kwargs}
                
                # Create the model
                model = NeuralCollaborativeFiltering(**model_args)
                logger.info(f"Created Neural Collaborative Filtering model: {model.name}")
                
                return model
            
            elif model_type.lower() == 'lightgcn':
                from src.models.lightgcn import LightGCN
                
                # Get default configuration for LightGCN
                lightgcn_config = model_config.get('lightgcn', {})
                
                # Update with any provided kwargs
                model_args = {**lightgcn_config, **kwargs}
                
                # Create the model
                model = LightGCN(**model_args)
                logger.info(f"Created LightGCN model: {model.name}")
                
                return model
            
            elif model_type.lower() == 'simple_mf' or model_type.lower() == 'matrix_factorization':
                from src.models.base_model import MatrixFactorizationBase
                
                # Default config for simple MF
                simple_mf_config = model_config.get('simple_mf', {
                    'embedding_dim': 50,
                    'max_iter': 20
                })
                
                # Update with any provided kwargs
                model_args = {**simple_mf_config, **kwargs}
                
                # Create the model
                model = MatrixFactorizationBase(
                    name="simple_mf",
                    version="1.0.0"
                )
                
                # Add additional properties specific to this model type
                model.embedding_dim = model_args.get('embedding_dim', 50)
                model.max_iter = model_args.get('max_iter', 20)
                
                logger.info(f"Created Simple Matrix Factorization model: {model.name}")
                
                return model
                
            elif model_type.lower() == 'content_based':
                from src.models.base_model import MatrixFactorizationBase
                
                # Default config for content-based model
                content_config = model_config.get('content_based', {
                    'n_components': 30
                })
                
                # Update with any provided kwargs
                model_args = {**content_config, **kwargs}
                
                # Create the model
                model = MatrixFactorizationBase(
                    name="content_based",
                    version="1.0.0"
                )
                
                # Add additional properties specific to this model type
                model.n_components = model_args.get('n_components', 30)
                
                logger.info(f"Created Content-Based model: {model.name}")
                
                return model
            
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        
        except ImportError as e:
            logger.error(f"Failed to import model: {e}")
            raise
    
    def load_model(self, model_path: str) -> RecommendationModel:
        """
        Load a recommendation model from disk.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded recommendation model
            
        Raises:
            FileNotFoundError: If model file does not exist
            ValueError: If model type cannot be determined
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Try to determine the model type from the path or file name
        model_type = self._infer_model_type(model_path)
        
        try:
            if model_type == 'ncf':
                from src.models.neural_collaborative_filtering import NeuralCollaborativeFiltering
                model = NeuralCollaborativeFiltering()
                model.load(model_path)
                logger.info(f"Loaded Neural Collaborative Filtering model from {model_path}")
                
            elif model_type == 'lightgcn':
                from src.models.lightgcn import LightGCN
                model = LightGCN()
                model.load(model_path)
                logger.info(f"Loaded LightGCN model from {model_path}")
            
            elif model_type == 'simple_mf':
                # For simple models without a class, we can use a base model
                from src.models.base_model import MatrixFactorizationBase
                model = MatrixFactorizationBase(name="simple_mf", version="1.0.0")
                
                # Load the pickle file directly
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                # Set model attributes from the saved data
                model.user_map = model_data.get('user_map', {})
                model.product_map = model_data.get('product_map', {})
                model.user_factors = model_data.get('user_factors')
                model.item_factors = model_data.get('item_factors')
                model.metadata = {
                    'trained': True,
                    'training_date': model_data.get('training_date', datetime.now()),
                    'name': model_data.get('name', 'simple_mf'),
                    'version': model_data.get('version', '1.0.0')
                }
                model.hyperparameters = model_data.get('hyperparameters', {})
                model.metrics = model_data.get('metrics', {})
                
                # Create reverse maps for prediction
                model.reverse_user_map = {idx: uid for uid, idx in model.user_map.items()}
                model.reverse_product_map = {idx: pid for pid, idx in model.product_map.items()}
                
                logger.info(f"Loaded Simple Matrix Factorization model from {model_path}")
            
            elif model_type == 'content_based':
                # For content-based models, we can also use a base model
                from src.models.base_model import MatrixFactorizationBase
                model = MatrixFactorizationBase(name="content_based", version="1.0.0")
                
                # Load the pickle file directly
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                # Set model attributes from the saved data
                model.user_map = model_data.get('user_map', {})
                model.product_map = model_data.get('product_map', {})
                model.user_factors = model_data.get('user_factors')
                model.item_factors = model_data.get('item_factors')
                model.metadata = {
                    'trained': True,
                    'training_date': model_data.get('training_date', datetime.now()),
                    'name': model_data.get('name', 'content_based'),
                    'version': model_data.get('version', '1.0.0')
                }
                model.hyperparameters = model_data.get('hyperparameters', {})
                model.metrics = model_data.get('metrics', {})
                
                # Create reverse maps for prediction
                model.reverse_user_map = {idx: uid for uid, idx in model.user_map.items()}
                model.reverse_product_map = {idx: pid for pid, idx in model.product_map.items()}
                
                logger.info(f"Loaded Content-Based model from {model_path}")
            
            else:
                raise ValueError(f"Could not determine model type for: {model_path}")
            
            return model
        
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _infer_model_type(self, model_path: str) -> str:
        """
        Infer model type from the model path.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Inferred model type
        """
        # Extract filename from path
        filename = os.path.basename(model_path).lower()
        
        # Check for model type in filename
        if 'ncf' in filename or 'neural' in filename:
            return 'ncf'
        elif 'lightgcn' in filename or 'gcn' in filename:
            return 'lightgcn'
        elif 'simple_mf' in filename:
            return 'simple_mf'
        elif 'content_based' in filename:
            return 'content_based'
        
        # If we can't determine from filename, try to check the parent directory
        parent_dir = os.path.basename(os.path.dirname(model_path)).lower()
        if 'ncf' in parent_dir or 'neural' in parent_dir:
            return 'ncf'
        elif 'lightgcn' in parent_dir or 'gcn' in parent_dir:
            return 'lightgcn'
        elif 'simple_mf' in parent_dir or 'matrix' in parent_dir:
            return 'simple_mf'
        elif 'content' in parent_dir or 'content_based' in parent_dir:
            return 'content_based'
        
        # Default to generic model type
        return 'unknown'
    
    def register_model(self, name: str, model: RecommendationModel) -> None:
        """
        Register a model instance with a name.
        
        Args:
            name: Name to register the model under
            model: Model instance to register
        """
        self.models[name] = model
        logger.info(f"Registered model: {name}")
    
    def get_model(self, name: Optional[str] = None) -> RecommendationModel:
        """
        Get a registered model by name.
        
        Args:
            name: Name of the registered model (None for default)
            
        Returns:
            Registered model instance
            
        Raises:
            KeyError: If model is not registered
        """
        if name is None:
            name = self.default_model
        
        if name not in self.models:
            raise KeyError(f"Model not registered: {name}")
        
        return self.models[name]
    
    def register_model_path(self, name: str, path: str) -> None:
        """
        Register a model path for later loading.
        
        Args:
            name: Name to register the path under
            path: Path to the model file
        """
        self.model_paths[name] = path
        logger.info(f"Registered model path: {name} -> {path}")
    
    def load_registered_model(self, name: Optional[str] = None) -> RecommendationModel:
        """
        Load a model from a registered path.
        
        Args:
            name: Name of the registered path (None for default)
            
        Returns:
            Loaded model instance
            
        Raises:
            KeyError: If path is not registered
        """
        if name is None:
            name = self.default_model
        
        if name not in self.model_paths:
            raise KeyError(f"Model path not registered: {name}")
        
        path = self.model_paths[name]
        model = self.load_model(path)
        self.register_model(name, model)
        
        return model
    
    def get_available_models(self) -> Dict[str, str]:
        """
        Get information about available models.
        
        Returns:
            Dictionary mapping model names to their info
        """
        available_models = {}
        
        # Add registered models
        for name, model in self.models.items():
            available_models[name] = f"{model.name} (v{model.version})"
        
        # Add registered paths
        for name, path in self.model_paths.items():
            if name not in available_models:
                available_models[name] = f"Not loaded (path: {path})"
        
        return available_models
    
    def save_model(self, model: RecommendationModel, path: str) -> str:
        """
        Save a model to disk.
        
        Args:
            model: Model to save
            path: Path to save the model to
            
        Returns:
            Path to the saved model
        """
        saved_path = model.save(path)
        logger.info(f"Saved model to {saved_path}")
        return saved_path
    
    def train_model(self, model_type: str, train_data: pd.DataFrame, 
                   validation_data: Optional[pd.DataFrame] = None, 
                   name: Optional[str] = None, save_path: Optional[str] = None, 
                   **kwargs) -> Tuple[RecommendationModel, Dict[str, float]]:
        """
        Create, train, and optionally save a model.
        
        Args:
            model_type: Type of model to create
            train_data: Training data
            validation_data: Validation data (optional)
            name: Name to register the model under (optional)
            save_path: Path to save the model to (optional)
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Tuple of (trained model, training metrics)
        """
        # Create the model
        model = self.create_model(model_type, **kwargs)
        
        # Train the model
        logger.info(f"Training {model_type} model...")
        metrics = model.train(train_data, validation_data)
        
        # Register the model if a name is provided
        if name is not None:
            self.register_model(name, model)
        
        # Save the model if a path is provided
        if save_path is not None:
            self.save_model(model, save_path)
            
            # Register the path if a name is provided
            if name is not None:
                self.register_model_path(name, save_path)
        
        return model, metrics


# Create a singleton instance
model_factory = ModelFactory()


# Convenience functions
def create_model(model_type: str, **kwargs) -> RecommendationModel:
    """
    Create a recommendation model of the specified type.
    
    Args:
        model_type: Type of model to create
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Instantiated recommendation model
    """
    return model_factory.create_model(model_type, **kwargs)


def load_model(model_path: str) -> RecommendationModel:
    """
    Load a recommendation model from disk.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded recommendation model
    """
    return model_factory.load_model(model_path)


def get_model(name: Optional[str] = None) -> RecommendationModel:
    """
    Get a registered model by name.
    
    Args:
        name: Name of the registered model (None for default)
        
    Returns:
        Registered model instance
    """
    return model_factory.get_model(name)


def train_model(model_type: str, train_data: pd.DataFrame, 
               validation_data: Optional[pd.DataFrame] = None, 
               name: Optional[str] = None, save_path: Optional[str] = None, 
               **kwargs) -> Tuple[RecommendationModel, Dict[str, float]]:
    """
    Create, train, and optionally save a model.
    
    Args:
        model_type: Type of model to create
        train_data: Training data
        validation_data: Validation data (optional)
        name: Name to register the model under (optional)
        save_path: Path to save the model to (optional)
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Tuple of (trained model, training metrics)
    """
    return model_factory.train_model(model_type, train_data, validation_data, name, save_path, **kwargs)


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Display available models
    logger.info("Available models:")
    for model_name, model_info in model_factory.get_available_models().items():
        logger.info(f"  - {model_name}: {model_info}")
    
    # Example: Create and register models
    try:
        # Create and register NCF model
        ncf_model = create_model('ncf', name="NCF_Small", embedding_dim=32, mlp_layers=[64, 32, 16])
        model_factory.register_model('ncf_small', ncf_model)
        
        # Create and register LightGCN model
        lightgcn_model = create_model('lightgcn', name="LightGCN_Small", embedding_dim=32, num_layers=2)
        model_factory.register_model('lightgcn_small', lightgcn_model)
        
        # Display updated available models
        logger.info("Updated available models:")
        for model_name, model_info in model_factory.get_available_models().items():
            logger.info(f"  - {model_name}: {model_info}")
    
    except ImportError as e:
        logger.warning(f"Could not create models: {e}")
        logger.warning("This is expected if running without all dependencies installed.") 