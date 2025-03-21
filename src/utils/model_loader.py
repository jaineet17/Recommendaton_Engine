"""
Utility module for loading models from the reorganized model directory structure.
"""

import os
import json
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Class for managing model loading from the organized model directory."""
    
    def __init__(self, models_dir='data/models'):
        """
        Initialize the model registry.
        
        Args:
            models_dir: Path to the models directory
        """
        self.models_dir = Path(models_dir)
        self.registry_path = self.models_dir / 'registry.json'
        self.registry = self._load_registry()
        self.loaded_models = {}
        
    def _load_registry(self):
        """Load the model registry file."""
        try:
            with open(self.registry_path, 'r') as f:
                registry = json.load(f)
                logger.info(f"Loaded model registry with {len(registry['models'])} model types")
                return registry
        except Exception as e:
            logger.error(f"Failed to load model registry: {e}")
            return {
                "models": {},
                "model_aliases": {}
            }
    
    def get_model_path(self, model_name, version=None):
        """
        Get the full path to a model file.
        
        Args:
            model_name: Name of the model or alias
            version: Specific version to load, or None for production version
            
        Returns:
            Path to the model file
        """
        # Resolve alias if necessary
        model_key = self.registry.get("model_aliases", {}).get(model_name, model_name)
        
        # Check if model exists in registry
        if model_key not in self.registry.get("models", {}):
            raise ValueError(f"Model {model_name} not found in registry")
        
        model_info = self.registry["models"][model_key]
        model_dir = model_info["path"]
        
        # Determine version to load
        if version is None:
            version = model_info["production_version"]
        elif version not in model_info["available_versions"]:
            raise ValueError(f"Version {version} not available for model {model_name}")
        
        # Load metadata to get filename
        metadata_path = self.models_dir / model_dir / 'metadata.json'
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            # Find the version information
            version_info = None
            for v in metadata.get("versions", []):
                if v["version"] == version:
                    version_info = v
                    break
                    
            if not version_info:
                raise ValueError(f"Version {version} metadata not found for model {model_name}")
                
            filename = version_info["filename"]
            model_path = self.models_dir / model_dir / filename
            return model_path
            
        except Exception as e:
            logger.error(f"Failed to load model metadata: {e}")
            raise
    
    def load_model(self, model_name, version=None, force_reload=False):
        """
        Load a model from the registry.
        
        Args:
            model_name: Name of the model or alias
            version: Specific version to load, or None for production version
            force_reload: Whether to reload the model if already loaded
            
        Returns:
            The loaded model
        """
        # Create a cache key for this model+version
        cache_key = f"{model_name}_{version}" if version else f"{model_name}_production"
        
        # Return cached model if available and not force_reload
        if cache_key in self.loaded_models and not force_reload:
            logger.debug(f"Using cached model for {cache_key}")
            return self.loaded_models[cache_key]
        
        try:
            model_path = self.get_model_path(model_name, version)
            
            logger.info(f"Loading model {model_name} (version: {version or 'production'}) from {model_path}")
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                
            self.loaded_models[cache_key] = model
            logger.info(f"Successfully loaded model {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def get_available_models(self):
        """Return a list of available models with their versions."""
        return self.registry.get("models", {})
    
    def get_model_aliases(self):
        """Return a dictionary of model aliases."""
        return self.registry.get("model_aliases", {})


# Singleton instance for easy access
_registry_instance = None

def get_model_registry(models_dir='data/models'):
    """Get or create the singleton model registry instance."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry(models_dir)
    return _registry_instance

def load_model(model_name, version=None, force_reload=False):
    """
    Convenience function to load a model from the registry.
    
    Args:
        model_name: Name of the model or alias
        version: Specific version to load, or None for production version
        force_reload: Whether to reload the model if already loaded
        
    Returns:
        The loaded model
    """
    registry = get_model_registry()
    return registry.load_model(model_name, version, force_reload) 