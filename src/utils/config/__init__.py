"""
Configuration module for the Amazon Recommendation Engine.

This module provides access to the centralized configuration system.
"""

from .config import (
    load_config, 
    validate_config, 
    get_connection_string, 
    get_redis_url,
    DEFAULT_CONFIG
)

from .mlflow_config import MLFlowConfig

import os
import logging
from typing import Dict, Any, Optional

# Set up logging
logger = logging.getLogger(__name__)

# Global configuration singleton
_CONFIG = None

def get_config(config_path: Optional[str] = None, force_reload: bool = False) -> Dict[str, Any]:
    """
    Get the configuration singleton, loading it if not already loaded.
    
    Args:
        config_path: Optional path to the configuration file
        force_reload: Whether to force a reload even if config is already loaded
        
    Returns:
        Dict containing the complete configuration
    """
    global _CONFIG
    
    if _CONFIG is None or force_reload:
        environment = os.environ.get("ENVIRONMENT")
        try:
            _CONFIG = load_config(config_path, environment)
            if not validate_config(_CONFIG):
                logger.warning("Configuration validation failed, some features may not work correctly")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            # Fall back to default configuration
            _CONFIG = DEFAULT_CONFIG.copy()
            _CONFIG["_error"] = str(e)
    
    return _CONFIG

# For backwards compatibility - these allow old code to continue working
def get_api_config():
    """Get API configuration section for backwards compatibility."""
    return get_config().get("api", {})

def get_database_config():
    """Get database configuration section for backwards compatibility."""
    return get_config().get("database", {})

def get_model_config():
    """Get model configuration section for backwards compatibility."""
    return get_config().get("models", {})

def get_kafka_config():
    """Get Kafka configuration section for backwards compatibility."""
    return get_config().get("kafka", {})

def get_feedback_loop_config():
    """Get feedback loop configuration section for backwards compatibility."""
    return get_config().get("feedback_loop", {})

# Alias for the get_config function for backwards compatibility
CONFIG = get_config

__all__ = [
    'load_config', 'validate_config', 'get_connection_string', 'get_redis_url',
    'DEFAULT_CONFIG', 'get_config', 'CONFIG', 'MLFlowConfig', 'get_api_config', 
    'get_database_config', 'get_model_config', 'get_kafka_config', 'get_feedback_loop_config'
] 