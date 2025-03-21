"""
Unified Configuration System for Amazon Recommendation Engine

This module provides a centralized configuration system that:
1. Loads configuration from YAML files
2. Supports environment variable overrides
3. Validates configuration values
4. Provides defaults for missing values
5. Enables different environments (development, staging, production)

Configuration is loaded in the following order of precedence:
1. Environment variables (highest priority)
2. Environment-specific config file (e.g., config/production.yaml)
3. Local configuration override (config/local_config.yaml)
4. Base configuration file (config/config.yaml)
5. Default values (lowest priority)
"""

import os
import re
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

# Set up module logger
logger = logging.getLogger(__name__)

# Configuration paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_DIR = ROOT_DIR / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "config.yaml"
LOCAL_CONFIG_PATH = CONFIG_DIR / "local_config.yaml"

# Environment variable prefix for config overrides
ENV_PREFIX = "RECOMMENDER_"

# Default configuration values
DEFAULT_CONFIG = {
    "environment": "development",
    "api": {
        "host": "0.0.0.0",
        "port": 5050,
        "threads": 4,
        "debug": False,
        "request_timeout": 30,
        "max_content_length": 16 * 1024 * 1024,  # 16MB
    },
    "database": {
        "host": "localhost",
        "port": 5432,
        "user": "postgres",
        "password": "postgres",
        "dbname": "recommender",
        "pool_size": 10,
        "max_overflow": 20,
        "pool_timeout": 30,
    },
    "redis": {
        "host": "localhost",
        "port": 6379,
        "db": 0,
        "password": None,
        "socket_timeout": 5,
    },
    "kafka": {
        "bootstrap_servers": "localhost:9092",
        "enable": True,
        "topics": {
            "product_events": "product-events",
            "recommendations": "recommendations",
            "metrics": "metrics",
        },
        "consumer_group": "recommender-api",
        "auto_offset_reset": "earliest",
    },
    "models": {
        "path": str(ROOT_DIR / "data" / "models"),
        "cache_size": 10000,
        "default_model": "lightgcn",
        "max_recommendations": 100,
        "default_recommendations": 10,
        "score_threshold": 0.01,
        "cold_start_strategy": "popular",
        "update_interval": 3600,  # 1 hour
        "preload_models": ["lightgcn", "ncf", "mf"],
    },
    "logging": {
        "level": "INFO",
        "file": str(ROOT_DIR / "logs" / "api.log"),
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "max_size": 10 * 1024 * 1024,  # 10MB
        "backup_count": 5,
    },
    "metrics": {
        "enable": True,
        "port": 8001,
        "path": "/metrics",
        "collection_interval": 15,
    },
    "feedback_loop": {
        "enable": True,
        "window_size": 48,  # hours
        "minimum_events": 10,
        "update_interval": 3600,  # seconds
    },
}


def _parse_value(value: str) -> Any:
    """Convert string value to appropriate Python type."""
    # Handle boolean values
    if value.lower() in ["true", "yes", "1", "on"]:
        return True
    if value.lower() in ["false", "no", "0", "off"]:
        return False
    
    # Handle None/null values
    if value.lower() in ["none", "null"]:
        return None
    
    # Handle numeric values
    try:
        # Try to convert to int first
        return int(value)
    except ValueError:
        try:
            # Then try float
            return float(value)
        except ValueError:
            # Otherwise, keep as string
            return value


def _deep_update(base_dict: Dict, update_dict: Dict) -> Dict:
    """Recursively update a nested dictionary with another dictionary."""
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            # Recursively update nested dictionaries
            base_dict[key] = _deep_update(base_dict[key], value)
        else:
            # Update or add simple values
            base_dict[key] = value
    return base_dict


def _load_yaml_config(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            logger.debug(f"Config file {file_path} not found")
            return {}
            
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f) or {}
            logger.debug(f"Loaded configuration from {file_path}")
            return config
    except Exception as e:
        logger.warning(f"Error loading config from {file_path}: {str(e)}")
        return {}


def _get_env_overrides() -> Dict[str, Any]:
    """Get configuration overrides from environment variables."""
    result = {}
    pattern = re.compile(f"^{ENV_PREFIX}(.+)$")
    
    for env_var, value in os.environ.items():
        match = pattern.match(env_var)
        if not match:
            continue
            
        # Convert environment variable path to nested dictionary
        # e.g., RECOMMENDER_API_PORT -> {"api": {"port": value}}
        path = match.group(1).lower().split('_')
        current = result
        
        # Build nested dictionary structure
        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Set the value at the end of the path
        current[path[-1]] = _parse_value(value)
    
    return result


def load_config(config_path: Optional[Union[str, Path]] = None, 
                environment: Optional[str] = None) -> Dict[str, Any]:
    """
    Load the configuration from files and environment variables.
    
    Args:
        config_path: Path to the base configuration file, defaults to config/config.yaml
        environment: The environment to use (development, staging, production)
                     If None, will be read from ENVIRONMENT env var or default to development
    
    Returns:
        Dict containing the complete configuration
    """
    # Start with default configuration
    config = DEFAULT_CONFIG.copy()
    
    # Load base configuration file
    base_config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    base_config = _load_yaml_config(base_config_path)
    config = _deep_update(config, base_config)
    
    # Determine environment
    if environment is None:
        environment = os.environ.get("ENVIRONMENT", config.get("environment", "development"))
    
    # Load environment-specific configuration
    env_config_path = base_config_path.parent / f"{environment}.yaml"
    env_config = _load_yaml_config(env_config_path)
    config = _deep_update(config, env_config)
    
    # Load local configuration overrides
    local_config = _load_yaml_config(LOCAL_CONFIG_PATH)
    config = _deep_update(config, local_config)
    
    # Apply environment variable overrides
    env_overrides = _get_env_overrides()
    config = _deep_update(config, env_overrides)
    
    # Ensure environment is set in config
    config["environment"] = environment
    
    logger.info(f"Configuration loaded for environment: {environment}")
    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate the configuration to ensure all required values are present and valid.
    
    Args:
        config: The configuration dictionary to validate
        
    Returns:
        bool: True if the configuration is valid, False otherwise
    """
    # Validate database connection settings
    if "database" in config:
        db_config = config["database"]
        required_db_fields = ["host", "port", "user", "dbname"]
        for field in required_db_fields:
            if field not in db_config or not db_config[field]:
                logger.error(f"Missing required database configuration: {field}")
                return False
    
    # Validate API settings
    if "api" in config:
        api_config = config["api"]
        if "port" in api_config and not (1024 <= api_config["port"] <= 65535):
            logger.error(f"Invalid API port: {api_config['port']}. Must be between 1024 and 65535.")
            return False
    
    # Validate model settings
    if "models" in config:
        model_config = config["models"]
        if "path" in model_config:
            model_path = Path(model_config["path"])
            if not model_path.exists():
                logger.warning(f"Model path does not exist: {model_path}. Will be created if needed.")
    
    return True


def get_connection_string(config: Dict[str, Any]) -> str:
    """
    Generate a SQLAlchemy database connection string from configuration.
    
    Args:
        config: The configuration dictionary
        
    Returns:
        str: The SQLAlchemy connection string
    """
    if "database" not in config:
        raise ValueError("Database configuration not found")
        
    db = config["database"]
    user = db.get("user", "postgres")
    password = db.get("password", "postgres")
    host = db.get("host", "localhost")
    port = db.get("port", 5432)
    dbname = db.get("dbname", "recommender")
    
    # Build connection string
    return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"


def get_redis_url(config: Dict[str, Any]) -> str:
    """
    Generate a Redis URL from configuration.
    
    Args:
        config: The configuration dictionary
        
    Returns:
        str: The Redis URL
    """
    if "redis" not in config:
        raise ValueError("Redis configuration not found")
        
    redis = config["redis"]
    host = redis.get("host", "localhost")
    port = redis.get("port", 6379)
    db = redis.get("db", 0)
    password = redis.get("password")
    
    # Build Redis URL
    if password:
        return f"redis://:{password}@{host}:{port}/{db}"
    else:
        return f"redis://{host}:{port}/{db}" 