"""Configuration module for the recommendation engine."""

import os
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Get project root directory
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
CONFIG_DIR = ROOT_DIR / "config"

def load_config(config_path=None):
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file, relative to project root.
            Defaults to config/config.yaml
            
    Returns:
        dict: Configuration dictionary
    """
    if config_path is None:
        config_path = os.environ.get("CONFIG_PATH", "config/config.yaml")
    
    # Convert to absolute path if needed
    if not os.path.isabs(config_path):
        config_path = os.path.join(ROOT_DIR, config_path)
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        # Return empty dict as fallback
        return {}

def load_kafka_config(config_path=None):
    """Load Kafka configuration from YAML file.
    
    Args:
        config_path: Path to Kafka config file, relative to project root.
            Defaults to config/kafka_config.yaml
            
    Returns:
        dict: Kafka configuration dictionary
    """
    if config_path is None:
        config_path = os.environ.get("KAFKA_CONFIG_PATH", "config/kafka_config.yaml")
    
    # Convert to absolute path if needed
    if not os.path.isabs(config_path):
        config_path = os.path.join(ROOT_DIR, config_path)
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded Kafka configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load Kafka configuration from {config_path}: {e}")
        # Return empty dict as fallback
        return {}

# Default configuration (will be loaded when module is imported)
CONFIG = load_config()
KAFKA_CONFIG = load_kafka_config() 