"""
Logging configuration for the recommendation engine.

This module provides a standardized logging setup to ensure consistent logging
across all components of the system.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path

# Import JSON logger for structured logging
try:
    from pythonjsonlogger import jsonlogger
    JSON_LOGGER_AVAILABLE = True
except ImportError:
    JSON_LOGGER_AVAILABLE = False


def configure_logging(
    logger_name: str = "recommendation_engine",
    log_level: str = "INFO",
    log_to_console: bool = True,
    log_to_file: bool = True,
    log_dir: str = "logs",
    use_json_format: bool = False,
    max_file_size_mb: int = 10,
    backup_count: int = 5,
) -> logging.Logger:
    """
    Configure logging for the application.
    
    Args:
        logger_name: Name of the logger
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file
        log_dir: Directory for log files
        use_json_format: Whether to use JSON format for logs
        max_file_size_mb: Maximum log file size in MB
        backup_count: Number of backup log files to keep
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(logger_name)
    
    # Set log level
    log_level_dict = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    logger.setLevel(log_level_dict.get(log_level.upper(), logging.INFO))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatters
    if use_json_format and JSON_LOGGER_AVAILABLE:
        # JSON formatter for structured logging
        formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s %(funcName)s %(lineno)d',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        # Standard formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Add console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler
    if log_to_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True, parents=True)
        
        # Create file name with date
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = log_path / f"{logger_name}_{timestamp}.log"
        
        # Use rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.info(f"Logging configured for {logger_name}")
    return logger


def get_logger(
    module_name: str, 
    parent_logger: str = "recommendation_engine"
) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        module_name: Name of the module
        parent_logger: Name of the parent logger
        
    Returns:
        Logger instance
    """
    logger_name = f"{parent_logger}.{module_name}"
    return logging.getLogger(logger_name)


# Example of usage in other modules:
# 
# from src.utils.logging_config import get_logger
# 
# logger = get_logger("models.lightgcn")
# 
# logger.debug("Debug message")
# logger.info("Info message")
# logger.warning("Warning message")
# logger.error("Error message")
# logger.critical("Critical message")
