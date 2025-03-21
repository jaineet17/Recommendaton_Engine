"""
Logging utilities for Amazon Recommendation Engine (Compatibility Module).

This module has been deprecated and consolidated into utility.py and logging_config.py.
It is maintained for backward compatibility only.
"""

import warnings
import logging
from typing import Optional, Dict, Any, Union

# Issue deprecation warning
warnings.warn(
    "This module has been deprecated. Please use src.utils.logging_config or "
    "src.utils.utility for logging functionality instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from the new locations
from src.utils.logging_config import configure_logging, get_logger
from src.utils.utility import Timer as LogTimer, log_exception, create_audit_log

# Re-export all the functions with stubs for backward compatibility

def setup_logging(
    log_level: Union[str, int] = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    console: bool = True,
    max_size: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Set up logging with the specified configuration.
    
    This function is now deprecated. Use configure_logging from src.utils.logging_config instead.
    """
    # Delegate to the new implementation
    return configure_logging(name="root", log_level=log_level, log_file=log_file, 
                           log_format=log_format, console=console)

# For backwards compatibility
def get_logger_compat(name: str, config: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    This function is now deprecated. Use get_logger from src.utils.logging_config instead.
    """
    # Delegate to the new implementation
    return get_logger(name=name)
