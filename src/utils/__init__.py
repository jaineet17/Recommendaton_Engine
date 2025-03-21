"""
Utilities package for Amazon Recommendation Engine.

This package contains various utility modules for the recommendation system:
- logging_utils: Configuring logging throughout the application
- timer: Performance measurement utilities
- serialization: Functions for serializing and deserializing data
"""

from .logging_utils import (
    setup_logging,
    get_logger,
    log_exception,
    create_audit_log,
    LogTimer,
)

__all__ = [
    'setup_logging',
    'get_logger',
    'log_exception',
    'create_audit_log',
    'LogTimer',
] 