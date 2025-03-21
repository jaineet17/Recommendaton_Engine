"""
Core utilities for the recommendation engine.

This module consolidates common functionality used across the recommendation engine:
- File operations (pickle, JSON, CSV)
- Math utilities (vector operations, similarity measures)
- Evaluation metrics (precision, recall, NDCG)
- Timing and performance monitoring

It provides a unified interface for common operations to maintain consistency.
"""

import json
import logging
import os
import pickle
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

# Get module logger
logger = get_logger("utils.utility")

# File operations
def load_pickle(filepath: str) -> Any:
    """
    Load a pickle file.
    
    Args:
        filepath: Path to the pickle file
        
    Returns:
        The unpickled object
        
    Raises:
        FileNotFoundError: If the file does not exist
        Exception: If there's an error unpickling the file
    """
    try:
        filepath = os.path.expanduser(filepath)
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading pickle file {filepath}: {e}")
        raise

def save_pickle(obj: Any, filepath: str) -> None:
    """
    Save an object to a pickle file.
    
    Args:
        obj: Object to save
        filepath: Path to save the pickle file
        
    Raises:
        Exception: If there's an error pickling the object
    """
    try:
        filepath = os.path.expanduser(filepath)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
        logger.debug(f"Successfully saved pickle file: {filepath}")
    except Exception as e:
        logger.error(f"Error saving pickle file {filepath}: {e}")
        raise

def load_json(filepath: str) -> Any:
    """
    Load a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        The parsed JSON object
        
    Raises:
        FileNotFoundError: If the file does not exist
        Exception: If there's an error parsing the JSON file
    """
    try:
        filepath = os.path.expanduser(filepath)
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading JSON file {filepath}: {e}")
        raise

def save_json(obj: Any, filepath: str, indent: int = 4) -> None:
    """
    Save an object to a JSON file.
    
    Args:
        obj: Object to save (must be JSON serializable)
        filepath: Path to save the JSON file
        indent: Number of spaces for indentation (default: 4)
        
    Raises:
        Exception: If there's an error serializing to JSON
    """
    try:
        filepath = os.path.expanduser(filepath)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(obj, f, indent=indent)
        logger.debug(f"Successfully saved JSON file: {filepath}")
    except Exception as e:
        logger.error(f"Error saving JSON file {filepath}: {e}")
        raise

# Vector operations
def normalize_vector(v: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.
    
    Args:
        v: Vector to normalize
        
    Returns:
        Normalized vector
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity (-1 to 1)
    """
    a_norm = normalize_vector(a)
    b_norm = normalize_vector(b)
    return np.dot(a_norm, b_norm)

# Recommendation metrics
def calculate_precision_at_k(recommended_items: List[str], 
                            relevant_items: List[str], 
                            k: int = 10) -> float:
    """
    Calculate precision@k for recommendation evaluation.
    
    Args:
        recommended_items: List of recommended item IDs
        relevant_items: List of relevant (true positive) item IDs
        k: The k value for precision@k
        
    Returns:
        Precision@k score (0 to 1)
    """
    if not recommended_items or not relevant_items:
        return 0.0
    
    # Convert to sets for faster lookup
    recommended_set = set(recommended_items[:k])
    relevant_set = set(relevant_items)
    
    # Calculate hits
    hits = len(recommended_set.intersection(relevant_set))
    
    # Calculate precision@k
    return hits / min(k, len(recommended_items))

def calculate_recall_at_k(recommended_items: List[str], 
                         relevant_items: List[str], 
                         k: int = 10) -> float:
    """
    Calculate recall@k for recommendation evaluation.
    
    Args:
        recommended_items: List of recommended item IDs
        relevant_items: List of relevant (true positive) item IDs
        k: The k value for recall@k
        
    Returns:
        Recall@k score (0 to 1)
    """
    if not recommended_items or not relevant_items:
        return 0.0
    
    # Convert to sets for faster lookup
    recommended_set = set(recommended_items[:k])
    relevant_set = set(relevant_items)
    
    # Calculate hits
    hits = len(recommended_set.intersection(relevant_set))
    
    # Calculate recall@k
    return hits / len(relevant_set) if relevant_set else 0.0

def calculate_ndcg_at_k(recommended_items: List[str], 
                       relevant_items: List[str], 
                       k: int = 10) -> float:
    """
    Calculate NDCG@k (Normalized Discounted Cumulative Gain) for recommendation evaluation.
    
    Args:
        recommended_items: List of recommended item IDs
        relevant_items: List of relevant (true positive) item IDs
        k: The k value for NDCG@k
        
    Returns:
        NDCG@k score (0 to 1)
    """
    if not recommended_items or not relevant_items:
        return 0.0
    
    # Convert relevant items to a set for faster lookup
    relevant_set = set(relevant_items)
    
    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(recommended_items[:k]):
        if item in relevant_set:
            # Using log base 2 for DCG calculation
            dcg += 1.0 / np.log2(i + 2)  # +2 because i is 0-indexed
    
    # Calculate ideal DCG (IDCG)
    idcg = 0.0
    for i in range(min(len(relevant_set), k)):
        idcg += 1.0 / np.log2(i + 2)
    
    # Calculate NDCG
    return dcg / idcg if idcg > 0 else 0.0

def get_top_n_indices(scores: np.ndarray, n: int) -> List[int]:
    """
    Get indices of top-n scores.
    
    Args:
        scores: Array of scores
        n: Number of top scores to return
        
    Returns:
        List of indices corresponding to top-n scores
    """
    # Get indices sorted by scores in descending order
    return np.argsort(scores)[::-1][:n].tolist()

# Timing and logging utilities
class Timer:
    """Context manager for timing operations and optionally logging the duration."""
    
    def __init__(self, name: str = "Operation", logger: Optional[logging.Logger] = None, 
                 log_level: int = logging.DEBUG):
        """
        Initialize a Timer.
        
        Args:
            name: Name of the operation being timed
            logger: Optional logger to log the timing result
            log_level: Logging level to use (default: DEBUG)
        """
        self.name = name
        self.logger = logger
        self.log_level = log_level
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        """Start the timer when entering the context."""
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Compute and optionally log the duration when exiting the context."""
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        if self.logger:
            self.logger.log(self.log_level, f"{self.name} completed in {duration:.4f} seconds")
            
        # Store the duration for later use
        self.duration = duration
        
    def get_duration(self) -> float:
        """Get the duration of the timed operation in seconds."""
        if self.start_time is None:
            return 0.0
        if self.end_time is None:
            # If timer is still running, get current duration
            return time.time() - self.start_time
        return self.end_time - self.start_time

def log_exception(logger: logging.Logger, msg: str, exc: Exception) -> None:
    """
    Log an exception with additional context information.
    
    Args:
        logger: The logger to use
        msg: Message describing the context of the exception
        exc: The exception object
    """
    logger.error(f"{msg}: {exc}")
    logger.debug(f"Exception details:", exc_info=True)
    
def create_audit_log(action: str, user_id: str, details: Dict[str, Any], 
                    logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Create a structured audit log entry.
    
    Args:
        action: The action being performed (e.g., "get_recommendations")
        user_id: ID of the user performing the action
        details: Additional details about the action
        logger: Optional logger to log the audit entry
        
    Returns:
        Dict containing the structured audit log entry
    """
    timestamp = datetime.now().isoformat()
    
    audit_entry = {
        "timestamp": timestamp,
        "action": action,
        "user_id": user_id,
        "details": details
    }
    
    if logger:
        logger.info(f"AUDIT: {action} by {user_id} at {timestamp} - {json.dumps(details)}")
        
    return audit_entry
