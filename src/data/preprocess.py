"""
Module for preprocessing the Amazon product reviews dataset.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.database import load_config

# Set up logging
logger = logging.getLogger(__name__)

# Load configuration
config = load_config()
data_config = config.get('data', {})
model_config = config.get('model', {}).get('training', {})

# Define constants
DEFAULT_RAW_DATA_PATH = data_config.get('raw_data_path', 'data/raw')
DEFAULT_PROCESSED_DATA_PATH = data_config.get('processed_data_path', 'data/processed')
DEFAULT_INTERIM_DATA_PATH = data_config.get('interim_data_path', 'data/interim')
DEFAULT_SAMPLE_SIZE = data_config.get('sample_size', 100000)
DEFAULT_TEST_SIZE = model_config.get('test_size', 0.2)
DEFAULT_VALIDATION_SIZE = model_config.get('validation_size', 0.25)
DEFAULT_RANDOM_STATE = model_config.get('random_state', 42)


def read_raw_data(file_path: Union[str, Path], nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Read the raw Amazon product reviews dataset.
    
    Args:
        file_path: Path to the raw data file
        nrows: Number of rows to read (None for all)
        
    Returns:
        DataFrame with the raw data
    """
    logger.info(f"Reading raw data from {file_path}")
    
    try:
        # Detect file format and read accordingly
        if str(file_path).endswith('.csv'):
            df = pd.read_csv(file_path, nrows=nrows)
        elif str(file_path).endswith('.json'):
            df = pd.read_json(file_path, lines=True, nrows=nrows)
        elif str(file_path).endswith('.parquet'):
            df = pd.read_parquet(file_path)
            if nrows is not None:
                df = df.head(nrows)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        logger.info(f"Read {len(df)} rows")
        return df
    
    except Exception as e:
        logger.error(f"Error reading raw data: {e}")
        raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw data.
    
    Args:
        df: DataFrame with raw data
        
    Returns:
        DataFrame with cleaned data
    """
    logger.info("Cleaning data")
    
    # Make a copy to avoid modifying the original dataframe
    df_clean = df.copy()
    
    # Standardize column names
    df_clean.columns = [col.lower().replace(' ', '_') for col in df_clean.columns]
    
    # Rename columns if needed based on common Amazon dataset columns
    column_mapping = {
        'product_id': 'product_id',
        'asin': 'product_id',
        'user_id': 'user_id',
        'reviewer_id': 'user_id',
        'rating': 'rating',
        'overall': 'rating',
        'review_text': 'review_text',
        'reviewtext': 'review_text',
        'review': 'review_text',
        'summary': 'summary',
        'title': 'summary',
        'timestamp': 'review_date',
        'review_date': 'review_date',
        'unixreviewtime': 'review_date',
        'unix_review_time': 'review_date',
    }
    
    # Apply column mapping where columns exist
    for old_col, new_col in column_mapping.items():
        if old_col in df_clean.columns and old_col != new_col:
            df_clean[new_col] = df_clean[old_col]
    
    # Ensure required columns exist
    required_columns = ['product_id', 'user_id', 'rating']
    missing_columns = [col for col in required_columns if col not in df_clean.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Handle missing values
    if 'review_text' in df_clean.columns:
        df_clean['review_text'] = df_clean['review_text'].fillna('')
    else:
        df_clean['review_text'] = ''
    
    if 'summary' in df_clean.columns:
        df_clean['summary'] = df_clean['summary'].fillna('')
    else:
        df_clean['summary'] = ''
    
    # Convert ratings to numeric
    df_clean['rating'] = pd.to_numeric(df_clean['rating'], errors='coerce')
    
    # Drop rows with missing required values
    df_clean = df_clean.dropna(subset=['product_id', 'user_id', 'rating'])
    
    # Convert review_date to datetime if it exists
    if 'review_date' in df_clean.columns:
        # Try different formats depending on what's in the data
        if df_clean['review_date'].dtype == 'int64' or df_clean['review_date'].dtype == 'float64':
            # Unix timestamp
            df_clean['review_date'] = pd.to_datetime(df_clean['review_date'], unit='s')
        else:
            # String date
            df_clean['review_date'] = pd.to_datetime(df_clean['review_date'], errors='coerce')
    else:
        df_clean['review_date'] = pd.Timestamp.now()
    
    # Add verified_purchase column if it doesn't exist
    if 'verified_purchase' not in df_clean.columns:
        df_clean['verified_purchase'] = False
    
    # Add votes column if it doesn't exist
    if 'votes' not in df_clean.columns or 'helpful_votes' in df_clean.columns:
        if 'helpful_votes' in df_clean.columns:
            df_clean['votes'] = df_clean['helpful_votes']
        else:
            df_clean['votes'] = 0
    
    logger.info(f"Data cleaned: {len(df_clean)} rows remaining")
    return df_clean


def filter_data(df: pd.DataFrame, min_user_reviews: int = 5, min_product_reviews: int = 5) -> pd.DataFrame:
    """
    Filter data to remove users and products with too few reviews.
    
    Args:
        df: DataFrame with cleaned data
        min_user_reviews: Minimum number of reviews per user
        min_product_reviews: Minimum number of reviews per product
        
    Returns:
        Filtered DataFrame
    """
    logger.info("Filtering data")
    
    # Count reviews per user and product
    user_counts = df['user_id'].value_counts()
    product_counts = df['product_id'].value_counts()
    
    # Filter users with enough reviews
    valid_users = user_counts[user_counts >= min_user_reviews].index
    df_filtered = df[df['user_id'].isin(valid_users)]
    
    # Filter products with enough reviews
    valid_products = product_counts[product_counts >= min_product_reviews].index
    df_filtered = df_filtered[df_filtered['product_id'].isin(valid_products)]
    
    logger.info(f"Data filtered: {len(df_filtered)} rows, {len(valid_users)} users, {len(valid_products)} products")
    return df_filtered


def sample_data(df: pd.DataFrame, n: int = DEFAULT_SAMPLE_SIZE, random_state: int = DEFAULT_RANDOM_STATE) -> pd.DataFrame:
    """
    Sample a subset of the data.
    
    Args:
        df: DataFrame to sample from
        n: Number of rows to sample
        random_state: Random seed for reproducibility
        
    Returns:
        Sampled DataFrame
    """
    if n >= len(df):
        logger.info(f"Requested sample size {n} >= data size {len(df)}, returning full dataset")
        return df
    
    logger.info(f"Sampling {n} rows")
    return df.sample(n=n, random_state=random_state).reset_index(drop=True)


def split_data(df: pd.DataFrame, 
               test_size: float = DEFAULT_TEST_SIZE, 
               validation_size: float = DEFAULT_VALIDATION_SIZE,
               random_state: int = DEFAULT_RANDOM_STATE) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        df: DataFrame to split
        test_size: Proportion of data to use for testing
        validation_size: Proportion of training data to use for validation
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train, validation, test) DataFrames
    """
    logger.info("Splitting data into train, validation, and test sets")
    
    # First split into train+validation and test
    train_val, test = train_test_split(df, test_size=test_size, random_state=random_state)
    
    # Then split train+validation into train and validation
    train, validation = train_test_split(train_val, test_size=validation_size, random_state=random_state)
    
    logger.info(f"Split data: {len(train)} train, {len(validation)} validation, {len(test)} test")
    return train, validation, test


def save_data(df: pd.DataFrame, 
             file_path: Union[str, Path], 
             format: str = 'parquet') -> None:
    """
    Save processed data to disk.
    
    Args:
        df: DataFrame to save
        file_path: Path to save to
        format: Format to save as ('csv', 'parquet', or 'pickle')
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    logger.info(f"Saving data to {file_path} in {format} format")
    
    # Save in the specified format
    if format == 'csv':
        df.to_csv(file_path, index=False)
    elif format == 'parquet':
        df.to_parquet(file_path, index=False)
    elif format == 'pickle':
        df.to_pickle(file_path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def preprocess_data(
    input_file: Union[str, Path],
    output_dir: Union[str, Path] = DEFAULT_PROCESSED_DATA_PATH,
    sample_size: Optional[int] = DEFAULT_SAMPLE_SIZE,
    min_user_reviews: int = 5,
    min_product_reviews: int = 5,
    test_size: float = DEFAULT_TEST_SIZE,
    validation_size: float = DEFAULT_VALIDATION_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE
) -> Dict[str, pd.DataFrame]:
    """
    Preprocess the Amazon product reviews dataset.
    
    Args:
        input_file: Path to the raw data file
        output_dir: Directory to save processed data
        sample_size: Number of rows to sample (None for all)
        min_user_reviews: Minimum number of reviews per user
        min_product_reviews: Minimum number of reviews per product
        test_size: Proportion of data to use for testing
        validation_size: Proportion of training data to use for validation
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing the train, validation, and test DataFrames
    """
    # Read raw data
    if sample_size is not None and sample_size > 0:
        df_raw = read_raw_data(input_file, nrows=sample_size*2)  # Read extra for filtering
    else:
        df_raw = read_raw_data(input_file)
    
    # Clean the data
    df_clean = clean_data(df_raw)
    
    # Filter users and products
    df_filtered = filter_data(df_clean, min_user_reviews, min_product_reviews)
    
    # Sample if needed
    if sample_size is not None and sample_size > 0 and len(df_filtered) > sample_size:
        df_sampled = sample_data(df_filtered, sample_size, random_state)
    else:
        df_sampled = df_filtered
    
    # Split the data
    train_df, val_df, test_df = split_data(df_sampled, test_size, validation_size, random_state)
    
    # Save the processed datasets
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    save_data(train_df, output_dir / 'train.parquet')
    save_data(val_df, output_dir / 'validation.parquet')
    save_data(test_df, output_dir / 'test.parquet')
    
    # Also save a small sample for quick testing
    small_sample = df_sampled.sample(min(1000, len(df_sampled)), random_state=random_state)
    save_data(small_sample, output_dir / 'sample.parquet')
    
    return {
        'train': train_df,
        'validation': val_df,
        'test': test_df,
        'sample': small_sample
    }


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess Amazon product reviews dataset')
    parser.add_argument('--input', type=str, required=True, help='Path to raw input file')
    parser.add_argument('--output', type=str, default=DEFAULT_PROCESSED_DATA_PATH, help='Output directory')
    parser.add_argument('--sample', type=int, default=DEFAULT_SAMPLE_SIZE, help='Sample size (0 for all)')
    parser.add_argument('--min-user', type=int, default=5, help='Minimum reviews per user')
    parser.add_argument('--min-product', type=int, default=5, help='Minimum reviews per product')
    
    args = parser.parse_args()
    
    sample_size = args.sample if args.sample > 0 else None
    
    preprocess_data(
        args.input,
        args.output,
        sample_size=sample_size,
        min_user_reviews=args.min_user,
        min_product_reviews=args.min_product
    ) 