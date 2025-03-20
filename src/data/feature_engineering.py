"""
Feature engineering for the Amazon recommendation system.
"""

import logging
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from src.data.database import load_config

# Set up logging
logger = logging.getLogger(__name__)

# Load configuration
config = load_config()
feature_config = config.get('features', {})

# Extract feature configurations
TEXT_FEATURES = feature_config.get('text_features', ['review_text', 'summary'])
NUMERICAL_FEATURES = feature_config.get('numerical_features', ['rating', 'votes'])
CATEGORICAL_FEATURES = feature_config.get('categorical_features', ['category', 'subcategory'])
TEMPORAL_FEATURES = feature_config.get('temporal_features', ['review_date'])


class FeatureEngineering:
    """
    Feature engineering for Amazon product reviews data.
    """
    
    def __init__(self, df: Optional[pd.DataFrame] = None):
        """
        Initialize feature engineering.
        
        Args:
            df: DataFrame with preprocessed data (optional)
        """
        self.df = df
        self.tfidf_vectorizers = {}
        self.label_encoders = {}
        self.standard_scalers = {}
        self.minmax_scalers = {}
        
        # Initialize feature extraction components
        self._initialize_text_vectorizers()
        
    def _initialize_text_vectorizers(self) -> None:
        """Initialize TF-IDF vectorizers for text features."""
        for feature in TEXT_FEATURES:
            self.tfidf_vectorizers[feature] = TfidfVectorizer(
                min_df=config.get('model', {}).get('content_based', {}).get('min_df', 0.01),
                max_df=config.get('model', {}).get('content_based', {}).get('max_df', 0.85),
                ngram_range=tuple(config.get('model', {}).get('content_based', {}).get('ngram_range', [1, 2])),
                stop_words='english'
            )
            
    def fit(self, df: pd.DataFrame) -> 'FeatureEngineering':
        """
        Fit feature engineering components to data.
        
        Args:
            df: DataFrame with preprocessed data
            
        Returns:
            Self for method chaining
        """
        self.df = df
        
        # Fit text vectorizers
        self._fit_text_vectorizers(df)
        
        # Fit categorical encoders
        self._fit_categorical_encoders(df)
        
        # Fit numerical scalers
        self._fit_numerical_scalers(df)
        
        return self
    
    def _fit_text_vectorizers(self, df: pd.DataFrame) -> None:
        """
        Fit TF-IDF vectorizers for text features.
        
        Args:
            df: DataFrame with text features
        """
        for feature in TEXT_FEATURES:
            if feature in df.columns:
                logger.info(f"Fitting TF-IDF vectorizer for {feature}")
                text_data = df[feature].fillna('').astype(str)
                self.tfidf_vectorizers[feature].fit(text_data)
                
    def _fit_categorical_encoders(self, df: pd.DataFrame) -> None:
        """
        Fit label encoders for categorical features.
        
        Args:
            df: DataFrame with categorical features
        """
        for feature in CATEGORICAL_FEATURES:
            if feature in df.columns:
                logger.info(f"Fitting label encoder for {feature}")
                self.label_encoders[feature] = LabelEncoder()
                # Add an 'unknown' category for new values during transform
                categories = df[feature].fillna('unknown').astype(str).tolist()
                if 'unknown' not in categories:
                    categories.append('unknown')
                self.label_encoders[feature].fit(categories)
                
    def _fit_numerical_scalers(self, df: pd.DataFrame) -> None:
        """
        Fit scalers for numerical features.
        
        Args:
            df: DataFrame with numerical features
        """
        for feature in NUMERICAL_FEATURES:
            if feature in df.columns:
                logger.info(f"Fitting scalers for {feature}")
                # Standard scaler for features that may have negative values
                self.standard_scalers[feature] = StandardScaler()
                self.standard_scalers[feature].fit(df[[feature]].fillna(0))
                
                # MinMax scaler for features that should be between 0 and 1
                self.minmax_scalers[feature] = MinMaxScaler()
                self.minmax_scalers[feature].fit(df[[feature]].fillna(0))
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted feature engineering components.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            DataFrame with engineered features
        """
        # Make a copy to avoid modifying the original DataFrame
        result_df = df.copy()
        
        # Transform text features
        result_df = self._transform_text_features(result_df)
        
        # Transform categorical features
        result_df = self._transform_categorical_features(result_df)
        
        # Transform numerical features
        result_df = self._transform_numerical_features(result_df)
        
        # Generate temporal features
        result_df = self._generate_temporal_features(result_df)
        
        # Generate interaction features
        result_df = self._generate_interaction_features(result_df)
        
        return result_df
    
    def _transform_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform text features using TF-IDF.
        
        Args:
            df: DataFrame with text features
            
        Returns:
            DataFrame with text features transformed
        """
        result_df = df.copy()
        
        for feature in TEXT_FEATURES:
            if feature in df.columns and feature in self.tfidf_vectorizers:
                logger.info(f"Transforming text feature: {feature}")
                text_data = df[feature].fillna('').astype(str)
                
                # For now, just add a feature indicating text length
                result_df[f"{feature}_length"] = text_data.str.len()
                result_df[f"{feature}_word_count"] = text_data.str.split().str.len()
                
                # We don't add the full TF-IDF matrix to the DataFrame as it would be too large
                # Instead, we'll use it directly in the content-based model
        
        return result_df
    
    def _transform_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform categorical features using label encoding.
        
        Args:
            df: DataFrame with categorical features
            
        Returns:
            DataFrame with categorical features encoded
        """
        result_df = df.copy()
        
        for feature in CATEGORICAL_FEATURES:
            if feature in df.columns and feature in self.label_encoders:
                logger.info(f"Transforming categorical feature: {feature}")
                # Handle missing values and new categories
                categories = df[feature].fillna('unknown').astype(str)
                
                # Handle unseen categories
                for cat in categories.unique():
                    if cat not in self.label_encoders[feature].classes_:
                        categories = categories.replace(cat, 'unknown')
                
                try:
                    result_df[f"{feature}_encoded"] = self.label_encoders[feature].transform(categories)
                except ValueError as e:
                    logger.warning(f"Error encoding {feature}: {e}")
                    # Fallback: assign 0 for unknown values
                    result_df[f"{feature}_encoded"] = 0
        
        return result_df
    
    def _transform_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform numerical features using scaling.
        
        Args:
            df: DataFrame with numerical features
            
        Returns:
            DataFrame with numerical features scaled
        """
        result_df = df.copy()
        
        for feature in NUMERICAL_FEATURES:
            if feature in df.columns:
                logger.info(f"Transforming numerical feature: {feature}")
                # Fill missing values
                values = df[[feature]].fillna(0)
                
                if feature in self.standard_scalers:
                    # Add standardized feature
                    result_df[f"{feature}_scaled"] = self.standard_scalers[feature].transform(values)
                
                if feature in self.minmax_scalers:
                    # Add normalized feature
                    result_df[f"{feature}_norm"] = self.minmax_scalers[feature].transform(values)
        
        return result_df
    
    def _generate_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features from temporal data.
        
        Args:
            df: DataFrame with temporal features
            
        Returns:
            DataFrame with additional temporal features
        """
        result_df = df.copy()
        
        for feature in TEMPORAL_FEATURES:
            if feature in df.columns:
                logger.info(f"Generating temporal features from: {feature}")
                # Ensure the column is datetime
                try:
                    datetime_col = pd.to_datetime(df[feature])
                    
                    # Extract useful time components
                    result_df[f"{feature}_year"] = datetime_col.dt.year
                    result_df[f"{feature}_month"] = datetime_col.dt.month
                    result_df[f"{feature}_day"] = datetime_col.dt.day
                    result_df[f"{feature}_dayofweek"] = datetime_col.dt.dayofweek
                    result_df[f"{feature}_hour"] = datetime_col.dt.hour
                    
                    # Calculate time-based features
                    now = datetime.now()
                    result_df[f"{feature}_days_since"] = (now - datetime_col).dt.days
                    
                except Exception as e:
                    logger.warning(f"Error generating temporal features from {feature}: {e}")
        
        return result_df
    
    def _generate_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate interaction features between different feature types.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with additional interaction features
        """
        result_df = df.copy()
        
        # Example: Interaction between rating and verified_purchase
        if 'rating' in df.columns and 'verified_purchase' in df.columns:
            logger.info("Generating rating x verified_purchase interaction feature")
            result_df['rating_verified'] = df['rating'] * df['verified_purchase'].astype(int)
        
        # Example: Ratio of votes to review length (if available)
        if 'votes' in df.columns and 'review_text_length' in result_df.columns:
            logger.info("Generating votes to review length ratio feature")
            result_df['votes_per_length'] = df['votes'] / (result_df['review_text_length'] + 1)  # Add 1 to avoid division by zero
        
        return result_df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit to data, then transform it.
        
        Args:
            df: DataFrame to fit and transform
            
        Returns:
            Transformed DataFrame with engineered features
        """
        return self.fit(df).transform(df)
    
    @lru_cache(maxsize=1)
    def get_text_features_matrix(self, feature: str, texts: Tuple[str]) -> np.ndarray:
        """
        Get TF-IDF matrix for text features (cached for performance).
        
        Args:
            feature: Name of the text feature
            texts: Tuple of text strings to transform
            
        Returns:
            TF-IDF matrix
        """
        if feature not in self.tfidf_vectorizers:
            raise ValueError(f"No vectorizer found for feature: {feature}")
        
        return self.tfidf_vectorizers[feature].transform(texts)
    
    def get_product_features(self, df: pd.DataFrame, product_id_col: str = 'product_id') -> pd.DataFrame:
        """
        Generate product-level features by aggregating review data.
        
        Args:
            df: DataFrame with review data
            product_id_col: Name of the product ID column
            
        Returns:
            DataFrame with product features
        """
        logger.info("Generating product-level features")
        
        # Group by product_id and calculate aggregates
        product_features = df.groupby(product_id_col).agg({
            'rating': ['mean', 'median', 'std', 'count'],
            'votes': ['sum', 'mean', 'max'],
        })
        
        # Flatten the multi-level column names
        product_features.columns = ['_'.join(col).strip() for col in product_features.columns.values]
        
        # Add more features
        if 'verified_purchase' in df.columns:
            verified_counts = df[df['verified_purchase'] == True].groupby(product_id_col).size()
            product_features['verified_purchase_count'] = verified_counts.reindex(product_features.index, fill_value=0)
            product_features['verified_purchase_ratio'] = product_features['verified_purchase_count'] / product_features['rating_count']
        
        # Reset index to make product_id a column
        product_features = product_features.reset_index()
        
        return product_features
    
    def get_user_features(self, df: pd.DataFrame, user_id_col: str = 'user_id') -> pd.DataFrame:
        """
        Generate user-level features by aggregating review data.
        
        Args:
            df: DataFrame with review data
            user_id_col: Name of the user ID column
            
        Returns:
            DataFrame with user features
        """
        logger.info("Generating user-level features")
        
        # Group by user_id and calculate aggregates
        user_features = df.groupby(user_id_col).agg({
            'rating': ['mean', 'median', 'std', 'count'],
            'votes': ['sum', 'mean', 'max'],
        })
        
        # Flatten the multi-level column names
        user_features.columns = ['_'.join(col).strip() for col in user_features.columns.values]
        
        # Add more features
        if 'verified_purchase' in df.columns:
            verified_counts = df[df['verified_purchase'] == True].groupby(user_id_col).size()
            user_features['verified_purchase_count'] = verified_counts.reindex(user_features.index, fill_value=0)
            user_features['verified_purchase_ratio'] = user_features['verified_purchase_count'] / user_features['rating_count']
        
        # Calculate rating deviation from average (to detect bias)
        overall_avg_rating = df['rating'].mean()
        user_features['rating_bias'] = user_features['rating_mean'] - overall_avg_rating
        
        # Reset index to make user_id a column
        user_features = user_features.reset_index()
        
        return user_features


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Example usage
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description='Feature engineering for Amazon reviews')
    parser.add_argument('--input', type=str, required=True, help='Path to preprocessed input file')
    parser.add_argument('--output', type=str, required=True, help='Output path for features')
    
    args = parser.parse_args()
    
    # Load preprocessed data
    input_path = Path(args.input)
    if input_path.suffix == '.csv':
        df = pd.read_csv(input_path)
    elif input_path.suffix == '.parquet':
        df = pd.read_parquet(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path}")
    
    # Apply feature engineering
    feature_eng = FeatureEngineering()
    df_features = feature_eng.fit_transform(df)
    
    # Generate product and user features
    product_features = feature_eng.get_product_features(df)
    user_features = feature_eng.get_user_features(df)
    
    # Save results
    output_path = Path(args.output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df_features.to_parquet(output_path)
    product_features.to_parquet(output_dir / 'product_features.parquet')
    user_features.to_parquet(output_dir / 'user_features.parquet')
    
    logger.info(f"Feature engineering complete. Results saved to {output_dir}") 