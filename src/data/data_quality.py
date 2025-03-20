"""
Data quality checks for the Amazon recommendation system.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.data.database import load_config

# Set up logging
logger = logging.getLogger(__name__)


class DataQualityChecker:
    """
    Data quality checker for the Amazon recommendation system.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize data quality checker.
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config if config is not None else load_config()
        self.results = {}
    
    def check_missing_values(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Check for missing values in each column.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dictionary mapping column names to percentage of missing values
        """
        logger.info("Checking for missing values")
        
        # Calculate percentage of missing values for each column
        missing_percentages = (df.isnull().sum() / len(df)) * 100
        
        # Convert to dictionary
        missing_dict = missing_percentages.to_dict()
        
        # Log results
        for col, pct in missing_dict.items():
            if pct > 0:
                logger.warning(f"Column '{col}' has {pct:.2f}% missing values")
            else:
                logger.info(f"Column '{col}' has no missing values")
        
        self.results['missing_values'] = missing_dict
        return missing_dict
    
    def check_duplicate_rows(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Check for duplicate rows.
        
        Args:
            df: DataFrame to check
            subset: List of columns to consider for duplicates (optional)
            
        Returns:
            Dictionary with duplicate count and percentage
        """
        logger.info("Checking for duplicate rows")
        
        # Count duplicate rows
        if subset is not None:
            duplicate_count = df.duplicated(subset=subset).sum()
            logger.info(f"Looking for duplicates in columns: {subset}")
        else:
            duplicate_count = df.duplicated().sum()
            logger.info("Looking for duplicates in all columns")
        
        # Calculate percentage
        duplicate_percentage = (duplicate_count / len(df)) * 100
        
        # Log results
        if duplicate_count > 0:
            logger.warning(f"Found {duplicate_count} duplicate rows ({duplicate_percentage:.2f}%)")
        else:
            logger.info("No duplicate rows found")
        
        result = {
            'duplicate_count': int(duplicate_count),
            'duplicate_percentage': float(duplicate_percentage)
        }
        
        self.results['duplicates'] = result
        return result
    
    def check_value_distributions(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Check value distributions for columns.
        
        Args:
            df: DataFrame to check
            columns: List of columns to check (optional, defaults to all)
            
        Returns:
            Dictionary mapping column names to distribution statistics
        """
        logger.info("Checking value distributions")
        
        # Determine columns to check
        if columns is None:
            columns = df.columns.tolist()
        
        result = {}
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in DataFrame")
                continue
            
            # Get column data
            col_data = df[col]
            
            # Build result dictionary based on data type
            if is_numeric_dtype(col_data):
                # Numeric data
                stats = {
                    'min': float(col_data.min()) if not pd.isna(col_data.min()) else None,
                    'max': float(col_data.max()) if not pd.isna(col_data.max()) else None,
                    'mean': float(col_data.mean()) if not pd.isna(col_data.mean()) else None,
                    'median': float(col_data.median()) if not pd.isna(col_data.median()) else None,
                    'std': float(col_data.std()) if not pd.isna(col_data.std()) else None,
                    'zeros_count': int((col_data == 0).sum()),
                    'zeros_percentage': float((col_data == 0).sum() / len(col_data) * 100),
                    'negative_count': int((col_data < 0).sum()),
                    'negative_percentage': float((col_data < 0).sum() / len(col_data) * 100)
                }
                
                # Check for unusual values
                if stats['std'] > 0:
                    z_scores = (col_data - stats['mean']) / stats['std']
                    outliers = (abs(z_scores) > 3).sum()
                    stats['outliers_count'] = int(outliers)
                    stats['outliers_percentage'] = float(outliers / len(col_data) * 100)
                
            else:
                # Categorical/text data
                value_counts = col_data.value_counts(dropna=False)
                unique_count = len(value_counts)
                
                stats = {
                    'unique_count': int(unique_count),
                    'unique_percentage': float(unique_count / len(col_data) * 100),
                    'top_value': str(col_data.mode().iloc[0]) if not col_data.mode().empty else None,
                    'top_value_count': int(value_counts.iloc[0]) if not value_counts.empty else 0,
                    'top_value_percentage': float(value_counts.iloc[0] / len(col_data) * 100) if not value_counts.empty else 0,
                    'empty_count': int((col_data == '').sum()) if col_data.dtype == 'object' else 0
                }
                
                if stats['unique_count'] == 1:
                    logger.warning(f"Column '{col}' has only one unique value: {stats['top_value']}")
                
                if stats['unique_count'] == len(col_data) and len(col_data) > 10:
                    logger.warning(f"Column '{col}' has all unique values, might be an ID column")
            
            result[col] = stats
            
            # Log basic stats
            logger.info(f"Column '{col}' stats: {stats}")
        
        self.results['value_distributions'] = result
        return result
    
    def check_data_consistency(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Check data consistency (e.g., ratings in valid range).
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dictionary with consistency check results
        """
        logger.info("Checking data consistency")
        
        result = {}
        
        # Check rating values (should be between 1 and 5)
        if 'rating' in df.columns:
            invalid_ratings = df[(df['rating'] < 1) | (df['rating'] > 5)]['rating']
            invalid_count = len(invalid_ratings)
            invalid_percentage = (invalid_count / len(df)) * 100
            
            result['rating'] = {
                'invalid_count': int(invalid_count),
                'invalid_percentage': float(invalid_percentage),
                'invalid_values': invalid_ratings.unique().tolist() if invalid_count > 0 else []
            }
            
            if invalid_count > 0:
                logger.warning(f"Found {invalid_count} invalid ratings ({invalid_percentage:.2f}%)")
            else:
                logger.info("All ratings are valid")
        
        # Check review dates (should not be in future)
        if 'review_date' in df.columns:
            df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
            future_dates = df[df['review_date'] > datetime.now()]['review_date']
            future_count = len(future_dates)
            future_percentage = (future_count / len(df)) * 100
            
            result['review_date'] = {
                'future_count': int(future_count),
                'future_percentage': float(future_percentage)
            }
            
            if future_count > 0:
                logger.warning(f"Found {future_count} reviews with dates in the future ({future_percentage:.2f}%)")
            else:
                logger.info("All review dates are valid")
        
        # Check user IDs and product IDs (should not be empty)
        for id_col in ['user_id', 'product_id']:
            if id_col in df.columns:
                empty_ids = df[df[id_col].isnull() | (df[id_col] == '')][id_col]
                empty_count = len(empty_ids)
                empty_percentage = (empty_count / len(df)) * 100
                
                result[id_col] = {
                    'empty_count': int(empty_count),
                    'empty_percentage': float(empty_percentage)
                }
                
                if empty_count > 0:
                    logger.warning(f"Found {empty_count} empty {id_col} values ({empty_percentage:.2f}%)")
                else:
                    logger.info(f"All {id_col} values are non-empty")
        
        self.results['data_consistency'] = result
        return result
    
    def check_data_relationships(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Check relationships between data fields.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dictionary with relationship check results
        """
        logger.info("Checking data relationships")
        
        result = {}
        
        # Check relationship between rating and review_text length
        if 'rating' in df.columns and 'review_text' in df.columns:
            df['review_length'] = df['review_text'].fillna('').str.len()
            
            # Group by rating and get mean review length
            length_by_rating = df.groupby('rating')['review_length'].mean().to_dict()
            
            # Check if low ratings have longer reviews (common pattern)
            if 1 in length_by_rating and 5 in length_by_rating:
                low_rating_length = length_by_rating[1]
                high_rating_length = length_by_rating[5]
                
                result['rating_review_length'] = {
                    'low_rating_length': float(low_rating_length),
                    'high_rating_length': float(high_rating_length),
                    'ratio': float(low_rating_length / high_rating_length) if high_rating_length > 0 else None
                }
                
                if low_rating_length > high_rating_length:
                    logger.info("Low ratings tend to have longer reviews, which is a common pattern")
                else:
                    logger.info("Low ratings do not have longer reviews, which is unusual")
        
        # Check relationship between verified_purchase and rating
        if 'verified_purchase' in df.columns and 'rating' in df.columns:
            verified_mean = df[df['verified_purchase'] == True]['rating'].mean()
            unverified_mean = df[df['verified_purchase'] == False]['rating'].mean()
            
            result['verified_rating'] = {
                'verified_mean': float(verified_mean),
                'unverified_mean': float(unverified_mean),
                'difference': float(verified_mean - unverified_mean)
            }
            
            if abs(verified_mean - unverified_mean) > 0.5:
                logger.warning(f"Large difference ({result['verified_rating']['difference']:.2f}) between verified and unverified purchase ratings")
            else:
                logger.info(f"Small difference ({result['verified_rating']['difference']:.2f}) between verified and unverified purchase ratings")
        
        self.results['data_relationships'] = result
        return result
    
    def check_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Check for temporal patterns in the data.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dictionary with temporal pattern check results
        """
        logger.info("Checking temporal patterns")
        
        result = {}
        
        # Check review date distribution
        if 'review_date' in df.columns:
            df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
            
            # Drop missing dates
            df_dates = df.dropna(subset=['review_date'])
            
            if len(df_dates) > 0:
                # Get basic date statistics
                min_date = df_dates['review_date'].min()
                max_date = df_dates['review_date'].max()
                date_range = (max_date - min_date).days
                
                result['date_range'] = {
                    'min_date': min_date.strftime('%Y-%m-%d'),
                    'max_date': max_date.strftime('%Y-%m-%d'),
                    'days': date_range
                }
                
                # Check for date clustering
                df_dates['month'] = df_dates['review_date'].dt.to_period('M')
                monthly_counts = df_dates.groupby('month').size()
                
                # Calculate stats on monthly distribution
                if len(monthly_counts) > 1:
                    mean_monthly = monthly_counts.mean()
                    max_monthly = monthly_counts.max()
                    spike_ratio = max_monthly / mean_monthly
                    
                    result['monthly_distribution'] = {
                        'months_count': int(len(monthly_counts)),
                        'mean_monthly': float(mean_monthly),
                        'max_monthly': float(max_monthly),
                        'min_monthly': float(monthly_counts.min()),
                        'spike_ratio': float(spike_ratio)
                    }
                    
                    if spike_ratio > 3:
                        logger.warning(f"Data has significant spikes in review counts (spike ratio: {spike_ratio:.2f})")
                    else:
                        logger.info(f"Review counts are relatively evenly distributed (spike ratio: {spike_ratio:.2f})")
        
        self.results['temporal_patterns'] = result
        return result
    
    def run_all_checks(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Run all data quality checks.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dictionary with all check results
        """
        logger.info(f"Running all data quality checks on DataFrame with {len(df)} rows and {len(df.columns)} columns")
        
        # Make a copy to avoid modifying the original DataFrame
        df_copy = df.copy()
        
        # Run all checks
        self.check_missing_values(df_copy)
        self.check_duplicate_rows(df_copy)
        self.check_value_distributions(df_copy)
        self.check_data_consistency(df_copy)
        self.check_data_relationships(df_copy)
        self.check_temporal_patterns(df_copy)
        
        # Log overall results
        logger.info("Data quality checks completed")
        
        return self.results
    
    def get_summary(self) -> Dict[str, Dict]:
        """
        Get a summary of data quality check results.
        
        Returns:
            Dictionary with summarized check results
        """
        summary = {}
        
        # Missing values summary
        if 'missing_values' in self.results:
            missing_summary = {
                'columns_with_missing': sum(1 for pct in self.results['missing_values'].values() if pct > 0),
                'columns_high_missing': sum(1 for pct in self.results['missing_values'].values() if pct > 20)
            }
            summary['missing_values'] = missing_summary
        
        # Duplicate rows summary
        if 'duplicates' in self.results:
            summary['duplicates'] = self.results['duplicates']
        
        # Data consistency summary
        if 'data_consistency' in self.results:
            consistency_summary = {
                'has_invalid_ratings': self.results['data_consistency'].get('rating', {}).get('invalid_count', 0) > 0,
                'has_future_dates': self.results['data_consistency'].get('review_date', {}).get('future_count', 0) > 0,
                'has_empty_ids': (
                    self.results['data_consistency'].get('user_id', {}).get('empty_count', 0) > 0 or
                    self.results['data_consistency'].get('product_id', {}).get('empty_count', 0) > 0
                )
            }
            summary['data_consistency'] = consistency_summary
        
        # Overall quality assessment
        issues_count = 0
        
        if 'missing_values' in summary and summary['missing_values']['columns_high_missing'] > 0:
            issues_count += summary['missing_values']['columns_high_missing']
        
        if 'duplicates' in summary and summary['duplicates']['duplicate_percentage'] > 5:
            issues_count += 1
        
        if 'data_consistency' in summary:
            for key, value in summary['data_consistency'].items():
                if value:
                    issues_count += 1
        
        if issues_count == 0:
            quality_level = "Excellent"
        elif issues_count <= 2:
            quality_level = "Good"
        elif issues_count <= 5:
            quality_level = "Fair"
        else:
            quality_level = "Poor"
        
        summary['overall'] = {
            'issues_count': issues_count,
            'quality_level': quality_level
        }
        
        return summary


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Example usage
    import argparse
    import json
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description='Data quality checks for Amazon reviews')
    parser.add_argument('--input', type=str, required=True, help='Path to input file')
    parser.add_argument('--output', type=str, help='Path to output JSON file for results')
    
    args = parser.parse_args()
    
    # Load data
    input_path = Path(args.input)
    if input_path.suffix == '.csv':
        df = pd.read_csv(input_path)
    elif input_path.suffix == '.parquet':
        df = pd.read_parquet(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path}")
    
    # Run data quality checks
    checker = DataQualityChecker()
    results = checker.run_all_checks(df)
    summary = checker.get_summary()
    
    # Print summary
    logger.info(f"Data quality summary: {summary['overall']['quality_level']} ({summary['overall']['issues_count']} issues found)")
    
    # Save results if output path is provided
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert datetime objects to strings for JSON serialization
        def json_serializable(obj):
            if isinstance(obj, (datetime, pd.Timestamp)):
                return obj.isoformat()
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            raise TypeError(f"Type {type(obj)} not serializable")
        
        with open(output_path, 'w') as f:
            json.dump({
                'results': results,
                'summary': summary
            }, f, default=json_serializable, indent=2)
        
        logger.info(f"Results saved to {output_path}") 