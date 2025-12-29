"""
Data Preprocessing Module

Handles loading, filtering, and labeling of token transaction CSV files.
Processes raw transaction data and prepares it for graph construction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class TokenPreprocessor:
    """
    Preprocessor for token transaction CSV files.
    
    Handles:
    - Loading CSV files from a directory
    - Filtering tokens with at least 3 months of history
    - Labeling tokens as malicious based on value decline threshold
    - Extracting first 1.5 months of data for graph construction
    - Deduplication using signature column
    """
    
    # Time period constants (in seconds)
    THREE_MONTHS_SECONDS = 90 * 24 * 60 * 60  # ~3 months
    SIX_WEEKS_SECONDS = 45 * 24 * 60 * 60  # 1.5 months
    TWO_WEEKS_SECONDS = 14 * 24 * 60 * 60
    ONE_WEEK_SECONDS = 7 * 24 * 60 * 60
    ONE_DAY_SECONDS = 24 * 60 * 60
    
    def __init__(
        self,
        data_dir: str,
        malice_threshold: float = 0.9,
        min_transactions: int = 10
    ):
        """
        Initialize the preprocessor.
        
        Args:
            data_dir: Directory containing token CSV files
            malice_threshold: Relative decline threshold for malicious classification
                             (e.g., 0.9 means 90% decline from peak to 3-month end)
            min_transactions: Minimum number of transactions required
        """
        self.data_dir = Path(data_dir)
        self.malice_threshold = malice_threshold
        self.min_transactions = min_transactions
        
    def get_csv_files(self) -> List[Path]:
        """Get all CSV files in the data directory."""
        return list(self.data_dir.glob("*.csv"))
    
    def load_csv(self, filepath: Path) -> pd.DataFrame:
        """
        Load and parse a token CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with parsed data
        """
        df = pd.read_csv(filepath)
        
        # Standardize column names (strip whitespace)
        df.columns = df.columns.str.strip()
        
        # Parse timestamp - prefer Block Time (unix timestamp)
        if 'Block Time' in df.columns:
            df['timestamp'] = pd.to_numeric(df['Block Time'], errors='coerce')
        elif 'Human Time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['Human Time']).astype(int) // 10**9
        
        # Ensure numeric columns
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce').fillna(0)
        
        return df
    
    def filter_by_duration(self, df: pd.DataFrame) -> bool:
        """
        Check if dataset covers at least 3 months.
        
        Args:
            df: Transaction DataFrame
            
        Returns:
            True if dataset spans at least 3 months
        """
        if len(df) < self.min_transactions:
            return False
            
        time_span = df['timestamp'].max() - df['timestamp'].min()
        return time_span >= self.THREE_MONTHS_SECONDS
    
    def compute_label(self, df: pd.DataFrame) -> int:
        """
        Compute malicious label based on value decline.
        
        Label is 1 (malicious) if:
        - (peak_value - value_at_3months) / peak_value >= threshold
        
        Args:
            df: Transaction DataFrame sorted by timestamp
            
        Returns:
            1 for malicious, 0 for benign
        """
        start_time = df['timestamp'].min()
        three_month_end = start_time + self.THREE_MONTHS_SECONDS
        
        # Get peak value in the dataset
        peak_value = df['Value'].max()
        
        if peak_value <= 0:
            return 1  # No value = likely malicious
        
        # Get value at 3-month mark (closest transaction)
        df_at_3m = df[df['timestamp'] <= three_month_end]
        if len(df_at_3m) == 0:
            return 0
        
        # Use the last value before 3-month mark
        value_at_3m = df_at_3m.iloc[-1]['Value']
        
        # Calculate relative decline
        relative_decline = (peak_value - value_at_3m) / peak_value
        
        return 1 if relative_decline >= self.malice_threshold else 0
    
    def extract_training_window(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract first 1.5 months of data for graph construction.
        
        Args:
            df: Full transaction DataFrame
            
        Returns:
            DataFrame with only first 1.5 months of transactions
        """
        start_time = df['timestamp'].min()
        window_end = start_time + self.SIX_WEEKS_SECONDS
        
        return df[df['timestamp'] <= window_end].copy()
    
    def deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate transactions based on Signature.
        
        Args:
            df: Transaction DataFrame
            
        Returns:
            Deduplicated DataFrame
        """
        if 'Signature' in df.columns:
            initial_len = len(df)
            df = df.drop_duplicates(subset=['Signature'])
            logger.debug(f"Removed {initial_len - len(df)} duplicate transactions")
        return df
    
    def compute_relative_times(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute relative time features from dataset start.
        
        Adds columns for time periods:
        - rel_time: relative time from start (normalized to 1.5 months)
        - period_6w: period within 1.5 months (0-1)
        - period_2w: period within 2 weeks (0-1 repeating)
        - period_1w: period within 1 week (0-1 repeating)
        - period_1d: period within 1 day (0-1 repeating)
        
        Args:
            df: Transaction DataFrame with timestamp column
            
        Returns:
            DataFrame with relative time columns added
        """
        start_time = df['timestamp'].min()
        df = df.copy()
        
        # Relative time from start
        df['rel_time_seconds'] = df['timestamp'] - start_time
        
        # Normalized to 1.5 months (0 to 1)
        df['rel_time_6w'] = df['rel_time_seconds'] / self.SIX_WEEKS_SECONDS
        df['rel_time_6w'] = df['rel_time_6w'].clip(0, 1)
        
        # Period features (cyclical within each period)
        df['period_2w'] = (df['rel_time_seconds'] % self.TWO_WEEKS_SECONDS) / self.TWO_WEEKS_SECONDS
        df['period_1w'] = (df['rel_time_seconds'] % self.ONE_WEEK_SECONDS) / self.ONE_WEEK_SECONDS
        df['period_1d'] = (df['rel_time_seconds'] % self.ONE_DAY_SECONDS) / self.ONE_DAY_SECONDS
        
        return df
    
    def process_single_token(
        self, 
        filepath: Path
    ) -> Optional[Tuple[pd.DataFrame, int]]:
        """
        Process a single token CSV file.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            Tuple of (processed_df, label) or None if invalid
        """
        try:
            # Load data
            df = self.load_csv(filepath)
            
            # Check duration
            if not self.filter_by_duration(df):
                logger.debug(f"Skipping {filepath.name}: insufficient duration")
                return None
            
            # Sort by timestamp (ascending - oldest first)
            df = df.sort_values('timestamp', ascending=True).reset_index(drop=True)
            
            # Compute label before extracting window
            label = self.compute_label(df)
            
            # Extract training window (first 1.5 months)
            df = self.extract_training_window(df)
            
            # Deduplicate
            df = self.deduplicate(df)
            
            if len(df) < self.min_transactions:
                logger.debug(f"Skipping {filepath.name}: too few transactions after filtering")
                return None
            
            # Compute relative times
            df = self.compute_relative_times(df)
            
            # Compute amount * value
            df['amount_x_value'] = df['Amount'] * df['Value']
            
            logger.info(f"Processed {filepath.name}: {len(df)} transactions, label={label}")
            return df, label
            
        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            return None
    
    def process_directory(self) -> List[Tuple[pd.DataFrame, int, str]]:
        """
        Process all valid token CSV files in the directory.
        
        Returns:
            List of (processed_df, label, token_name) tuples
        """
        results = []
        csv_files = self.get_csv_files()
        
        logger.info(f"Found {len(csv_files)} CSV files in {self.data_dir}")
        
        for filepath in csv_files:
            result = self.process_single_token(filepath)
            if result is not None:
                df, label = result
                token_name = filepath.stem  # filename without extension
                results.append((df, label, token_name))
        
        # Log statistics
        labels = [r[1] for r in results]
        logger.info(f"Processed {len(results)} valid tokens: "
                   f"{sum(labels)} malicious, {len(labels) - sum(labels)} benign")
        
        return results
