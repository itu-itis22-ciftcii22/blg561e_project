"""
Data Preprocessing Module

Handles loading, filtering, and labeling of token transaction CSV files.
Processes raw transaction data and prepares it for graph construction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TokenPreprocessor:
    """
    Preprocessor for token transaction CSV files.
    
    Handles:
    - Loading CSV files from a directory
    - Filtering tokens with at least the required observation period
    - Labeling tokens as malicious based on value decline threshold
    - Extracting training window data for graph construction
    - Deduplication using signature column
    """
    
    # Time period constants (in seconds)
    TWO_WEEKS_SECONDS = 14 * 24 * 60 * 60
    ONE_WEEK_SECONDS = 7 * 24 * 60 * 60
    ONE_DAY_SECONDS = 24 * 60 * 60
    
    def __init__(
        self,
        data_dir: str,
        malice_threshold: float = 0.9,
        min_transactions: int = 10,
        observation_days: int = 90,
        training_window_days: int = 45,
        sudden_drop_threshold: float = 0.8,
        sudden_drop_window_hours: int = 48,
        classification_mode: str = 'sudden_drop'
    ):
        """
        Initialize the preprocessor.
        
        Args:
            data_dir: Directory containing token CSV files
            malice_threshold: Relative decline threshold for malicious classification
                             (e.g., 0.9 means 90% decline from peak to observation end)
            min_transactions: Minimum number of transactions required
            observation_days: Number of days for observation period (for labeling).
                             Default 90 (~3 months). Token must have this much history.
            training_window_days: Number of days of data to extract for training.
                                  Default 45 (~1.5 months). Should be < observation_days.
            sudden_drop_threshold: For 'sudden_drop' mode - minimum drop percentage (0.8 = 80%)
            sudden_drop_window_hours: For 'sudden_drop' mode - time window to detect sudden drops
            classification_mode: How to classify malicious tokens. Options:
                - 'peak_decline': Original method - (peak - end) / peak >= threshold
                - 'sudden_drop': Detect sudden drops (>threshold within window_hours)
                - 'initial_decline': Compare final value to initial trading value
                - 'combined': Use multiple signals (recommended)
        """
        self.data_dir = Path(data_dir)
        self.malice_threshold = malice_threshold
        self.min_transactions = min_transactions
        self.observation_days = observation_days
        self.training_window_days = training_window_days
        self.sudden_drop_threshold = sudden_drop_threshold
        self.sudden_drop_window_hours = sudden_drop_window_hours
        self.classification_mode = classification_mode
        
        # Computed time periods in seconds
        self.observation_seconds = observation_days * self.ONE_DAY_SECONDS
        self.training_window_seconds = training_window_days * self.ONE_DAY_SECONDS
        self.sudden_drop_window_seconds = sudden_drop_window_hours * 3600
        
    def get_csv_files(self) -> List[Path]:
        """Get all CSV files in the data directory."""
        return list(self.data_dir.glob("*.csv"))
    
    def load_csv(self, filepath: Path) -> pd.DataFrame:
        """
        Load and parse a token CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with parsed data including calculated token price
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
        df['Decimals'] = pd.to_numeric(df['Decimals'], errors='coerce').fillna(6)
        
        # CRITICAL FIX: Calculate actual token price
        # Value = USD value of transaction
        # Amount = raw token amount (needs decimal adjustment)
        # Price = Value / (Amount / 10^Decimals)
        decimals = df['Decimals'].iloc[0] if len(df) > 0 else 6
        df['token_amount'] = df['Amount'] / (10 ** decimals)
        
        # Calculate price per token (avoid division by zero)
        df['Price'] = df.apply(
            lambda row: row['Value'] / row['token_amount'] if row['token_amount'] > 0 else 0,
            axis=1
        )
        
        # Keep original Value as transaction_value for other uses
        df['transaction_value'] = df['Value']
        
        # Replace Value with Price for classification purposes
        # This is the actual token price we should use for malice detection
        df['Value'] = df['Price']
        
        return df
    
    def filter_by_duration(self, df: pd.DataFrame) -> bool:
        """
        Check if dataset covers at least the observation period.
        
        Args:
            df: Transaction DataFrame
            
        Returns:
            True if dataset spans at least observation_days
        """
        if len(df) < self.min_transactions:
            return False
            
        time_span = df['timestamp'].max() - df['timestamp'].min()
        return time_span >= self.observation_seconds
    
    def compute_label(self, df: pd.DataFrame) -> int:
        """
        Compute malicious label based on the selected classification mode.
        
        Args:
            df: Transaction DataFrame sorted by timestamp
            
        Returns:
            1 for malicious, 0 for benign
        """
        if self.classification_mode == 'peak_decline':
            return self._label_peak_decline(df)
        elif self.classification_mode == 'sudden_drop':
            return self._label_sudden_drop(df)
        elif self.classification_mode == 'initial_decline':
            return self._label_initial_decline(df)
        elif self.classification_mode == 'combined':
            return self._label_combined(df)
        else:
            raise ValueError(f"Unknown classification_mode: {self.classification_mode}")
    
    def _label_peak_decline(self, df: pd.DataFrame) -> int:
        """Original method: Label based on peak-to-end decline."""
        start_time = df['timestamp'].min()
        observation_end = start_time + self.observation_seconds
        
        df_observation = df[df['timestamp'] <= observation_end]
        if len(df_observation) == 0:
            return 0
        
        peak_value = df_observation['Value'].max()
        if peak_value <= 0:
            return 1
        
        value_at_end = df_observation.iloc[-1]['Value']
        relative_decline = (peak_value - value_at_end) / peak_value
        
        return 1 if relative_decline >= self.malice_threshold else 0
    
    def _label_sudden_drop(self, df: pd.DataFrame) -> int:
        """
        Detect sudden drops - characteristic of rug pulls.
        
        A sudden drop is defined as a drop of >threshold within a short time window.
        This is more indicative of malicious activity than gradual decline.
        """
        start_time = df['timestamp'].min()
        observation_end = start_time + self.observation_seconds
        
        df_observation = df[df['timestamp'] <= observation_end].copy()
        if len(df_observation) < 2:
            return 0
        
        # Sort by timestamp
        df_observation = df_observation.sort_values('timestamp')
        
        # Look for any sudden drop within the observation period
        values = df_observation['Value'].values
        timestamps = df_observation['timestamp'].values
        
        for i in range(len(df_observation)):
            current_value = values[i]
            current_time = timestamps[i]
            
            if current_value <= 0:
                continue
            
            # Look ahead within the sudden drop window
            window_end = current_time + self.sudden_drop_window_seconds
            future_mask = (timestamps > current_time) & (timestamps <= window_end)
            
            if not future_mask.any():
                continue
            
            # Find minimum value within the window
            min_future_value = values[future_mask].min()
            
            # Calculate drop
            drop = (current_value - min_future_value) / current_value
            
            if drop >= self.sudden_drop_threshold:
                return 1  # Sudden drop detected = malicious
        
        return 0  # No sudden drop = benign
    
    def _label_initial_decline(self, df: pd.DataFrame) -> int:
        """
        Compare final value to initial trading value (first few days).
        
        More robust than peak comparison as it doesn't penalize temporary spikes.
        """
        start_time = df['timestamp'].min()
        observation_end = start_time + self.observation_seconds
        
        df_observation = df[df['timestamp'] <= observation_end]
        if len(df_observation) < 2:
            return 0
        
        # Get average value from first 3 days as "initial value"
        initial_window_end = start_time + (3 * self.ONE_DAY_SECONDS)
        df_initial = df_observation[df_observation['timestamp'] <= initial_window_end]
        
        if len(df_initial) == 0 or df_initial['Value'].max() <= 0:
            return 0
        
        initial_value = df_initial['Value'].mean()
        if initial_value <= 0:
            initial_value = df_initial['Value'].max()
        
        if initial_value <= 0:
            return 1
        
        # Get final value (average of last 10% of transactions or last day)
        final_window_start = observation_end - self.ONE_DAY_SECONDS
        df_final = df_observation[df_observation['timestamp'] >= final_window_start]
        
        if len(df_final) == 0:
            df_final = df_observation.tail(max(1, len(df_observation) // 10))
        
        final_value = df_final['Value'].mean()
        
        # Calculate decline from initial
        decline = (initial_value - final_value) / initial_value
        
        return 1 if decline >= self.malice_threshold else 0
    
    def _label_combined(self, df: pd.DataFrame) -> int:
        """
        Combined approach using multiple signals.
        
        A token is malicious if ANY of these are true:
        1. Sudden drop detected (strongest signal)
        2. Both peak decline AND initial decline exceed threshold (confirms it's not just noise)
        """
        # Check for sudden drop first (strongest signal for rug pulls)
        if self._label_sudden_drop(df) == 1:
            return 1
        
        # Check if both peak and initial decline indicate malicious
        # This helps filter out false positives
        peak_malicious = self._label_peak_decline(df)
        initial_malicious = self._label_initial_decline(df)
        
        # Both methods agree = likely malicious
        if peak_malicious == 1 and initial_malicious == 1:
            return 1
        
        return 0
    
    def extract_training_window(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract training window data for graph construction.
        
        Args:
            df: Full transaction DataFrame
            
        Returns:
            DataFrame with only training window transactions
        """
        start_time = df['timestamp'].min()
        window_end = start_time + self.training_window_seconds
        
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
        
        # Normalized to training window (0 to 1)
        df['rel_time_normalized'] = df['rel_time_seconds'] / self.training_window_seconds
        df['rel_time_normalized'] = df['rel_time_normalized'].clip(0, 1)
        
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
            
            # Compute amount * value (transaction USD value = token_amount * price)
            # Note: token_amount is decimal-adjusted, Value is now token price
            df['amount_x_value'] = df['token_amount'] * df['Value']
            
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
