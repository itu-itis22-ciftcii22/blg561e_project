"""Token Preprocessor - CSV loading and standardization with parallel processing."""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Optional

import pandas as pd


class PreprocessorError(Exception):
    """Raised when preprocessing fails."""
    pass


class Cols:
    """Column name constants for preprocessed DataFrames."""
    FROM = 'from'
    TO = 'to'
    AMOUNT = 'amount'
    REL_TIME = 'rel_time'


@dataclass
class TokenDataFrame:
    """Preprocessed token data with DataFrame and metadata."""
    df: pd.DataFrame
    label: int
    token_name: str
    num_transactions: int
    num_nodes: int


def _load_single_file(filepath: Path):
    """Module-level function for multiprocessing (must be picklable)."""
    filepath = Path(filepath)
    
    try:
        df_raw = pd.read_csv(filepath)
    except Exception as e:
        raise PreprocessorError(f"Failed to read {filepath}: {e}") from e
    
    df_raw.columns = df_raw.columns.str.strip()
    
    required = ["from_address", "to_address", "value", "block_timestamp"]
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        raise PreprocessorError(f"Missing columns in {filepath.name}: {missing}")
    
    try:
        df_raw['_timestamp'] = pd.to_datetime(df_raw['block_timestamp'], format='ISO8601', utc=True)
    except Exception as e:
        raise PreprocessorError(f"Failed to parse block_timestamp in {filepath.name}: {e}") from e
    
    df_raw = df_raw.sort_values('_timestamp').reset_index(drop=True)
    
    all_addresses = pd.concat([df_raw['from_address'], df_raw['to_address']]).unique()
    addr_to_idx = {addr: idx for idx, addr in enumerate(all_addresses)}
    num_nodes = len(all_addresses)
    
    first_ts = df_raw['_timestamp'].min()
    last_ts = df_raw['_timestamp'].max()
    duration = (last_ts - first_ts).total_seconds()
    if duration == 0:
        duration = 1
    
    df = pd.DataFrame()
    df[Cols.FROM] = df_raw['from_address'].map(addr_to_idx)
    df[Cols.TO] = df_raw['to_address'].map(addr_to_idx)
    df[Cols.AMOUNT] = df_raw['value'].astype(float)
    df[Cols.REL_TIME] = (df_raw['_timestamp'] - first_ts).dt.total_seconds() / duration
    
    # Extract label from filename
    parts = filepath.stem.rsplit("_", 1)
    if len(parts) != 2 or parts[1] not in ("0", "1"):
        raise PreprocessorError(
            f"Invalid filename format: {filepath.name}. "
            "Expected: {{address}}_{{0|1}}.csv"
        )
    label = int(parts[1])
    
    return TokenDataFrame(
        df=df,
        label=label,
        token_name=filepath.stem.split("_")[0],
        num_transactions=len(df),
        num_nodes=num_nodes,
    )


def _load_file_safe(filepath: Path):
    """Wrapper that catches exceptions and returns (result, error) tuple."""
    try:
        return (_load_single_file(filepath), None)
    except PreprocessorError as e:
        return (None, str(e))


class TokenPreprocessor:
    """Load and standardize token transaction CSVs."""
    
    def __init__(self):
        pass
    
    def load_file(self, filepath: Path):
        """Load and preprocess a single CSV file."""
        return _load_single_file(filepath)
    
    def load_directory(
        self,
        directory: Path | str,
        max_files: Optional[int] = None,
        num_workers: Optional[int] = None,
    ):
        """Load all CSV files from a directory.
        
        Args:
            directory: Path to directory containing CSV files.
            max_files: Maximum number of files to load.
            num_workers: Number of parallel workers. None uses all CPU cores.
                         Set to 1 for sequential processing.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise PreprocessorError(f"Directory does not exist: {directory}")
        
        csv_files = list(directory.glob("*.csv"))
        if not csv_files:
            raise PreprocessorError(f"No CSV files found in {directory}")
        
        if max_files:
            csv_files = csv_files[:max_files]
        
        # Determine number of workers
        if num_workers is None:
            num_workers = cpu_count()
        
        results: list[TokenDataFrame] = []
        errors: list[str] = []
        
        if num_workers == 1:
            # Sequential processing
            for filepath in csv_files:
                result, error = _load_file_safe(filepath)
                if result is not None:
                    results.append(result)
                if error is not None:
                    errors.append(error)
        else:
            # Parallel processing
            with Pool(processes=num_workers) as pool:
                outputs = pool.map(_load_file_safe, csv_files)
            
            for result, error in outputs:
                if result is not None:
                    results.append(result)
                if error is not None:
                    errors.append(error)
        
        if not results:
            raise PreprocessorError(f"No valid files found. Errors: {errors[:5]}")
        
        return results, errors
    
    def _extract_label(self, filepath: Path):
        """Extract label from filename: {address}_{0|1}.csv"""
        parts = filepath.stem.rsplit("_", 1)
        if len(parts) != 2 or parts[1] not in ("0", "1"):
            raise PreprocessorError(
                f"Invalid filename format: {filepath.name}. "
                "Expected: {{address}}_{{0|1}}.csv"
            )
        label: int = int(parts[1])
        return label
