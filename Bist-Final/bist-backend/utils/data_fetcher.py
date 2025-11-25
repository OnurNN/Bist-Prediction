import requests
import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

def detect_outliers(data: pd.DataFrame, column: str = 'Close', threshold: float = 3.5) -> pd.Series:
    """
    Detect outliers using modified Z-score method
    
    Args:
        data: DataFrame with price data
        column: Column to check for outliers
        threshold: Z-score threshold (default 3.5)
    
    Returns:
        Boolean series indicating outliers
    """
    if column not in data.columns:
        return pd.Series([False] * len(data), index=data.index)
    
    # Calculate modified Z-score using median absolute deviation
    median = data[column].median()
    mad = np.median(np.abs(data[column] - median))
    
    if mad == 0:
        return pd.Series([False] * len(data), index=data.index)
    
    modified_z_scores = 0.6745 * (data[column] - median) / mad
    outliers = np.abs(modified_z_scores) > threshold
    
    return outliers

def handle_outliers(data: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
    """
    Handle outliers in price data
    
    Args:
        data: DataFrame with OHLCV data
        method: Method to handle outliers ('interpolate', 'clip', or 'remove')
    
    Returns:
        DataFrame with outliers handled
    """
    df = data.copy()
    
    for col in ['Open', 'High', 'Low', 'Close']:
        if col not in df.columns:
            continue
            
        outliers = detect_outliers(df, col)
        n_outliers = outliers.sum()
        
        if n_outliers > 0:
            logger.info(f"Found {n_outliers} outliers in {col}")
            
            if method == 'interpolate':
                # Replace outliers with interpolated values
                df.loc[outliers, col] = np.nan
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
            elif method == 'clip':
                # Clip to 3 standard deviations from mean
                mean = df[col].mean()
                std = df[col].std()
                df[col] = df[col].clip(lower=mean - 3*std, upper=mean + 3*std)
            elif method == 'remove':
                # Remove rows with outliers (not recommended for time series)
                df = df[~outliers]
    
    return df

def validate_data(data: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate the quality of fetched data
    
    Args:
        data: DataFrame with OHLCV data
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if data is None or len(data) == 0:
        return False, "No data available"
    
    # Check minimum data points
    if len(data) < 60:
        return False, f"Insufficient data: only {len(data)} days (need at least 60)"
    
    # Check for required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        return False, f"Missing columns: {missing_cols}"
    
    # Check for excessive NaN values
    for col in required_cols:
        nan_pct = data[col].isna().sum() / len(data) * 100
        if nan_pct > 10:
            return False, f"Too many NaN values in {col}: {nan_pct:.1f}%"
    
    # Check for zero or negative prices
    for col in ['Open', 'High', 'Low', 'Close']:
        if (data[col] <= 0).any():
            return False, f"Invalid prices (zero or negative) in {col}"
    
    # Check OHLC relationships
    invalid_ohlc = (
        (data['High'] < data['Low']) |
        (data['High'] < data['Open']) |
        (data['High'] < data['Close']) |
        (data['Low'] > data['Open']) |
        (data['Low'] > data['Close'])
    )
    
    if invalid_ohlc.any():
        n_invalid = invalid_ohlc.sum()
        logger.warning(f"Found {n_invalid} rows with invalid OHLC relationships")
        # This is a warning, not a failure
    
    # Check for gaps (more than 7 consecutive days missing is suspicious)
    date_diffs = pd.Series(data.index).diff().dt.days
    max_gap = date_diffs.max()
    if max_gap > 14:
        logger.warning(f"Large gap in data: {max_gap} days")
    
    return True, "Data validation passed"

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare data for modeling
    
    Args:
        data: Raw DataFrame with OHLCV data
    
    Returns:
        Cleaned DataFrame
    """
    df = data.copy()
    
    # Remove duplicate indices
    df = df[~df.index.duplicated(keep='first')]
    
    # Sort by date
    df = df.sort_index()
    
    # Handle NaN values in Close price (most important)
    if df['Close'].isna().any():
        logger.warning("Filling NaN values in Close price")
        df['Close'] = df['Close'].fillna(method='ffill').fillna(method='bfill')
    
    # Handle NaN in other columns
    for col in ['Open', 'High', 'Low']:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df['Close'])
    
    if 'Volume' in df.columns and df['Volume'].isna().any():
        df['Volume'] = df['Volume'].fillna(0)
    
    # Handle outliers
    df = handle_outliers(df, method='interpolate')
    
    return df

def fetch_stock_data(stock_id: str, period: str = "2y", validate: bool = True) -> Optional[pd.DataFrame]:
    try:
        logger.info(f"Fetching {stock_id} directly from Yahoo Finance API, period: {period}")
        
        # Yahoo Finance API endpoint (same as your n8n workflow)
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{stock_id}"
        
        # Parameters
        params = {
            'interval': '1d',
            'range': period  # '1y', '6mo', '3mo', etc.
        }
        
        # Headers to avoid blocking
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        logger.info(f"Calling Yahoo Finance: {url} with params: {params}")
        
        response = requests.get(url, params=params, headers=headers, timeout=30)
        
        logger.info(f"Yahoo Finance response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Parse Yahoo Finance response structure
            chart = data['chart']['result'][0]
            
            # Extract timestamps and convert to dates
            timestamps = chart['timestamp']
            dates = [datetime.fromtimestamp(ts) for ts in timestamps]
            
            # Extract OHLCV data
            quote = chart['indicators']['quote'][0]
            
            # Create DataFrame
            df = pd.DataFrame({
                'date': dates,
                'Open': quote['open'],
                'High': quote['high'],
                'Low': quote['low'],
                'Close': quote['close'],
                'Volume': quote['volume']
            })
            
            # Set date as index
            df = df.set_index('date')
            
            # Clean the data
            df = clean_data(df)
            
            # Validate data if requested
            if validate:
                is_valid, message = validate_data(df)
                if not is_valid:
                    logger.error(f"Data validation failed: {message}")
                    return None
                logger.info(message)
            
            logger.info(f"Successfully fetched {len(df)} data points")
            logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
            logger.info(f"Latest close price: {df['Close'].iloc[-1]}")
            
            return df
            
        else:
            logger.error(f"Yahoo Finance returned status {response.status_code}")
            logger.error(f"Response: {response.text[:500]}")  # First 500 chars
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error calling Yahoo Finance: {str(e)}")
        return None
    except KeyError as e:
        logger.error(f"Unexpected Yahoo Finance response structure - missing key: {str(e)}")
        logger.error(f"Response structure: {data.keys() if 'data' in locals() else 'No data'}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None