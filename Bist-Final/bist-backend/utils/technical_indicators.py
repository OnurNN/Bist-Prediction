import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def calculate_rsi(data: pd.Series, periods: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        data: Price series (typically Close prices)
        periods: RSI period (default 14)
    
    Returns:
        RSI values (0-100)
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Args:
        data: Price series
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
    
    Returns:
        DataFrame with macd, signal, and histogram columns
    """
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()
    
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    
    return pd.DataFrame({
        'macd': macd,
        'macd_signal': signal_line,
        'macd_histogram': histogram
    })

def calculate_bollinger_bands(data: pd.Series, periods: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """
    Calculate Bollinger Bands
    
    Args:
        data: Price series
        periods: Moving average period
        std_dev: Standard deviation multiplier
    
    Returns:
        DataFrame with upper, middle, and lower bands
    """
    middle = data.rolling(window=periods).mean()
    std = data.rolling(window=periods).std()
    
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    
    # Calculate bandwidth and %B
    bandwidth = (upper - lower) / middle
    percent_b = (data - lower) / (upper - lower)
    
    return pd.DataFrame({
        'bb_upper': upper,
        'bb_middle': middle,
        'bb_lower': lower,
        'bb_bandwidth': bandwidth,
        'bb_percent': percent_b
    })

def calculate_ema(data: pd.Series, periods: int) -> pd.Series:
    """
    Calculate Exponential Moving Average
    
    Args:
        data: Price series
        periods: EMA period
    
    Returns:
        EMA values
    """
    return data.ewm(span=periods, adjust=False).mean()

def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV)
    
    Args:
        close: Close price series
        volume: Volume series
    
    Returns:
        OBV values
    """
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv

def calculate_volume_roc(volume: pd.Series, periods: int = 14) -> pd.Series:
    """
    Calculate Volume Rate of Change
    
    Args:
        volume: Volume series
        periods: Period for ROC calculation
    
    Returns:
        Volume ROC values
    """
    roc = ((volume - volume.shift(periods)) / volume.shift(periods)) * 100
    return roc

def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                         k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """
    Calculate Stochastic Oscillator
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        k_period: %K period
        d_period: %D period
    
    Returns:
        DataFrame with %K and %D values
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=d_period).mean()
    
    return pd.DataFrame({
        'stoch_k': k,
        'stoch_d': d
    })

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, periods: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR)
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        periods: ATR period
    
    Returns:
        ATR values
    """
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    
    atr = true_range.rolling(window=periods).mean()
    return atr

def calculate_momentum(data: pd.Series, periods: int = 10) -> pd.Series:
    """
    Calculate Price Momentum
    
    Args:
        data: Price series
        periods: Lookback period
    
    Returns:
        Momentum values
    """
    return data.diff(periods)

def calculate_roc(data: pd.Series, periods: int = 12) -> pd.Series:
    """
    Calculate Rate of Change (ROC)
    
    Args:
        data: Price series
        periods: Period for ROC calculation
    
    Returns:
        ROC values (percentage)
    """
    roc = ((data - data.shift(periods)) / data.shift(periods)) * 100
    return roc

def add_lagged_features(data: pd.DataFrame, lags: list = [1, 3, 7, 14]) -> pd.DataFrame:
    """
    Add lagged price features
    
    Args:
        data: DataFrame with price data
        lags: List of lag periods
    
    Returns:
        DataFrame with lagged features added
    """
    df = data.copy()
    
    for lag in lags:
        df[f'close_lag_{lag}'] = df['Close'].shift(lag)
        df[f'return_lag_{lag}'] = df['Close'].pct_change(lag)
    
    return df

def add_rolling_statistics(data: pd.DataFrame, windows: list = [7, 14, 30]) -> pd.DataFrame:
    """
    Add rolling statistics
    
    Args:
        data: DataFrame with price data
        windows: List of window sizes
    
    Returns:
        DataFrame with rolling statistics added
    """
    df = data.copy()
    
    for window in windows:
        # Rolling mean and std
        df[f'rolling_mean_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df['Close'].rolling(window=window).std()
        
        # Rolling momentum
        df[f'rolling_momentum_{window}'] = df['Close'].pct_change(window)
        
        # Rolling min/max
        df[f'rolling_min_{window}'] = df['Close'].rolling(window=window).min()
        df[f'rolling_max_{window}'] = df['Close'].rolling(window=window).max()
        
        # Position within range
        df[f'range_position_{window}'] = (
            (df['Close'] - df[f'rolling_min_{window}']) / 
            (df[f'rolling_max_{window}'] - df[f'rolling_min_{window}'])
        )
    
    return df

def detect_market_regime(data: pd.DataFrame) -> pd.DataFrame:
    """
    Detect market regime (trending vs ranging)
    
    Args:
        data: DataFrame with price data
    
    Returns:
        DataFrame with market regime indicators added
    """
    df = data.copy()
    
    # ADX-like trend strength indicator
    # Calculate directional movement
    high_diff = df['High'].diff()
    low_diff = -df['Low'].diff()
    
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    
    # True range
    tr = pd.concat([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift()),
        abs(df['Low'] - df['Close'].shift())
    ], axis=1).max(axis=1)
    
    # Smooth indicators
    period = 14
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean())
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean())
    
    # Trend strength
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df['trend_strength'] = dx.rolling(window=period).mean()
    
    # Regime classification
    # High trend_strength (>25) = trending, low (<20) = ranging
    df['is_trending'] = (df['trend_strength'] > 25).astype(int)
    df['is_ranging'] = (df['trend_strength'] < 20).astype(int)
    
    # Trend direction
    df['trend_direction'] = np.where(
        plus_di > minus_di, 1,  # Uptrend
        np.where(plus_di < minus_di, -1, 0)  # Downtrend or neutral
    )
    
    # Volatility regime
    rolling_vol = df['Close'].pct_change().rolling(window=30).std() * np.sqrt(252)
    df['volatility_regime'] = np.where(
        rolling_vol > rolling_vol.rolling(window=60).mean(), 1, 0  # High vol = 1, Low vol = 0
    )
    
    return df

def calculate_pivot_points(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """
    Calculate Pivot Points (Classic) and Support/Resistance levels
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
    
    Returns:
        DataFrame with Pivot, R1-R3, S1-S3
    """
    pivot = (high + low + close) / 3
    
    r1 = (2 * pivot) - low
    s1 = (2 * pivot) - high
    
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    
    return pd.DataFrame({
        'pivot': pivot,
        'r1': r1,
        's1': s1,
        'r2': r2,
        's2': s2,
        'r3': r3,
        's3': s3
    })

def add_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to the dataframe
    
    Args:
        data: DataFrame with OHLCV data
    
    Returns:
        DataFrame with all technical indicators added
    """
    df = data.copy()
    
    try:
        # RSI
        df['rsi'] = calculate_rsi(df['Close'])
        
        # MACD
        macd_data = calculate_macd(df['Close'])
        df = pd.concat([df, macd_data], axis=1)
        
        # Bollinger Bands
        bb_data = calculate_bollinger_bands(df['Close'])
        df = pd.concat([df, bb_data], axis=1)
        
        # EMAs
        df['ema_12'] = calculate_ema(df['Close'], 12)
        df['ema_26'] = calculate_ema(df['Close'], 26)
        df['ema_50'] = calculate_ema(df['Close'], 50)
        df['ema_200'] = calculate_ema(df['Close'], 200)
        
        # Volume indicators
        df['obv'] = calculate_obv(df['Close'], df['Volume'])
        df['volume_roc'] = calculate_volume_roc(df['Volume'])
        
        # Stochastic
        stoch_data = calculate_stochastic(df['High'], df['Low'], df['Close'])
        df = pd.concat([df, stoch_data], axis=1)
        
        # ATR (volatility)
        df['atr'] = calculate_atr(df['High'], df['Low'], df['Close'])
        
        # Momentum indicators
        df['momentum'] = calculate_momentum(df['Close'])
        df['roc'] = calculate_roc(df['Close'])
        
        # Pivot Points
        pivot_data = calculate_pivot_points(df['High'], df['Low'], df['Close'])
        df = pd.concat([df, pivot_data], axis=1)
        
        # Price position relative to moving averages
        df['price_to_ema50'] = (df['Close'] / df['ema_50'] - 1) * 100
        df['price_to_ema200'] = (df['Close'] / df['ema_200'] - 1) * 100
        
        # Volatility
        df['volatility'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        
        # Lagged features
        df = add_lagged_features(df)
        
        # Rolling statistics
        df = add_rolling_statistics(df)
        
        # Market regime detection
        df = detect_market_regime(df)
        
        logger.info(f"Added {len(df.columns) - 5} technical indicators and features")
        
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {str(e)}")
        raise
    
    return df

def get_feature_columns(include_all: bool = True) -> list:
    """
    Get list of all technical indicator column names
    
    Args:
        include_all: If False, return only top features for Prophet
    
    Returns:
        List of feature column names
    """
    # Top features for Prophet (most important, numerically stable)
    top_features = [
        'rsi', 'macd', 'ema_12', 'ema_50', 
        'bb_percent', 'atr', 'volatility',
        'momentum', 'roc', 'price_to_ema50',
        'pivot', 'r1', 's1',
        'close_lag_1', 'close_lag_7',
        'rolling_mean_7', 'rolling_mean_14',
        'trend_strength', 'trend_direction'
    ]
    
    if not include_all:
        return top_features
    
    base_features = [
        'rsi', 'macd', 'macd_signal', 'macd_histogram',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_bandwidth', 'bb_percent',
        'ema_12', 'ema_26', 'ema_50', 'ema_200',
        'obv', 'volume_roc', 'stoch_k', 'stoch_d', 'atr',
        'momentum', 'roc', 'price_to_ema50', 'price_to_ema200', 'volatility',
        'pivot', 'r1', 's1', 'r2', 's2', 'r3', 's3'
    ]
    
    # Add lagged features
    lagged_features = []
    for lag in [1, 3, 7, 14]:
        lagged_features.extend([f'close_lag_{lag}', f'return_lag_{lag}'])
    
    # Add rolling statistics
    rolling_features = []
    for window in [7, 14, 30]:
        rolling_features.extend([
            f'rolling_mean_{window}', f'rolling_std_{window}',
            f'rolling_momentum_{window}', f'rolling_min_{window}',
            f'rolling_max_{window}', f'range_position_{window}'
        ])
    
    # Add market regime features
    regime_features = [
        'trend_strength', 'is_trending', 'is_ranging',
        'trend_direction', 'volatility_regime'
    ]
    
    return base_features + lagged_features + rolling_features + regime_features

def prepare_features_for_prophet(data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features specifically for Prophet model (handles NaN values)
    
    Args:
        data: DataFrame with technical indicators
    
    Returns:
        DataFrame with cleaned features for Prophet
    """
    df = data.copy()
    feature_cols = get_feature_columns(include_all=False)  # Use top features only
    
    # Forward fill then backward fill to handle NaN values
    for col in feature_cols:
        if col in df.columns:
            # Replace inf values with NaN first
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
            # Forward fill, then backward fill (new pandas syntax)
            df[col] = df[col].ffill().bfill()
            
            # If still NaN (e.g., at the beginning), fill with median
            if df[col].isna().any():
                median_val = df[col].median()
                # If median is also NaN, use 0
                if pd.isna(median_val):
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna(median_val)
            
            # Clip extreme values to prevent numerical issues
            if df[col].std() > 0:
                mean = df[col].mean()
                std = df[col].std()
                df[col] = df[col].clip(lower=mean - 5*std, upper=mean + 5*std)
    
    return df

