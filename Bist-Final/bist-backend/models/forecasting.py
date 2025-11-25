import pandas as pd
import numpy as np
from prophet import Prophet
from typing import Dict, List
import logging
import sys
import os

# Add parent directory to path to import technical_indicators
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.technical_indicators import add_all_indicators, prepare_features_for_prophet, get_feature_columns

logger = logging.getLogger(__name__)

# Workaround for Prophet 1.2.1 stan_backend bug on Windows
# Cache the backend globally to avoid reinitialization issues
_cached_stan_backend = None
_original_load_stan_backend = Prophet._load_stan_backend

def _patched_load_stan_backend(self, stan_backend):
    global _cached_stan_backend
    try:
        _original_load_stan_backend(self, stan_backend)
        # Cache the successfully loaded backend
        if _cached_stan_backend is None:
            _cached_stan_backend = self.stan_backend
    except (AttributeError, UnicodeDecodeError) as e:
        # Reuse the cached backend if available
        if _cached_stan_backend is not None:
            self.stan_backend = _cached_stan_backend
        else:
            raise

Prophet._load_stan_backend = _patched_load_stan_backend

def _determine_trend_from_predictions(predictions: List[Dict[str, float]], threshold: float = 2.0) -> str:
    """
    Determine trend direction from a list of prediction dictionaries.

    Args:
        predictions: List of dicts that include a 'predicted' price.
        threshold: Percent change needed to classify as up/down.
    """
    if not predictions:
        return "neutral"

    start_price = predictions[0].get('predicted')
    end_price = predictions[-1].get('predicted')

    if start_price in (None, 0) or end_price is None:
        return "neutral"

    change_percent = ((end_price - start_price) / start_price) * 100

    if change_percent > threshold:
        return "upward"
    if change_percent < -threshold:
        return "downward"
    return "neutral"


class StockForecaster:
    def __init__(self):
        self.model = None
        self.best_params = None
        
    def forecast(self, data: pd.DataFrame, days: int = 30, model_type: str = "prophet", 
                 use_grid_search: bool = False, prophet_params: Dict = None) -> Dict:
        try:
            if model_type == "prophet":
                try:
                    # Run grid search if requested and not done yet
                    if use_grid_search and self.best_params is None:
                        logger.info("Running grid search to find best Prophet parameters")
                        self.best_params = self.grid_search_prophet(data, days)
                        logger.info(f"Best parameters found: {self.best_params}")
                    
                    # Use provided params, or best_params from grid search, or defaults
                    params = prophet_params or self.best_params
                    return self._forecast_prophet(data, days, params)
                except Exception as prophet_error:
                    logger.warning(f"Prophet failed, falling back to simple MA: {prophet_error}")
                    return self._forecast_simple_ma(data, days)
            elif model_type == "simple":
                return self._forecast_simple_ma(data, days)
            else:
                raise ValueError(f"Unsupported model: {model_type}")
        except Exception as e:
            logger.error(f"Forecast error: {str(e)}")
            raise
    
    def _forecast_simple_ma(self, data: pd.DataFrame, days: int) -> Dict:
        """Simple Moving Average forecasting - reliable fallback method"""
        logger.info(f"Using Simple Moving Average forecasting for {days} days")
        
        # Calculate moving averages for trend detection
        ma7 = data['Close'].rolling(window=7).mean()
        ma30 = data['Close'].rolling(window=30).mean()
        
        # Calculate trend from recent 30 days
        recent_prices = data['Close'].iloc[-30:]
        trend_change = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
        daily_change_rate = trend_change / 30
        
        # Get starting point
        last_price = data['Close'].iloc[-1]
        last_date = data.index[-1]
        
        # Generate predictions
        predictions = []
        for i in range(1, days + 1):
            # Linear projection with trend
            predicted_price = last_price * (1 + daily_change_rate * i)
            
            # Calculate uncertainty bounds (wider as we go further out)
            uncertainty = 0.08 + (i / days * 0.05)  # 8% to 13% uncertainty
            lower = predicted_price * (1 - uncertainty)
            upper = predicted_price * (1 + uncertainty)
            
            # Calculate future date using business days
            future_date = last_date + pd.tseries.offsets.BusinessDay(n=i)
            
            predictions.append({
                'date': future_date.strftime('%Y-%m-%d'),
                'predicted': round(predicted_price, 2),
                'lower': round(lower, 2),
                'upper': round(upper, 2)
            })
        
        # Calculate confidence based on historical volatility
        volatility = data['Close'].pct_change().std()
        # Lower volatility = higher confidence
        confidence = max(55, min(85, 75 - (volatility * 1000)))
        
        logger.info(f"Simple MA forecast: {len(predictions)} predictions, confidence: {confidence:.1f}%")
        
        return {
            'predictions': predictions,
            'confidence': round(confidence, 2),
            'model': 'Simple Moving Average',
            'trend': _determine_trend_from_predictions(predictions)
        }
    
    def _forecast_prophet(self, data: pd.DataFrame, days: int, params: Dict = None) -> Dict:
        # Add technical indicators to data
        logger.info("Adding technical indicators to data")
        data_with_indicators = add_all_indicators(data)
        data_with_indicators = prepare_features_for_prophet(data_with_indicators)
        
        # Prepare data for Prophet
        df_prophet = pd.DataFrame({
            'ds': data_with_indicators.index,
            'y': data_with_indicators['Close'].values
        })
        
        # Add technical indicators as additional regressors
        # Use only top features for Prophet to avoid numerical issues
        feature_cols = get_feature_columns(include_all=False)  # Use top 16 features only
        available_features = [col for col in feature_cols if col in data_with_indicators.columns]
        
        for col in available_features:
            df_prophet[col] = data_with_indicators[col].values
        
        logger.info(f"Using {len(available_features)} top technical indicators as regressors for Prophet")
        
        # Default parameters (based on your best test results)
        default_params = {
            'daily_seasonality': False,
            'weekly_seasonality': True,
            'yearly_seasonality': True,
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 0.2,
            'interval_width': 0.95,
            'changepoint_range': 0.9
        }
        
        # Use provided params or defaults
        model_params = params if params is not None else default_params
        
        # Train model with technical indicators
        model = Prophet(**model_params)
        
        # Add regressors to the model
        for col in available_features:
            model.add_regressor(col, standardize=True)
        
        model.fit(df_prophet)
        
        # Generate predictions
        # Use 'B' for business days (Mon-Fri) to skip weekends
        future = model.make_future_dataframe(periods=days, freq='B')
        
        # For future dates, we need to extrapolate technical indicators
        # Use simple forward fill for simplicity (can be improved with forecasting each indicator)
        for col in available_features:
            # Extend the technical indicator values for future dates
            last_values = df_prophet[col].tail(days).values
            # Ensure we have enough values to fill
            num_future_rows = len(future) - len(df_prophet)
            if num_future_rows > 0:
                # Repeat last values enough times to cover future rows
                tiled_values = np.tile(last_values, (num_future_rows // len(last_values)) + 2)
                future.loc[future.index >= len(df_prophet), col] = tiled_values[:num_future_rows]
            
            # Fill historical values
            future.loc[future.index < len(df_prophet), col] = df_prophet[col].values
        
        forecast = model.predict(future)
        forecast_only = forecast.tail(days)
        
        # Calculate confidence
        confidence = self._calculate_confidence(forecast_only)
        
        # Format results
        predictions = []
        for _, row in forecast_only.iterrows():
            predictions.append({
                'date': row['ds'].strftime('%Y-%m-%d'),
                'predicted': round(row['yhat'], 2),
                'lower': round(row['yhat_lower'], 2),
                'upper': round(row['yhat_upper'], 2)
            })
        
        return {
            'predictions': predictions,
            'confidence': confidence,
            'model': 'Prophet',
            'trend': self._determine_trend(forecast_only)
        }
    
    def _calculate_confidence(self, forecast: pd.DataFrame) -> float:
        interval_widths = forecast['yhat_upper'] - forecast['yhat_lower']
        avg_width = interval_widths.mean()
        avg_price = forecast['yhat'].mean()
        relative_width = avg_width / avg_price
        confidence = max(0, min(100, (1 - relative_width) * 100))
        return confidence
    
    def _determine_trend(self, forecast: pd.DataFrame) -> str:
        first_price = forecast['yhat'].iloc[0]
        last_price = forecast['yhat'].iloc[-1]
        change_percent = ((last_price - first_price) / first_price) * 100
        
        if change_percent > 2:
            return "upward"
        elif change_percent < -2:
            return "downward"
        else:
            return "neutral"
    
    def grid_search_prophet(self, data: pd.DataFrame, days: int, n_splits: int = 3) -> Dict:
        """
        Perform grid search to find best Prophet parameters using walk-forward validation
        
        Args:
            data: Historical price data
            days: Forecast horizon
            n_splits: Number of validation splits
        
        Returns:
            Dictionary with best parameters
        """
        logger.info(f"Starting grid search with {n_splits} validation splits")
        
        # Parameter grid based on your experiments
        param_grid = {
            'changepoint_prior_scale': [0.03, 0.05, 0.07, 0.1],
            'seasonality_prior_scale': [0.1, 0.2, 0.5, 1.0],
            'daily_seasonality': [False],  # Keep False based on your tests
            'weekly_seasonality': [True],
            'yearly_seasonality': [True],
            'changepoint_range': [0.85, 0.9, 0.95],
            'interval_width': [0.95]
        }
        
        best_score = float('inf')
        best_params = None
        
        # Generate parameter combinations (limit to avoid too many combinations)
        from itertools import product
        
        changepoint_scales = param_grid['changepoint_prior_scale']
        seasonality_scales = param_grid['seasonality_prior_scale']
        changepoint_ranges = param_grid['changepoint_range']
        
        total_combinations = len(changepoint_scales) * len(seasonality_scales) * len(changepoint_ranges)
        logger.info(f"Testing {total_combinations} parameter combinations")
        
        combination_count = 0
        for cp_scale in changepoint_scales:
            for s_scale in seasonality_scales:
                for cp_range in changepoint_ranges:
                    combination_count += 1
                    
                    params = {
                        'daily_seasonality': False,
                        'weekly_seasonality': True,
                        'yearly_seasonality': True,
                        'changepoint_prior_scale': cp_scale,
                        'seasonality_prior_scale': s_scale,
                        'interval_width': 0.95,
                        'changepoint_range': cp_range
                    }
                    
                    # Walk-forward validation
                    errors = []
                    for split in range(n_splits):
                        try:
                            # Calculate split point
                            test_size = days
                            split_offset = split * days
                            train_end = len(data) - test_size - split_offset
                            
                            if train_end < 252:  # Need at least 1 year of training data
                                continue
                            
                            train_data = data.iloc[:train_end]
                            test_data = data.iloc[train_end:train_end + test_size]
                            
                            if len(test_data) < days:
                                continue
                            
                            # Make prediction
                            forecast_result = self._forecast_prophet(train_data, days, params)
                            
                            # Calculate error
                            predicted_price = forecast_result['predictions'][-1]['predicted']
                            actual_price = test_data['Close'].iloc[-1]
                            error = abs(predicted_price - actual_price) / actual_price
                            errors.append(error)
                            
                        except Exception as e:
                            logger.warning(f"Error in split {split}: {str(e)}")
                            continue
                    
                    if errors:
                        avg_error = np.mean(errors)
                        
                        if avg_error < best_score:
                            best_score = avg_error
                            best_params = params.copy()
                            logger.info(f"[{combination_count}/{total_combinations}] New best: {avg_error:.4f} with cp_scale={cp_scale}, s_scale={s_scale}, cp_range={cp_range}")
        
        if best_params is None:
            logger.warning("Grid search failed, using default parameters")
            best_params = {
                'daily_seasonality': False,
                'weekly_seasonality': True,
                'yearly_seasonality': True,
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 0.2,
                'interval_width': 0.95,
                'changepoint_range': 0.9
            }
        
        logger.info(f"Grid search complete. Best error: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        return best_params