import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


def determine_trend_from_predictions(predictions: List[Dict[str, float]], threshold: float = 2.0) -> str:
    """
    Determine trend direction from forecast predictions.

    Args:
        predictions: List of dicts with 'predicted' price field.
        threshold: Percent change boundary for classifying trend.
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

class ARIMAForecaster:
    """ARIMA/SARIMAX time series forecasting model"""
    
    def __init__(self):
        self.model = None
        self.fitted_model = None
        
    def forecast(self, data: pd.DataFrame, days: int = 30) -> Dict:
        """
        Forecast using ARIMA model
        
        Args:
            data: Historical price data
            days: Number of days to forecast
        
        Returns:
            Dictionary with predictions and confidence
        """
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            
            logger.info(f"Training ARIMA model for {days}-day forecast")
            
            # Prepare data
            prices = data['Close'].values
            
            # Auto-detect best ARIMA parameters (simplified approach)
            # In production, you'd want to do proper grid search
            order = (1, 1, 1)  # (p, d, q)
            seasonal_order = (1, 1, 1, 7)  # (P, D, Q, s) - weekly seasonality
            
            # Fit SARIMAX model
            model = SARIMAX(
                prices,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            self.fitted_model = model.fit(disp=False)
            
            # Make predictions
            forecast = self.fitted_model.forecast(steps=days)
            forecast_ci = self.fitted_model.get_forecast(steps=days).conf_int()
            
            # Format predictions
            predictions = []
            last_date = data.index[-1]
            
            for i in range(days):
                future_date = last_date + pd.tseries.offsets.BusinessDay(n=i+1)
                predictions.append({
                    'date': future_date.strftime('%Y-%m-%d'),
                    'predicted': round(float(forecast.iloc[i]), 2),
                    'lower': round(float(forecast_ci.iloc[i, 0]), 2),
                    'upper': round(float(forecast_ci.iloc[i, 1]), 2)
                })
            
            # Calculate confidence based on prediction interval width
            avg_interval_width = (forecast_ci.iloc[:, 1] - forecast_ci.iloc[:, 0]).mean()
            avg_price = forecast.mean()
            relative_width = avg_interval_width / avg_price
            confidence = max(50, min(90, (1 - relative_width) * 100))
            
            logger.info(f"ARIMA forecast complete with {confidence:.1f}% confidence")
            
            return {
                'predictions': predictions,
                'confidence': round(confidence, 2),
                'model': 'ARIMA',
                'trend': determine_trend_from_predictions(predictions)
            }
            
        except Exception as e:
            logger.error(f"ARIMA forecasting error: {str(e)}")
            raise

class XGBoostForecaster:
    """XGBoost regression model for time series forecasting"""
    
    def __init__(self):
        self.model = None
        self.best_params = {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        self.feature_importance = None
        
    def create_features(self, data: pd.DataFrame, forecast_horizon: int = 1) -> tuple:
        """
        Create features for XGBoost
        
        Args:
            data: Historical data with indicators
            forecast_horizon: Days ahead to predict
        
        Returns:
            X (features), y (target), feature_cols (list of feature names)
        """
        import numpy as np
        import pandas as pd
        from utils.technical_indicators import add_all_indicators, get_feature_columns
        
        # Add all technical indicators
        df = add_all_indicators(data)
        
        # Get all feature columns
        feature_cols = get_feature_columns(include_all=True)  # Use all features for XGBoost
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        # Clean features for XGBoost
        for col in feature_cols:
            # Replace inf with NaN
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN with forward/backward fill
            df[col] = df[col].ffill().bfill()
            
            # If still NaN, fill with median or 0
            if df[col].isna().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0)
            
            # Clip extreme values
            if df[col].std() > 0:
                mean = df[col].mean()
                std = df[col].std()
                df[col] = df[col].clip(lower=mean - 5*std, upper=mean + 5*std)
        
        # Create target (future price)
        df['target'] = df['Close'].shift(-forecast_horizon)
        
        # Remove rows with NaN in target
        df = df.dropna(subset=['target'])
        
        # Final check: remove any remaining NaN or inf
        for col in feature_cols:
            df[col] = df[col].replace([np.inf, -np.inf], 0).fillna(0)
        
        X = df[feature_cols]
        y = df['target']
        
        return X, y, feature_cols
    
    def forecast(self, data: pd.DataFrame, days: int = 30) -> Dict:
        """
        Forecast using XGBoost model
        
        Args:
            data: Historical price data
            days: Number of days to forecast
        
        Returns:
            Dictionary with predictions and confidence
        """
        try:
            import xgboost as xgb
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.metrics import mean_squared_error
            
            logger.info(f"Training XGBoost model for {days}-day forecast")
            
            # Create features (predicting 1 day ahead iteratively)
            X, y, feature_cols = self.create_features(data, forecast_horizon=1)
            
            # Train/Test split for validation
            tscv = TimeSeriesSplit(n_splits=5)
            val_scores = []
            
            for train_index, val_index in tscv.split(X):
                X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                
                model = xgb.XGBRegressor(**self.best_params)
                model.fit(X_train, y_train)
                
                preds = model.predict(X_val)
                mse = mean_squared_error(y_val, preds)
                val_scores.append(np.sqrt(mse))
            
            # Train final model on all data
            self.model = xgb.XGBRegressor(**self.best_params)
            self.model.fit(X, y)
            
            # Store feature importance
            self.feature_importance = dict(zip(feature_cols, self.model.feature_importances_))
            
            # Make multi-step predictions
            predictions = []
            last_date = data.index[-1]
            
            # We need to iteratively update features for multi-step forecasting
            # This is complex with technical indicators, so we use a recursive strategy
            # with simplified feature updates or just direct multi-step if supported
            # Here we use a recursive strategy assuming features are mostly price-derived
            
            current_data = data.copy()
            
            for i in range(days):
                # Re-calculate features for the latest state
                # Note: This is computationally expensive but accurate
                # For speed, we could just update the price-based features
                
                # Get latest features
                X_latest, _, _ = self.create_features(current_data, forecast_horizon=0) # 0 just to get features
                latest_features = X_latest.iloc[[-1]]
                
                # Predict next day
                pred_price = self.model.predict(latest_features)[0]
                
                # Calculate uncertainty
                avg_val_error = np.mean(val_scores)
                uncertainty = avg_val_error * (1 + i * 0.1)  # Uncertainty grows with time
                
                future_date = last_date + pd.tseries.offsets.BusinessDay(n=i+1)
                predictions.append({
                    'date': future_date.strftime('%Y-%m-%d'),
                    'predicted': round(float(pred_price), 2),
                    'lower': round(float(pred_price - 1.96 * uncertainty), 2),
                    'upper': round(float(pred_price + 1.96 * uncertainty), 2)
                })
                
                # Append prediction to data to forecast next step
                new_row = pd.DataFrame({
                    'Open': [pred_price],
                    'High': [pred_price],
                    'Low': [pred_price],
                    'Close': [pred_price],
                    'Volume': [current_data['Volume'].iloc[-1]] # Assume volume stays same
                }, index=[future_date])
                
                current_data = pd.concat([current_data, new_row])
            
            # Calculate confidence
            avg_price = data['Close'].mean()
            avg_error = np.mean(val_scores)
            confidence = max(50, min(90, (1 - avg_error/avg_price) * 100))
            
            logger.info(f"XGBoost forecast complete with {confidence:.1f}% confidence")
            
            return {
                'predictions': predictions,
                'confidence': round(confidence, 2),
                'model': 'XGBoost',
                'trend': determine_trend_from_predictions(predictions)
            }
            
        except Exception as e:
            logger.error(f"XGBoost forecasting error: {str(e)}")
            raise

    def grid_search_xgboost(self, data: pd.DataFrame, n_splits: int = 3) -> Dict:
        """
        Perform grid search for XGBoost hyperparameters
        
        Args:
            data: Historical price data
            n_splits: Number of time series splits
            
        Returns:
            Best parameters dictionary
        """
        import xgboost as xgb
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import mean_squared_error
        import itertools
        
        logger.info("Starting XGBoost grid search...")
        
        # Create features
        X, y, _ = self.create_features(data, forecast_horizon=1)
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        keys, values = zip(*param_grid.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        best_score = float('inf')
        best_params = None
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        print(f"Testing {len(combinations)} parameter combinations...")
        
        for i, params in enumerate(combinations):
            params['n_jobs'] = -1
            params['random_state'] = 42
            
            scores = []
            for train_index, val_index in tscv.split(X):
                X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                
                model = xgb.XGBRegressor(**params)
                model.fit(X_train, y_train)
                
                preds = model.predict(X_val)
                mse = mean_squared_error(y_val, preds)
                scores.append(np.sqrt(mse))
            
            avg_score = np.mean(scores)
            
            if avg_score < best_score:
                best_score = avg_score
                best_params = params
                # logger.info(f"New best XGBoost params: {best_params} (MSE: {best_score:.4f})")
        
        logger.info(f"XGBoost grid search complete. Best params: {best_params}")
        self.best_params = best_params
        return best_params

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if self.feature_importance:
            # Sort by importance
            return dict(sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True))
        return {}

class LSTMForecaster:
    """LSTM neural network for time series forecasting"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        
    def prepare_sequences(self, data: np.ndarray, lookback: int = 60) -> tuple:
        """
        Prepare sequences for LSTM training
        
        Args:
            data: Time series data
            lookback: Number of previous time steps to use
        
        Returns:
            X (sequences), y (targets)
        """
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def forecast(self, data: pd.DataFrame, days: int = 30) -> Dict:
        """
        Forecast using LSTM model
        
        Args:
            data: Historical price data
            days: Number of days to forecast
        
        Returns:
            Dictionary with predictions and confidence
        """
        try:
            from sklearn.preprocessing import MinMaxScaler
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
            
            # Suppress TensorFlow warnings
            tf.get_logger().setLevel('ERROR')
            
            logger.info(f"Training LSTM model for {days}-day forecast")
            
            # Prepare data
            prices = data['Close'].values.reshape(-1, 1)
            
            # Scale data
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = self.scaler.fit_transform(prices)
            
            # Create sequences
            lookback = min(60, len(data) // 3)
            X, y = self.prepare_sequences(scaled_data, lookback)
            
            if len(X) < 50:
                raise ValueError("Insufficient data for LSTM training")
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Build LSTM model
            self.model = keras.Sequential([
                layers.LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
                layers.Dropout(0.2),
                layers.LSTM(50, return_sequences=False),
                layers.Dropout(0.2),
                layers.Dense(25),
                layers.Dense(1)
            ])
            
            self.model.compile(optimizer='adam', loss='mse')
            
            # Train model
            self.model.fit(
                X_train, y_train,
                epochs=20,
                batch_size=32,
                validation_data=(X_val, y_val),
                verbose=0
            )
            
            # Make multi-step predictions
            predictions = []
            last_date = data.index[-1]
            
            # Start with last sequence
            current_sequence = scaled_data[-lookback:].reshape(1, lookback, 1)
            
            for i in range(days):
                # Predict next value
                pred_scaled = self.model.predict(current_sequence, verbose=0)[0, 0]
                
                # Convert back to original scale
                pred_price = self.scaler.inverse_transform([[pred_scaled]])[0, 0]
                
                # Calculate uncertainty based on validation error
                val_predictions = self.model.predict(X_val, verbose=0)
                val_error = np.std(self.scaler.inverse_transform(y_val) - 
                                 self.scaler.inverse_transform(val_predictions))
                
                # Uncertainty increases with forecast horizon
                uncertainty = val_error * (1 + i * 0.1)
                
                future_date = last_date + pd.tseries.offsets.BusinessDay(n=i+1)
                predictions.append({
                    'date': future_date.strftime('%Y-%m-%d'),
                    'predicted': round(float(pred_price), 2),
                    'lower': round(float(pred_price - 1.96 * uncertainty), 2),
                    'upper': round(float(pred_price + 1.96 * uncertainty), 2)
                })
                
                # Update sequence for next prediction
                current_sequence = np.append(current_sequence[:, 1:, :], 
                                            [[pred_scaled]], axis=1)
            
            # Calculate confidence
            val_pred_prices = self.scaler.inverse_transform(val_predictions)
            val_true_prices = self.scaler.inverse_transform(y_val)
            mape = np.mean(np.abs((val_true_prices - val_pred_prices) / val_true_prices)) * 100
            confidence = max(50, min(90, 100 - mape))
            
            logger.info(f"LSTM forecast complete with {confidence:.1f}% confidence")
            
            return {
                'predictions': predictions,
                'confidence': round(confidence, 2),
                'model': 'LSTM',
                'trend': determine_trend_from_predictions(predictions)
            }
            
        except Exception as e:
            logger.error(f"LSTM forecasting error: {str(e)}")
            raise
