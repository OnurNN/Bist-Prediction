import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

from models.forecasting import StockForecaster
from models.ml_models import ARIMAForecaster, XGBoostForecaster, LSTMForecaster

logger = logging.getLogger(__name__)

class EnsembleForecaster:
    """
    Ensemble forecasting system that combines multiple models
    with intelligent weighting based on performance
    """
    
    def __init__(self):
        self.prophet = StockForecaster()
        self.arima = ARIMAForecaster()
        self.xgboost = XGBoostForecaster()
        self.lstm = LSTMForecaster()
        
        # Default weights (can be updated based on backtest performance)
        self.weights = {
            'prophet': 0.30,
            'arima': 0.20,
            'xgboost': 0.30,
            'lstm': 0.20
        }
        
    def forecast(self, data: pd.DataFrame, days: int = 30, 
                 models: List[str] = None, use_adaptive_weights: bool = True) -> Dict:
        """
        Generate ensemble forecast combining multiple models
        
        Args:
            data: Historical price data
            days: Number of days to forecast
            models: List of models to use (None = all models)
            use_adaptive_weights: Whether to adjust weights based on confidence
        
        Returns:
            Dictionary with ensemble predictions and metadata
        """
        if models is None:
            models = ['prophet', 'xgboost']  # Default: use faster models
        
        logger.info(f"Generating ensemble forecast with models: {models}")
        
        # Collect predictions from each model
        model_results = {}
        model_errors = []
        
        for model_name in models:
            try:
                logger.info(f"Running {model_name} model...")
                
                if model_name == 'prophet':
                    result = self.prophet.forecast(data, days, model_type='prophet')
                elif model_name == 'arima':
                    result = self.arima.forecast(data, days)
                elif model_name == 'xgboost':
                    result = self.xgboost.forecast(data, days)
                elif model_name == 'lstm':
                    result = self.lstm.forecast(data, days)
                else:
                    logger.warning(f"Unknown model: {model_name}")
                    continue
                
                model_results[model_name] = result
                logger.info(f"{model_name} complete: confidence={result['confidence']:.1f}%")
                
            except Exception as e:
                logger.error(f"Error running {model_name}: {str(e)}")
                model_errors.append(model_name)
                continue
        
        if not model_results:
            raise ValueError("All models failed to generate predictions")
        
        # Calculate adaptive weights based on confidence scores
        if use_adaptive_weights and len(model_results) > 1:
            weights = self._calculate_adaptive_weights(model_results)
        else:
            # Use default weights, normalized for available models
            weights = {k: self.weights.get(k, 0.25) for k in model_results.keys()}
            total = sum(weights.values())
            weights = {k: v/total for k, v in weights.items()}
        
        logger.info(f"Using weights: {weights}")
        
        # Combine predictions
        ensemble_predictions = self._combine_predictions(model_results, weights, days)
        
        # Calculate ensemble confidence
        ensemble_confidence = self._calculate_ensemble_confidence(model_results, weights)
        
        # Calculate model agreement score
        agreement_score = self._calculate_agreement_score(model_results, days)
        
        # Determine trend
        trend = self._determine_ensemble_trend(ensemble_predictions)
        
        return {
            'predictions': ensemble_predictions,
            'confidence': round(ensemble_confidence, 2),
            'model': 'Ensemble',
            'trend': trend,
            'individual_models': {
                name: {
                    'confidence': res['confidence'],
                    'final_prediction': res['predictions'][-1]['predicted']
                }
                for name, res in model_results.items()
            },
            'weights': weights,
            'agreement_score': round(agreement_score, 2),
            'models_used': list(model_results.keys()),
            'models_failed': model_errors
        }
    
    def _calculate_adaptive_weights(self, model_results: Dict) -> Dict[str, float]:
        """
        Calculate weights based on model confidence scores
        
        Args:
            model_results: Dictionary of model results
        
        Returns:
            Dictionary of normalized weights
        """
        # Weight by confidence scores
        confidences = {name: res['confidence'] for name, res in model_results.items()}
        
        # Square confidences to emphasize differences
        squared_conf = {name: conf**1.5 for name, conf in confidences.items()}
        
        # Normalize
        total = sum(squared_conf.values())
        weights = {name: conf/total for name, conf in squared_conf.items()}
        
        return weights
    
    def _combine_predictions(self, model_results: Dict, weights: Dict, days: int) -> List[Dict]:
        """
        Combine predictions from multiple models using weighted average
        
        Args:
            model_results: Dictionary of model results
            weights: Dictionary of model weights
            days: Forecast horizon
        
        Returns:
            List of ensemble predictions
        """
        ensemble_predictions = []
        
        for day_idx in range(days):
            # Weighted average of predictions
            predicted_sum = 0
            lower_sum = 0
            upper_sum = 0
            date_str = None
            
            for model_name, result in model_results.items():
                weight = weights[model_name]
                pred = result['predictions'][day_idx]
                
                predicted_sum += pred['predicted'] * weight
                lower_sum += pred['lower'] * weight
                upper_sum += pred['upper'] * weight
                date_str = pred['date']
            
            ensemble_predictions.append({
                'date': date_str,
                'predicted': round(predicted_sum, 2),
                'lower': round(lower_sum, 2),
                'upper': round(upper_sum, 2)
            })
        
        return ensemble_predictions
    
    def _calculate_ensemble_confidence(self, model_results: Dict, weights: Dict) -> float:
        """
        Calculate overall ensemble confidence
        
        Args:
            model_results: Dictionary of model results
            weights: Dictionary of model weights
        
        Returns:
            Ensemble confidence score
        """
        # Weighted average of confidences
        confidence = sum(
            weights[name] * result['confidence']
            for name, result in model_results.items()
        )
        
        # Boost confidence if multiple models agree
        if len(model_results) > 1:
            confidence *= 1.1  # 10% boost for ensemble
        
        return min(95, confidence)  # Cap at 95%
    
    def _calculate_agreement_score(self, model_results: Dict, days: int) -> float:
        """
        Calculate how much models agree with each other
        
        Args:
            model_results: Dictionary of model results
            days: Forecast horizon
        
        Returns:
            Agreement score (0-100)
        """
        if len(model_results) < 2:
            return 100.0  # Perfect agreement if only one model
        
        # Compare final predictions
        final_predictions = [
            res['predictions'][-1]['predicted']
            for res in model_results.values()
        ]
        
        # Calculate coefficient of variation (lower = more agreement)
        mean_pred = np.mean(final_predictions)
        std_pred = np.std(final_predictions)
        
        if mean_pred == 0:
            return 50.0
        
        cv = std_pred / mean_pred
        
        # Convert to 0-100 score (lower CV = higher score)
        # CV < 0.02 = excellent agreement (>90)
        # CV > 0.10 = poor agreement (<50)
        agreement = max(0, min(100, 100 - (cv * 500)))
        
        return agreement
    
    def _determine_ensemble_trend(self, predictions: List[Dict]) -> str:
        """
        Determine overall trend from ensemble predictions
        
        Args:
            predictions: List of ensemble predictions
        
        Returns:
            Trend direction string
        """
        first_price = predictions[0]['predicted']
        last_price = predictions[-1]['predicted']
        change_percent = ((last_price - first_price) / first_price) * 100
        
        if change_percent > 2:
            return "upward"
        elif change_percent < -2:
            return "downward"
        else:
            return "neutral"
    
    def update_weights(self, backtest_results: Dict):
        """
        Update model weights based on backtest performance
        
        Args:
            backtest_results: Dictionary with model performance metrics
        """
        # Calculate weights based on inverse of error rates
        errors = {name: results['avg_error'] for name, results in backtest_results.items()}
        
        # Inverse errors (lower error = higher weight)
        inverse_errors = {name: 1 / (error + 1) for name, error in errors.items()}
        
        # Normalize
        total = sum(inverse_errors.values())
        self.weights = {name: inv_err / total for name, inv_err in inverse_errors.items()}
        
        logger.info(f"Updated weights based on backtest: {self.weights}")
    
    def get_best_model(self, data: pd.DataFrame, days: int = 7, 
                       test_splits: int = 3) -> str:
        """
        Determine which single model performs best for this stock
        
        Args:
            data: Historical price data
            days: Forecast horizon for testing
            test_splits: Number of validation splits
        
        Returns:
            Name of best performing model
        """
        logger.info("Running model comparison to find best model...")
        
        models_to_test = {
            'prophet': self.prophet,
            'xgboost': self.xgboost
        }
        
        model_scores = {}
        
        for model_name, model in models_to_test.items():
            errors = []
            
            for split in range(test_splits):
                try:
                    # Walk-forward validation
                    test_size = days
                    split_offset = split * days
                    train_end = len(data) - test_size - split_offset
                    
                    if train_end < 252:
                        continue
                    
                    train_data = data.iloc[:train_end]
                    test_data = data.iloc[train_end:train_end + test_size]
                    
                    if len(test_data) < days:
                        continue
                    
                    # Make prediction
                    if model_name == 'prophet':
                        result = model.forecast(train_data, days, model_type='prophet')
                    else:
                        result = model.forecast(train_data, days)
                    
                    # Calculate error
                    predicted_price = result['predictions'][-1]['predicted']
                    actual_price = test_data['Close'].iloc[-1]
                    error = abs(predicted_price - actual_price) / actual_price
                    errors.append(error)
                    
                except Exception as e:
                    logger.warning(f"Error testing {model_name} split {split}: {str(e)}")
                    continue
            
            if errors:
                avg_error = np.mean(errors)
                model_scores[model_name] = avg_error
                logger.info(f"{model_name}: avg error = {avg_error:.4f}")
        
        if not model_scores:
            logger.warning("Could not compare models, using prophet as default")
            return 'prophet'
        
        best_model = min(model_scores, key=model_scores.get)
        logger.info(f"Best model: {best_model} with error {model_scores[best_model]:.4f}")
        
        return best_model

