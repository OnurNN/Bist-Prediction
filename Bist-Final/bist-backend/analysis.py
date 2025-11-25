import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from utils.data_fetcher import fetch_stock_data

logger = logging.getLogger(__name__)

class ForecastAnalyzer:
    """Advanced analysis tools for stock forecasting"""
    
    def __init__(self):
        pass
        
    def analyze_errors(self, results: List[Dict]) -> Dict:
        """
        Analyze prediction errors to find patterns
        
        Args:
            results: List of backtest results
            
        Returns:
            Dictionary with error analysis
        """
        if not results:
            return {}
            
        df = pd.DataFrame(results)
        
        analysis = {
            'avg_error': df['error_percent'].mean(),
            'median_error': df['error_percent'].median(),
            'max_error': df['error_percent'].max(),
            'std_error': df['error_percent'].std(),
            'direction_accuracy': df['direction_correct'].mean() * 100,
            'error_distribution': {
                'under_1_percent': (df['error_percent'] < 1).mean() * 100,
                'under_3_percent': (df['error_percent'] < 3).mean() * 100,
                'under_5_percent': (df['error_percent'] < 5).mean() * 100,
                'over_10_percent': (df['error_percent'] > 10).mean() * 100
            }
        }
        
        return analysis
        
    def analyze_volatility_impact(self, stock_id: str, days: int = 30) -> Dict:
        """
        Analyze how volatility affects forecast accuracy
        
        Args:
            stock_id: Stock symbol
            days: Lookback period
            
        Returns:
            Volatility analysis
        """
        data = fetch_stock_data(stock_id, period="1y")
        
        if data is None:
            return {}
            
        # Calculate volatility
        returns = data['Close'].pct_change()
        volatility = returns.rolling(window=days).std() * np.sqrt(252)
        
        current_volatility = volatility.iloc[-1]
        avg_volatility = volatility.mean()
        
        regime = "High" if current_volatility > avg_volatility * 1.2 else \
                 "Low" if current_volatility < avg_volatility * 0.8 else "Normal"
                 
        return {
            'current_volatility': round(current_volatility * 100, 2),
            'average_volatility': round(avg_volatility * 100, 2),
            'regime': regime,
            'forecast_difficulty': "Hard" if regime == "High" else "Easy" if regime == "Low" else "Medium"
        }

if __name__ == "__main__":
    analyzer = ForecastAnalyzer()
    print("Analyzer ready.")
