import pandas as pd
import numpy as np
from typing import Dict, List
from models.forecasting import StockForecaster
from models.ensemble import EnsembleForecaster
from utils.data_fetcher import fetch_stock_data
from datetime import datetime, timedelta
import logging
import sys
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# BIST 30 stocks for multi-stock testing
BIST30_STOCKS = [
    'THYAO.IS', 'AKBNK.IS', 'GARAN.IS', 'ISCTR.IS', 'KCHOL.IS',
    'EREGL.IS', 'PETKM.IS', 'SISE.IS', 'SAHOL.IS', 'TCELL.IS',
    'TTKOM.IS', 'TUPRS.IS', 'YKBNK.IS', 'ARCLK.IS', 'ASELS.IS'
]

def backtest_model(stock_id: str, forecast_days: int = 7, test_periods: int = 6):
    """
    Backtest forecasting model by testing on historical data
    """
    print(f"\n{'='*60}")
    print(f"BACKTESTING: {stock_id} ({forecast_days}-day forecasts)")
    print(f"{'='*60}\n")
    
    # Fetch full historical data
    full_data = fetch_stock_data(stock_id, period="3y")
    
    if full_data is None:
        print(f"[ERROR] Could not fetch data for {stock_id}")
        return
    
    print(f"[OK] Fetched {len(full_data)} days of historical data")
    print(f"   Date range: {full_data.index[0].date()} to {full_data.index[-1].date()}\n")
    
    forecaster = StockForecaster()
    results = []
    
    # Test multiple periods
    for i in range(test_periods):
        # Calculate cutoff date (go back forecast_days * (i+1) from the end)
        days_back = forecast_days * (i + 1) + forecast_days
        cutoff_idx = len(full_data) - days_back
        
        if cutoff_idx < 252:  # Need at least 1 year of training data
            continue
            
        # Split data: training vs actual
        train_data = full_data.iloc[:cutoff_idx]
        actual_future = full_data.iloc[cutoff_idx:cutoff_idx + forecast_days]
        
        if len(actual_future) < forecast_days:
            continue
            
        print(f"Test Period #{i+1}:")
        print(f"  Training data: {train_data.index[0].date()} to {train_data.index[-1].date()}")
        print(f"  Predicting:    {actual_future.index[0].date()} to {actual_future.index[-1].date()}")
        
        try:
            # Make prediction using training data only
            forecast_result = forecaster.forecast(train_data, days=forecast_days, model_type="prophet")
            
            # Get predicted price for the last day
            predicted_price = forecast_result['predictions'][-1]['predicted']
            
            # Get actual price at the last day
            actual_price = actual_future['Close'].iloc[-1]
            
            # Calculate accuracy
            prediction_error = abs(predicted_price - actual_price)
            error_percent = (prediction_error / actual_price) * 100
            
            # Direction accuracy
            train_last_price = train_data['Close'].iloc[-1]
            predicted_direction = "UP" if predicted_price > train_last_price else "DOWN"
            actual_direction = "UP" if actual_price > train_last_price else "DOWN"
            direction_correct = predicted_direction == actual_direction
            
            result = {
                'period': i + 1,
                'forecast_days': forecast_days,
                'train_last_date': train_data.index[-1].date(),
                'train_last_price': train_last_price,
                'predicted_price': predicted_price,
                'actual_price': actual_price,
                'error_percent': error_percent,
                'predicted_direction': predicted_direction,
                'actual_direction': actual_direction,
                'direction_correct': direction_correct,
                'confidence': forecast_result['confidence']
            }
            results.append(result)
            
            # Print result
            print(f"  Last training price: {train_last_price:.2f} TRY")
            print(f"  Predicted ({forecast_days}d):     {predicted_price:.2f} TRY")
            print(f"  Actual ({forecast_days}d):        {actual_price:.2f} TRY")
            print(f"  Error:               {error_percent:.2f}%")
            print(f"  Direction:           {'[CORRECT]' if direction_correct else '[WRONG]'} "
                  f"(Pred: {predicted_direction}, Act: {actual_direction})")
            print(f"  Model confidence:    {forecast_result['confidence']:.1f}%\n")
            
        except Exception as e:
            print(f"  [ERROR]: {str(e)}\n")
            continue
    
    # Summary statistics
    if results:
        print(f"\n{'='*60}")
        print(f"SUMMARY STATISTICS ({forecast_days}-DAY FORECASTS)")
        print(f"{'='*60}\n")
        
        avg_error = sum(r['error_percent'] for r in results) / len(results)
        direction_accuracy = sum(1 for r in results if r['direction_correct']) / len(results) * 100
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        
        print(f"Tests performed:       {len(results)}")
        print(f"Average price error:   {avg_error:.2f}%")
        print(f"Direction accuracy:    {direction_accuracy:.1f}%")
        print(f"Average confidence:    {avg_confidence:.1f}%")
        
        # Rating
        if avg_error < 5 and direction_accuracy > 60:
            print(f"\n[EXCELLENT MODEL!]")
        elif avg_error < 10 and direction_accuracy > 50:
            print(f"\n[GOOD MODEL]")
        elif avg_error < 15:
            print(f"\n[ACCEPTABLE MODEL]")
        else:
            print(f"\n[POOR MODEL] - Consider different approach")
        
        print(f"\n{'='*60}\n")
        
        return {
            'results': results,
            'avg_error': avg_error,
            'direction_accuracy': direction_accuracy,
            'avg_confidence': avg_confidence
        }
    else:
        print("[ERROR] No valid test results")
        return None

def walk_forward_validation(stock_id: str, forecast_days: int = 7, 
                           n_splits: int = 10, model_type: str = 'prophet') -> Dict:
    """
    Perform walk-forward validation
    """
    print(f"\n{'='*60}")
    print(f"WALK-FORWARD VALIDATION: {stock_id}")
    print(f"Model: {model_type}, Horizon: {forecast_days} days, Splits: {n_splits}")
    print(f"{'='*60}\n")
    
    # Fetch data
    full_data = fetch_stock_data(stock_id, period="2y")
    if full_data is None:
        print(f"[ERROR] Could not fetch data for {stock_id}")
        return None
    
    print(f"[OK] Fetched {len(full_data)} days of data")
    
    if model_type == 'ensemble':
        forecaster = EnsembleForecaster()
    elif model_type == 'xgboost':
        from models.ml_models import XGBoostForecaster
        forecaster = XGBoostForecaster()
    else:
        forecaster = StockForecaster()
    
    results = []
    
    # Walk-forward splits
    for split in range(n_splits):
        split_offset = split * forecast_days
        train_end = len(full_data) - forecast_days - split_offset
        
        if train_end < 252:  # Need at least 1 year
            break
        
        train_data = full_data.iloc[:train_end]
        test_data = full_data.iloc[train_end:train_end + forecast_days]
        
        if len(test_data) < forecast_days:
            break
        
        print(f"\nSplit {split + 1}/{n_splits}:")
        print(f"  Train: {train_data.index[0].date()} to {train_data.index[-1].date()}")
        print(f"  Test:  {test_data.index[0].date()} to {test_data.index[-1].date()}")
        
        try:
            # Make prediction
            if model_type == 'ensemble':
                forecast_result = forecaster.forecast(train_data, days=forecast_days)
            elif model_type == 'xgboost':
                forecast_result = forecaster.forecast(train_data, days=forecast_days)
            else:
                forecast_result = forecaster.forecast(train_data, days=forecast_days, 
                                                     model_type=model_type)
            
            # Evaluate
            predicted_price = forecast_result['predictions'][-1]['predicted']
            actual_price = test_data['Close'].iloc[-1]
            train_last_price = train_data['Close'].iloc[-1]
            
            error = abs(predicted_price - actual_price) / actual_price
            
            # Direction accuracy
            predicted_direction = "UP" if predicted_price > train_last_price else "DOWN"
            actual_direction = "UP" if actual_price > train_last_price else "DOWN"
            direction_correct = predicted_direction == actual_direction
            
            results.append({
                'split': split + 1,
                'error': error,
                'error_percent': error * 100,
                'direction_correct': direction_correct,
                'confidence': forecast_result['confidence']
            })
            
            print(f"  Pred: {predicted_price:.2f}, Act: {actual_price:.2f}")
            print(f"  Error: {error*100:.2f}%, Direction: {'[CORRECT]' if direction_correct else '[WRONG]'}")
            
        except Exception as e:
            print(f"  [ERROR]: {str(e)}")
            continue
    
    # Summary
    if results:
        print(f"\n{'='*60}")
        print(f"WALK-FORWARD VALIDATION SUMMARY")
        print(f"{'='*60}\n")
        
        avg_error = np.mean([r['error_percent'] for r in results])
        direction_accuracy = sum(1 for r in results if r['direction_correct']) / len(results) * 100
        
        print(f"Splits completed:      {len(results)}/{n_splits}")
        print(f"Average error:         {avg_error:.2f}%")
        print(f"Direction accuracy:    {direction_accuracy:.1f}%")
        print(f"Model: {model_type}")
        
        return {
            'model': model_type,
            'stock': stock_id,
            'avg_error': avg_error,
            'direction_accuracy': direction_accuracy,
            'n_splits': len(results),
            'results': results
        }
    
    return None

def test_ensemble_models(stock_id: str, forecast_days: int = 7, test_periods: int = 6):
    """
    Test ensemble model vs individual models
    """
    print(f"\n{'='*60}")
    print(f"ENSEMBLE MODEL TESTING: {stock_id}")
    print(f"{'='*60}\n")
    
    # Test different configurations
    configurations = [
        ('prophet', 'Prophet Only'),
        ('xgboost', 'XGBoost Only'),
        ('ensemble', 'Ensemble (Prophet + XGBoost)')
    ]
    
    results_summary = {}
    
    for model_type, model_name in configurations:
        print(f"\n[TESTING]: {model_name}")
        print("-" * 60)
        
        result = walk_forward_validation(
            stock_id, 
            forecast_days=forecast_days,
            n_splits=test_periods,
            model_type=model_type
        )
        
        if result:
            results_summary[model_name] = result
    
    # Compare results
    if results_summary:
        print(f"\n{'='*60}")
        print(f"ENSEMBLE COMPARISON RESULTS")
        print(f"{'='*60}\n")
        print(f"{'Model':<30} {'Avg Error':<15} {'Direction Acc':<15} {'Rating'}")
        print("-" * 70)
        
        for model_name, result in results_summary.items():
            rating = "[BEST]" if result['avg_error'] == min(r['avg_error'] for r in results_summary.values()) else "[GOOD]"
            print(f"{model_name:<30} {result['avg_error']:.2f}%           {result['direction_accuracy']:.1f}%           {rating}")
        
        print("\n" + "="*60)
    
    return results_summary

def test_multiple_stocks(stock_ids: List[str] = None, forecast_days: int = 7, 
                        n_splits: int = 5, model_type: str = 'ensemble'):
    """
    Test model performance across multiple stocks
    """
    if stock_ids is None:
        stock_ids = BIST30_STOCKS[:5]  # Test first 5 stocks
    
    print(f"\n{'#'*60}")
    print(f"MULTI-STOCK TESTING")
    print(f"Testing {len(stock_ids)} stocks with {model_type} model")
    print(f"{'#'*60}\n")
    
    all_results = []
    
    for i, stock_id in enumerate(stock_ids, 1):
        print(f"\n[{i}/{len(stock_ids)}] Testing {stock_id}...")
        
        result = walk_forward_validation(
            stock_id,
            forecast_days=forecast_days,
            n_splits=n_splits,
            model_type=model_type
        )
        
        if result:
            all_results.append(result)
    
    # Aggregate results
    if all_results:
        print(f"\n{'='*60}")
        print(f"MULTI-STOCK AGGREGATE RESULTS")
        print(f"{'='*60}\n")
        
        avg_error = np.mean([r['avg_error'] for r in all_results])
        avg_direction_acc = np.mean([r['direction_accuracy'] for r in all_results])
        
        print(f"Stocks tested:   {len(all_results)}")
        print(f"Average error:   {avg_error:.2f}%")
        print(f"Average direction acc:   {avg_direction_acc:.1f}%")
        print(f"Model: {model_type}")
        
        # Best and worst performers
        best = min(all_results, key=lambda x: x['avg_error'])
        worst = max(all_results, key=lambda x: x['avg_error'])
        
        print(f"\nBest performer:  {best['stock']} ({best['avg_error']:.2f}% error)")
        print(f"Worst performer: {worst['stock']} ({worst['avg_error']:.2f}% error)")
        
        print("\n" + "="*60)
        
        return {
            'avg_error': avg_error,
            'avg_direction_accuracy': avg_direction_acc,
            'stocks_tested': len(all_results),
            'individual_results': all_results,
            'best_stock': best['stock'],
            'worst_stock': worst['stock']
        }
    
    return None

def analyze_feature_importance(stock_id: str):
    """Analyze which features are most important for XGBoost"""
    from models.ml_models import XGBoostForecaster
    
    print(f"\n{'='*60}")
    print(f"FEATURE IMPORTANCE ANALYSIS: {stock_id}")
    print(f"{'='*60}\n")
    
    # Fetch data
    data = fetch_stock_data(stock_id, period="2y")
    
    if data is None:
        print("[ERROR] Could not fetch data")
        return
        
    model = XGBoostForecaster()
    
    # Run grid search first
    print("Running grid search for optimal parameters...")
    model.grid_search_xgboost(data)
    
    # Train model
    print("Training model...")
    model.forecast(data, days=7)
    
    # Get importance
    importance = model.get_feature_importance()
    
    print(f"\nTop 20 Most Important Features:")
    print("-" * 40)
    
    for i, (feature, score) in enumerate(list(importance.items())[:20], 1):
        print(f"{i:<2}. {feature:<30} {score:.4f}")
        
    print("-" * 40)
    
    return importance

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='Stock Market Backtesting Tool')
    
    parser.add_argument('--mode', type=str, default='standard', 
                        choices=['standard', 'ensemble', 'multi', 'walk', 'analyze'],
                        help='Backtest mode: standard, ensemble, multi, walk, analyze')
    
    parser.add_argument('--stock', type=str, default='THYAO.IS',
                        help='Stock symbol to test (default: THYAO.IS)')
    
    parser.add_argument('--days', type=int, default=7,
                        help='Forecast horizon in days (default: 7)')
    
    parser.add_argument('--periods', type=int, default=6,
                        help='Number of test periods/splits (default: 6)')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Fix encoding for Windows
    if os.name == 'nt':
        sys.stdout.reconfigure(encoding='utf-8')
    
    args = parse_arguments()
    
    if args.mode == 'ensemble':
        # Test ensemble models
        print(f"\n[ENSEMBLE] ENSEMBLE MODEL TESTING MODE: {args.stock}")
        test_ensemble_models(args.stock, forecast_days=args.days, test_periods=args.periods)
    
    elif args.mode == 'multi':
        # Test multiple stocks
        print("\n[MULTI-STOCK] MULTI-STOCK TESTING MODE")
        stocks = [args.stock] if args.stock != 'THYAO.IS' else ['THYAO.IS', 'AKBNK.IS', 'GARAN.IS', 'TUPRS.IS', 'KCHOL.IS']
        test_multiple_stocks(
            stock_ids=stocks,
            forecast_days=args.days,
            n_splits=args.periods,
            model_type='ensemble'
        )
    
    elif args.mode == 'walk':
        # Walk-forward validation
        print(f"\n[WALK-FORWARD] WALK-FORWARD VALIDATION MODE: {args.stock}")
        walk_forward_validation(args.stock, forecast_days=args.days, n_splits=args.periods, model_type='prophet')
        
    elif args.mode == 'analyze':
        # Analyze feature importance
        print(f"\n[ANALYZE] FEATURE IMPORTANCE ANALYSIS: {args.stock}")
        analyze_feature_importance(args.stock)
    
    else:
        # Original comparison test (default)
        print("\n" + "="*60)
        print(f"IMPROVED MODEL TESTING WITH TECHNICAL INDICATORS: {args.stock}")
        print("="*60)
        print("\nUsage Examples:")
        print(f"  python backtest.py --stock {args.stock}              - Standard backtest (Prophet)")
        print(f"  python backtest.py --mode ensemble --stock {args.stock} - Ensemble backtest (Best Results)")
        print(f"  python backtest.py --mode analyze --stock {args.stock}  - Feature importance")
        print("\n" + "="*60)
        
        # Test 7-day forecasts with improved model
        print(f"\n[7-DAY] Testing 7-DAY forecasts with enhanced model for {args.stock}...")
        results_7d = backtest_model(args.stock, forecast_days=7, test_periods=args.periods)
        
        # Test 14-day forecasts
        print(f"\n[14-DAY] Testing 14-DAY forecasts for {args.stock}...")
        results_14d = backtest_model(args.stock, forecast_days=14, test_periods=args.periods)
        
        # Test 30-day forecasts
        print(f"\n[30-DAY] Testing 30-DAY forecasts for {args.stock}...")
        results_30d = backtest_model(args.stock, forecast_days=30, test_periods=args.periods)
        
        # Compare all
        if all([results_7d, results_14d, results_30d]):
            print("\n" + "="*60)
            print("COMPARISON SUMMARY - IMPROVED MODEL")
            print("="*60 + "\n")
            print(f"{'Window':<10} {'Avg Error':<15} {'Direction Acc':<20} {'Rating'}")
            print("-" * 60)
            print(f"7 days     {results_7d['avg_error']:.2f}%           {results_7d['direction_accuracy']:.1f}%              {'[BEST]' if results_7d['avg_error'] < results_14d['avg_error'] else '[GOOD]'}")
            print(f"14 days    {results_14d['avg_error']:.2f}%           {results_14d['direction_accuracy']:.1f}%              {'[BEST]' if results_14d['avg_error'] < results_7d['avg_error'] and results_14d['avg_error'] < results_30d['avg_error'] else '[GOOD]'}")
            print(f"30 days    {results_30d['avg_error']:.2f}%           {results_30d['direction_accuracy']:.1f}%              [HARDER]")
            print("\n" + "="*60)
