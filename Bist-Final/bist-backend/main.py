from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import logging

from models.forecasting import StockForecaster
from models.ensemble import EnsembleForecaster
from utils.data_fetcher import fetch_stock_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="BIST ML Service", version="1.0.0")

# Allow frontend and n8n to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5678", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ForecastRequest(BaseModel):
    stockId: str
    days: Optional[int] = 14
    model: Optional[str] = "auto"  # auto, prophet, xgboost, lstm, arima, ensemble
    useGridSearch: Optional[bool] = False

class ForecastResponse(BaseModel):
    success: bool
    stockSymbol: str
    currentPrice: float
    predictedPrice: float
    priceChange: float
    changePercent: float
    confidence: float
    timeHorizon: str
    predictions: List[dict]
    metadata: dict
    error: Optional[str] = None
    individualModels: Optional[Dict] = None
    modelWeights: Optional[Dict] = None
    agreementScore: Optional[float] = None

class ModelComparisonRequest(BaseModel):
    stockId: str
    days: Optional[int] = 7
    models: Optional[List[str]] = ["prophet", "xgboost", "ensemble"]

class ModelComparisonResponse(BaseModel):
    success: bool
    stockSymbol: str
    results: Dict
    bestModel: str
    metadata: dict

class AnalysisRequest(BaseModel):
    stockId: str
    days: Optional[int] = 30

class AnalysisResponse(BaseModel):
    success: bool
    stockSymbol: str
    volatility: Dict
    metadata: dict

forecaster = StockForecaster()
ensemble_forecaster = EnsembleForecaster()

@app.get("/")
async def root():
    return {"status": "healthy", "service": "BIST ML Service"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": True}

@app.post("/api/forecast", response_model=ForecastResponse)
async def forecast_stock(request: ForecastRequest):
    try:
        logger.info(f"Forecasting {request.stockId} for {request.days} days with model: {request.model}")
        
        # Fetch historical data (2 years for better model training)
        historical_data = fetch_stock_data(request.stockId, period="2y")
        
        if historical_data is None or len(historical_data) == 0:
            raise HTTPException(status_code=404, detail=f"No data for {request.stockId}")
        
        # Auto model selection
        if request.model == "auto":
            logger.info("Auto-selecting best model...")
            best_model = ensemble_forecaster.get_best_model(historical_data, days=request.days)
            logger.info(f"Selected model: {best_model}")
            request.model = best_model
        
        # Generate forecast
        if request.model == "ensemble":
            forecast_result = ensemble_forecaster.forecast(
                data=historical_data,
                days=request.days,
                models=['prophet', 'xgboost']
            )
        elif request.model in ["arima", "xgboost", "lstm"]:
            # Use specific ML model
            from models.ml_models import ARIMAForecaster, XGBoostForecaster, LSTMForecaster
            
            if request.model == "arima":
                model = ARIMAForecaster()
            elif request.model == "xgboost":
                model = XGBoostForecaster()
            else:  # lstm
                model = LSTMForecaster()
            
            forecast_result = model.forecast(historical_data, days=request.days)
        else:
            # Use Prophet (default)
            forecast_result = forecaster.forecast(
                data=historical_data,
                days=request.days,
                model_type='prophet',
                use_grid_search=request.useGridSearch
            )
        
        # Calculate metrics
        current_price = historical_data['Close'].iloc[-1]
        predicted_price = forecast_result['predictions'][-1]['predicted']
        price_change = predicted_price - current_price
        change_percent = (price_change / current_price) * 100
        
        # Prepare response
        response = ForecastResponse(
            success=True,
            stockSymbol=request.stockId.replace('.IS', ''),
            currentPrice=round(current_price, 2),
            predictedPrice=round(predicted_price, 2),
            priceChange=round(price_change, 2),
            changePercent=round(change_percent, 2),
            confidence=round(forecast_result['confidence'], 2),
            timeHorizon=f"{request.days} days",
            predictions=forecast_result['predictions'],
            metadata={
                "model": forecast_result.get('model', request.model),
                "historicalDays": len(historical_data),
                "lastUpdate": historical_data.index[-1].strftime("%Y-%m-%d"),
                "dataSource": "Yahoo Finance",
                "trend": forecast_result.get('trend', 'unknown')
            }
        )
        
        # Add ensemble-specific fields
        if request.model == "ensemble":
            response.individualModels = forecast_result.get('individual_models')
            response.modelWeights = forecast_result.get('weights')
            response.agreementScore = forecast_result.get('agreement_score')
        
        return response
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return ForecastResponse(
            success=False, stockSymbol=request.stockId.replace('.IS', ''),
            currentPrice=0.0, predictedPrice=0.0, priceChange=0.0,
            changePercent=0.0, confidence=0.0, timeHorizon="",
            predictions=[], metadata={}, error=str(e)
        )

@app.post("/api/compare-models", response_model=ModelComparisonResponse)
async def compare_models(request: ModelComparisonRequest):
    """
    Compare performance of different models for a stock
    """
    try:
        logger.info(f"Comparing models for {request.stockId}")
        
        # Fetch historical data
        historical_data = fetch_stock_data(request.stockId, period="2y")
        
        if historical_data is None or len(historical_data) == 0:
            raise HTTPException(status_code=404, detail=f"No data for {request.stockId}")
        
        results = {}
        errors = {}
        
        # Test each model
        for model_name in request.models:
            try:
                logger.info(f"Testing {model_name}...")
                
                if model_name == "ensemble":
                    forecast_result = ensemble_forecaster.forecast(
                        historical_data,
                        days=request.days,
                        models=['prophet', 'xgboost']
                    )
                elif model_name in ["arima", "xgboost", "lstm"]:
                    from models.ml_models import ARIMAForecaster, XGBoostForecaster, LSTMForecaster
                    
                    if model_name == "arima":
                        model = ARIMAForecaster()
                    elif model_name == "xgboost":
                        model = XGBoostForecaster()
                    else:
                        model = LSTMForecaster()
                    
                    forecast_result = model.forecast(historical_data, days=request.days)
                else:  # prophet
                    forecast_result = forecaster.forecast(
                        historical_data,
                        days=request.days,
                        model_type='prophet'
                    )
                
                current_price = historical_data['Close'].iloc[-1]
                predicted_price = forecast_result['predictions'][-1]['predicted']
                
                results[model_name] = {
                    'predicted_price': round(predicted_price, 2),
                    'change_percent': round(((predicted_price - current_price) / current_price) * 100, 2),
                    'confidence': round(forecast_result['confidence'], 2),
                    'model': forecast_result.get('model', model_name)
                }
                
            except Exception as e:
                logger.error(f"Error testing {model_name}: {str(e)}")
                errors[model_name] = str(e)
                continue
        
        if not results:
            raise HTTPException(status_code=500, detail="All models failed")
        
        # Determine best model (highest confidence)
        best_model = max(results.items(), key=lambda x: x[1]['confidence'])[0]
        
        return ModelComparisonResponse(
            success=True,
            stockSymbol=request.stockId.replace('.IS', ''),
            results=results,
            bestModel=best_model,
            metadata={
                "testedModels": len(results),
                "failedModels": len(errors),
                "errors": errors,
                "currentPrice": round(historical_data['Close'].iloc[-1], 2)
            }
        )
        
    except Exception as e:
        logger.error(f"Error in model comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_stock(request: AnalysisRequest):
    """
    Analyze stock volatility and forecast difficulty
    """
    try:
        from analysis import ForecastAnalyzer
        analyzer = ForecastAnalyzer()
        
        logger.info(f"Analyzing {request.stockId}...")
        
        volatility_analysis = analyzer.analyze_volatility_impact(request.stockId, days=request.days)
        
        if not volatility_analysis:
            raise HTTPException(status_code=404, detail=f"Could not analyze {request.stockId}")
            
        return AnalysisResponse(
            success=True,
            stockSymbol=request.stockId.replace('.IS', ''),
            volatility=volatility_analysis,
            metadata={
                "analysisType": "volatility_impact",
                "lookbackDays": request.days
            }
        )
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def list_models():
    """
    List available forecasting models
    """
    return {
        "models": [
            {
                "name": "auto",
                "description": "Automatically select the best model for the stock",
                "recommended": True
            },
            {
                "name": "prophet",
                "description": "Facebook Prophet with technical indicators",
                "speed": "fast",
                "accuracy": "good"
            },
            {
                "name": "xgboost",
                "description": "XGBoost gradient boosting with features",
                "speed": "medium",
                "accuracy": "very good"
            },
            {
                "name": "lstm",
                "description": "LSTM neural network for sequence prediction",
                "speed": "slow",
                "accuracy": "good"
            },
            {
                "name": "arima",
                "description": "Classical ARIMA/SARIMAX time series model",
                "speed": "medium",
                "accuracy": "good"
            },
            {
                "name": "ensemble",
                "description": "Combines Prophet + XGBoost with weighted voting",
                "speed": "medium",
                "accuracy": "best",
                "recommended": True
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("🚀 BIST ML Service is starting...")
    print("="*50)
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Alternative docs: http://localhost:8000/redoc")
    print("❤️  Health check: http://localhost:8000/health")
    print("="*50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)