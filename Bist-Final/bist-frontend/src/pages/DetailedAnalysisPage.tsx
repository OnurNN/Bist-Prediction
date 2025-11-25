import { useLocation, useNavigate } from 'react-router-dom';
import { ArrowLeft, TrendingUp, Newspaper, LineChart, Brain, Calendar, RefreshCw } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { useState } from 'react';

interface LocationState {
  stockData: any[];
  selectedStock: string;
  selectedStockInfo?: {
    name: string;
    sector: string;
  };
  metaData: any;
  prediction: any;
  technicalIndicators: any;
  aiAnalysis: any;
}

const DetailedAnalysisPage = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const state = location.state as LocationState;

  // State management for sentiment data
  const [sentimentData, setSentimentData] = useState<any>(null);
  const [sentimentLoading, setSentimentLoading] = useState(false);
  const [sentimentError, setSentimentError] = useState<string | null>(null);
  const N8N_SENTIMENT_WEBHOOK = 'http://localhost:5678/webhook/sentiment_analysis';

  // State management for forecast data
  const [forecastData, setForecastData] = useState<any>(null);
  const [forecastLoading, setForecastLoading] = useState(false);
  const [forecastError, setForecastError] = useState<string | null>(null);
  const N8N_FORECAST_WEBHOOK = 'http://localhost:5678/webhook/forecast';

  // State management for AI analysis
  const [aiAnalysis, setAiAnalysis] = useState<any>(null);
  const [aiAnalysisLoading, setAiAnalysisLoading] = useState(false);
  const [aiAnalysisError, setAiAnalysisError] = useState<string | null>(null);
  const N8N_AI_ANALYSIS_WEBHOOK = 'http://localhost:5678/webhook/detailed_ai_analysis';

  // If no data, redirect back
  if (!state || !state.stockData) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 p-6 flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-white mb-4">No data available</h1>
          <button
            onClick={() => navigate('/')}
            className="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
          >
            Go Back Home
          </button>
        </div>
      </div>
    );
  }

  const { selectedStock, selectedStockInfo, metaData, stockData } = state;
  const stockSymbol = selectedStock.replace('.IS', '');
  const currentPrice = metaData?.currentPrice || 0;

  // Fetch sentiment data from n8n workflow
  const fetchSentimentData = async () => {
    setSentimentLoading(true);
    setSentimentError(null);

    try {
      console.log('Fetching sentiment data for:', selectedStock);

      const response = await fetch(N8N_SENTIMENT_WEBHOOK, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stockId: selectedStock }) // Already in "GARAN.IS" format
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Parse n8n response (array format)
      const result = await response.json();
      console.log('Received sentiment data:', result);

      const data = Array.isArray(result) ? result[0] : result;

      // Transform n8n data to component format
      const transformedData = {
        overall: data.OVERALL_SENTIMENT || 'NEUTRAL',
        score: parseFloat(data.AVG_SENTIMENT_SCORE) || 0.5,
        positive: data.POSITIVE_PERCENT || 0,
        neutral: data.NEUTRAL_PERCENT || 0,
        negative: data.NEGATIVE_PERCENT || 0,
        totalArticles: data.TOTAL_NEWS || 0,
        positiveCount: data.POSITIVE_COUNT || 0,
        neutralCount: data.NEUTRAL_COUNT || 0,
        negativeCount: data.NEGATIVE_COUNT || 0,
        articles: (() => {
          try {
            const parsed = JSON.parse(data.LATEST_3_NEWS || '[]');
            return parsed.map((news: any) => ({
              title: news.title || 'No title',
              sentiment: news.sentiment || 'neutral',
              score: news.score || 0.5
            }));
          } catch (e) {
            console.error('Failed to parse LATEST_3_NEWS:', e);
            return [];
          }
        })()
      };

      console.log('Transformed sentiment data:', transformedData);
      setSentimentData(transformedData);
    } catch (err) {
      console.error('Error fetching sentiment data:', err);
      setSentimentError(err instanceof Error ? err.message : 'Unknown error occurred');
    } finally {
      setSentimentLoading(false);
    }
  };

  // Fetch forecast data from n8n workflow
  const fetchForecastData = async () => {
    setForecastLoading(true);
    setForecastError(null);

    try {
      console.log('Fetching forecast data for:', selectedStock);

      const response = await fetch(N8N_FORECAST_WEBHOOK, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stockId: selectedStock })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      console.log('Received forecast data:', result);

      // Handle array wrapping from n8n
      const data = Array.isArray(result) ? result[0] : result;

      if (!data.success && !data.predictions) {
        console.error('Forecast data validation failed. Received:', data);
        throw new Error(data.error || `Invalid forecast data received: ${JSON.stringify(data)}`);
      }

      // Prepare chart data: Historical + Forecast
      const historicalPoints = stockData.slice(-30).map((d: any) => ({
        date: d.date,
        actual: d.close,
        predicted: null,
        lower: null,
        upper: null
      }));

      const forecastPoints = (data.predictions || []).map((p: any) => ({
        date: p.date,
        actual: null,
        predicted: p.predicted,
        lower: p.lower,
        upper: p.upper
      }));

      // Connect the last historical point to the first forecast point
      if (historicalPoints.length > 0 && forecastPoints.length > 0) {
        const lastHistorical = historicalPoints[historicalPoints.length - 1];
        // Add a bridge point to make the line continuous
        forecastPoints.unshift({
          date: lastHistorical.date,
          actual: null,
          predicted: lastHistorical.actual, // Start prediction from last actual
          lower: lastHistorical.actual,
          upper: lastHistorical.actual
        });
      }

      const combinedData = [...historicalPoints, ...forecastPoints];

      setForecastData({
        predicted: data.predictedPrice,
        change: data.changePercent,
        confidence: data.confidence,
        timeHorizon: data.timeHorizon || '14 days',
        data: combinedData,
        model: data.metadata?.model || 'Ensemble'
      });

    } catch (err) {
      console.error('Error fetching forecast data:', err);
      setForecastError(err instanceof Error ? err.message : 'Unknown error occurred');
    } finally {
      setForecastLoading(false);
    }
  };

  // Fetch AI analysis from n8n workflow
  const fetchAiAnalysis = async () => {
    setAiAnalysisLoading(true);
    setAiAnalysisError(null);

    try {
      console.log('Fetching AI analysis for:', selectedStock);

      const response = await fetch(N8N_AI_ANALYSIS_WEBHOOK, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stockId: selectedStock })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      console.log('Received AI analysis:', result);

      const data = Array.isArray(result) ? result[0] : result;

      if (data.success) {
        setAiAnalysis(data);
      } else {
        throw new Error(data.error || 'AI analysis failed');
      }
    } catch (err) {
      console.error('Error fetching AI analysis:', err);
      setAiAnalysisError(err instanceof Error ? err.message : 'Unknown error occurred');
    } finally {
      setAiAnalysisLoading(false);
    }
  };

  // Use forecast data if available, otherwise use mock (or empty state)
  // Initial state is null, so we can show a "Generate" state
  const displayForecast = forecastData || {
    predicted: 0,
    change: 0,
    confidence: 0,
    timeHorizon: '14 days',
    data: stockData.slice(-30).map((d: any) => ({ date: d.date, actual: d.close, predicted: null }))
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 p-6">
      <div className="max-w-7xl mx-auto">

        {/* Header Section */}
        <div className="mb-8">
          <button
            onClick={() => navigate('/')}
            className="flex items-center gap-2 text-blue-300 hover:text-blue-200 mb-4 transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
            <span>Back to Home</span>
          </button>

          <div className="bg-white/10 backdrop-blur-lg rounded-2xl shadow-lg p-6 border border-white/20">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-4xl font-bold text-white mb-2">
                  Detailed Analysis: {stockSymbol}
                </h1>
                <p className="text-xl text-blue-200">{selectedStockInfo?.name || stockSymbol}</p>
                <div className="flex items-center gap-4 mt-4">
                  <span className="text-3xl font-bold text-white">
                    {metaData?.currency || 'TRY'} {currentPrice.toFixed(2)}
                  </span>
                  <span className="px-3 py-1 bg-blue-500/20 text-blue-300 rounded-full text-sm">
                    {selectedStockInfo?.sector || 'Financial Services'}
                  </span>
                </div>
              </div>
              <div className="text-right">
                <div className="text-sm text-blue-300 mb-2">Analysis Date</div>
                <div className="flex items-center gap-2 text-white">
                  <Calendar className="w-5 h-5" />
                  <span>{new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* News Sentiment Analysis Section */}
        <div className="bg-white/10 backdrop-blur-lg rounded-2xl shadow-lg p-6 border border-white/20 mb-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-2">
              <Newspaper className="w-6 h-6 text-blue-400" />
              <h2 className="text-2xl font-bold text-white"> News Analysis</h2>
            </div>
            <button
              onClick={fetchSentimentData}
              disabled={sentimentLoading}
              className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <RefreshCw className={`w-4 h-4 ${sentimentLoading ? 'animate-spin' : ''}`} />
              {sentimentLoading ? 'Analyzing...' : (sentimentData ? 'Refresh Analysis' : 'Generate Sentiment Analysis')}
            </button>
          </div>

          {/* Error Display */}
          {sentimentError && (
            <div className="mb-4 p-3 bg-red-500/20 border border-red-400/50 rounded-lg">
              <p className="text-sm text-red-200">
                Sentiment analysis failed.
                <br />
                <span className="text-xs">Error: {sentimentError}</span>
              </p>
            </div>
          )}

          {!sentimentData && !sentimentLoading && !sentimentError && (
            <div className="text-center py-12 bg-white/5 rounded-xl border border-white/10">
              <Newspaper className="w-12 h-12 text-blue-400 mx-auto mb-3 opacity-50" />
              <h3 className="text-xl font-semibold text-white mb-2">Ready to Analyze</h3>
              <p className="text-blue-200 mb-6 max-w-md mx-auto">
                Generate sentiment analysis from recent news articles about {stockSymbol}.
              </p>
              <button
                onClick={fetchSentimentData}
                className="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors font-semibold"
              >
                Generate Sentiment Analysis
              </button>
            </div>
          )}

          {/* Overall Sentiment Score */}
          {sentimentData && (
          <>
          <div className="grid md:grid-cols-4 gap-4 mb-6">
            <div className="md:col-span-1 bg-gradient-to-br from-green-500/20 to-emerald-500/20 rounded-xl border border-green-400/30 p-6">
              <div className="text-sm text-green-200 mb-2">
                Overall Sentiment ({sentimentData.totalArticles} news)
              </div>
              <div className={`text-3xl font-bold mb-2 ${sentimentData.overall === 'POSITIVE' ? 'text-green-300' :
                sentimentData.overall === 'NEGATIVE' ? 'text-red-300' :
                  'text-yellow-300'
                }`}>
                {sentimentData.overall === 'POSITIVE' ? '😊' : sentimentData.overall === 'NEGATIVE' ? '😟' : '😐'} {sentimentData.overall}
              </div>
              <div className="text-sm text-green-300">Score: {sentimentData.score.toFixed(2)}/1.00</div>
            </div>

            <div className="p-6 bg-gradient-to-br from-green-500/20 to-green-600/20 rounded-xl border border-green-400/30">
              <div className="text-sm text-green-200 mb-2">
                Positive ({sentimentData.positiveCount} news)
              </div>
              <div className="text-3xl font-bold text-white mb-1">{sentimentData.positive}%</div>
              <div className="text-xs text-green-300">Based on news articles</div>
            </div>

            <div className="p-6 bg-gradient-to-br from-yellow-500/20 to-yellow-600/20 rounded-xl border border-yellow-400/30">
              <div className="text-sm text-yellow-200 mb-2">
                Neutral ({sentimentData.neutralCount} news)
              </div>
              <div className="text-3xl font-bold text-white mb-1">{sentimentData.neutral}%</div>
              <div className="text-xs text-yellow-300">Informational</div>
            </div>

            <div className="p-6 bg-gradient-to-br from-red-500/20 to-red-600/20 rounded-xl border border-red-400/30">
              <div className="text-sm text-red-200 mb-2">
                Negative ({sentimentData.negativeCount} news)
              </div>
              <div className="text-3xl font-bold text-white mb-1">{sentimentData.negative}%</div>
              <div className="text-xs text-red-300">Risk factors</div>
            </div>
          </div>

          {/* Recent News */}
          <div className="bg-white/5 rounded-xl p-5">
            <h3 className="text-lg font-semibold text-white mb-4">Recent News:</h3>
            <div className="space-y-3">
              {sentimentData.articles.map((article: any, idx: number) => (
                <div key={idx} className="flex items-start gap-3 p-3 bg-white/5 rounded-lg hover:bg-white/10 transition-colors">
                  <div className={`mt-1 ${article.sentiment === 'positive' ? 'text-green-400' :
                    article.sentiment === 'negative' ? 'text-red-400' :
                      'text-yellow-400'
                    }`}>
                    {article.sentiment === 'positive' ? '✓' : article.sentiment === 'negative' ? '✗' : '−'}
                  </div>
                  <div className="flex-1">
                    <p className="text-white">{article.title}</p>
                    <div className="flex items-center gap-2 mt-1">
                      <span className={`text-xs px-2 py-1 rounded ${article.sentiment === 'positive' ? 'bg-green-500/20 text-green-300' :
                        article.sentiment === 'negative' ? 'bg-red-500/20 text-red-300' :
                          'bg-yellow-500/20 text-yellow-300'
                        }`}>
                        {article.sentiment.toUpperCase()}
                      </span>
                      <span className="text-xs text-blue-300">Score: {article.score.toFixed(2)}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
          </>
          )}

        </div>

        {/* Time Series Forecast Section */}
        <div className="bg-white/10 backdrop-blur-lg rounded-2xl shadow-lg p-6 border border-white/20 mb-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-2">
              <LineChart className="w-6 h-6 text-purple-400" />
              <h2 className="text-2xl font-bold text-white"> AI Price Forecast</h2>
            </div>
            <button
              onClick={fetchForecastData}
              disabled={forecastLoading}
              className="flex items-center gap-2 px-4 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <RefreshCw className={`w-4 h-4 ${forecastLoading ? 'animate-spin' : ''}`} />
              {forecastLoading ? 'Generating Forecast...' : 'Generate New Forecast'}
            </button>
          </div>

          {/* Error Display */}
          {forecastError && (
            <div className="mb-4 p-3 bg-red-500/20 border border-red-400/50 rounded-lg">
              <p className="text-sm text-red-200">
                Forecast generation failed.
                <br />
                <span className="text-xs">Error: {forecastError}</span>
              </p>
            </div>
          )}

          {!forecastData && !forecastLoading && !forecastError && (
            <div className="text-center py-12 bg-white/5 rounded-xl border border-white/10">
              <LineChart className="w-12 h-12 text-purple-400 mx-auto mb-3 opacity-50" />
              <h3 className="text-xl font-semibold text-white mb-2">Ready to Forecast</h3>
              <p className="text-blue-200 mb-6 max-w-md mx-auto">
                Generate a 14-day price prediction using our advanced ensemble model.
              </p>
              <button
                onClick={fetchForecastData}
                className="px-6 py-3 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition-colors font-semibold"
              >
                Generate Forecast
              </button>
            </div>
          )}

          {/* Forecast Summary Cards */}
          {forecastData && (
            <>
              <div className="grid md:grid-cols-3 gap-4 mb-6">
                <div className="p-6 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-xl border border-purple-400/30">
                  <div className="text-sm text-purple-200 mb-2">Predicted Price</div>
                  <div className="text-3xl font-bold text-white mb-1">
                    {metaData?.currency || 'TRY'} {displayForecast.predicted.toFixed(2)}
                  </div>
                  <div className="flex items-center gap-1 text-sm text-purple-300">
                    <TrendingUp className="w-4 h-4" />
                    {displayForecast.change > 0 ? '+' : ''}{displayForecast.change}% expected
                  </div>
                </div>

                <div className="p-6 bg-gradient-to-br from-blue-500/20 to-cyan-500/20 rounded-xl border border-blue-400/30">
                  <div className="text-sm text-blue-200 mb-2">Confidence Level</div>
                  <div className="text-3xl font-bold text-white mb-1">{displayForecast.confidence}%</div>
                  <div className="text-sm text-blue-300">Model: {displayForecast.model || 'Ensemble'}</div>
                </div>

                <div className="p-6 bg-gradient-to-br from-indigo-500/20 to-purple-500/20 rounded-xl border border-indigo-400/30">
                  <div className="text-sm text-indigo-200 mb-2">Time Horizon</div>
                  <div className="text-2xl font-bold text-white mb-1">{displayForecast.timeHorizon}</div>
                  <div className="text-sm text-indigo-300">Forecast period</div>
                </div>
              </div>

              {/* Forecast Chart */}
              <div className="bg-white/5 rounded-xl p-5">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-white">Interactive Forecast Chart</h3>
                  <div className="flex gap-3 text-xs">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-blue-500 rounded"></div>
                      <span className="text-blue-200">Historical prices (solid line)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-purple-500 rounded border-2 border-dashed"></div>
                      <span className="text-purple-200">Forecasted prices (dashed line)</span>
                    </div>
                  </div>
                </div>

                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={displayForecast.data}>
                    <defs>
                      <linearGradient id="colorActual" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8} />
                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                      </linearGradient>
                      <linearGradient id="colorPredicted" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#a855f7" stopOpacity={0.8} />
                        <stop offset="95%" stopColor="#a855f7" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis
                      dataKey="date"
                      stroke="rgba(255,255,255,0.5)"
                      tick={{ fill: 'rgba(255,255,255,0.7)', fontSize: 11 }}
                      tickFormatter={(value) => {
                        const date = new Date(value);
                        return `${date.getMonth() + 1}/${date.getDate()}`;
                      }}
                    />
                    <YAxis
                      stroke="rgba(255,255,255,0.5)"
                      tick={{ fill: 'rgba(255,255,255,0.7)', fontSize: 11 }}
                      domain={['auto', 'auto']}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'rgba(0, 0, 0, 0.9)',
                        border: '1px solid rgba(59, 130, 246, 0.5)',
                        borderRadius: '8px',
                        color: 'white'
                      }}
                    />
                    <Area
                      type="monotone"
                      dataKey="actual"
                      stroke="#3b82f6"
                      strokeWidth={3}
                      fill="url(#colorActual)"
                      name="Historical"
                    />
                    <Area
                      type="monotone"
                      dataKey="predicted"
                      stroke="#a855f7"
                      strokeWidth={3}
                      strokeDasharray="5 5"
                      fill="url(#colorPredicted)"
                      name="Forecast"
                    />
                  </AreaChart>
                </ResponsiveContainer>

                <div className="mt-4 p-3 bg-blue-500/10 border border-blue-400/30 rounded-lg">
                  <p className="text-xs text-blue-200">
                    <strong>Confidence interval:</strong> The shaded area represents the range of possible price movements based on historical volatility patterns.
                  </p>
                </div>
              </div>
            </>
          )}
        </div>

        {/* Combined AI Investment Analysis */}
        {forecastData && sentimentData && (
          <div className="bg-gradient-to-br from-purple-500/20 to-blue-500/20 backdrop-blur-lg rounded-2xl shadow-lg p-6 border border-purple-400/30 mb-6">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-2">
                <Brain className="w-6 h-6 text-purple-400" />
                <h2 className="text-2xl font-bold text-white"> Detailed AI Investment Analysis</h2>
              </div>
              <button
                onClick={fetchAiAnalysis}
                disabled={aiAnalysisLoading}
                className="flex items-center gap-2 px-4 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <RefreshCw className={`w-4 h-4 ${aiAnalysisLoading ? 'animate-spin' : ''}`} />
                {aiAnalysisLoading ? 'Generating Analysis...' : (aiAnalysis ? 'Refresh Analysis' : 'Generate AI Analysis')}
              </button>
            </div>

            {/* Error Display */}
            {aiAnalysisError && (
              <div className="mb-4 p-3 bg-red-500/20 border border-red-400/50 rounded-lg">
                <p className="text-sm text-red-200">
                  AI analysis generation failed.
                  <br />
                  <span className="text-xs">Error: {aiAnalysisError}</span>
                </p>
              </div>
            )}

            {!aiAnalysis && !aiAnalysisLoading && !aiAnalysisError && (
              <div className="text-center py-12 bg-white/5 rounded-xl border border-white/10">
                <Brain className="w-12 h-12 text-purple-400 mx-auto mb-3 opacity-50" />
                <h3 className="text-xl font-semibold text-white mb-2">Ready to Analyze</h3>
                <p className="text-blue-200 mb-6 max-w-md mx-auto">
                  Generate a comprehensive AI-powered investment analysis combining sentiment and forecast data for {stockSymbol}.
                </p>
                <button
                  onClick={fetchAiAnalysis}
                  className="px-6 py-3 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition-colors font-semibold"
                >
                  Generate AI Analysis
                </button>
              </div>
            )}

            {aiAnalysis && (
              <div className="bg-white/10 rounded-xl p-6">
                <div className="mb-4 p-4 bg-white/10 rounded-xl">
                  <p className="text-base text-blue-100 leading-relaxed">
                    {aiAnalysis.analysis.summary}
                  </p>
                </div>

                <div className="bg-white/5 p-4 rounded-lg mb-4">
                  <p className="font-semibold text-white mb-3">💡 Key Investment Points:</p>
                  <ul className="space-y-2 ml-4">
                    {aiAnalysis.analysis.keyPoints.map((point: string, idx: number) => (
                      <li key={idx} className="flex items-start gap-2">
                        <span className="text-purple-400 mt-1">•</span>
                        <span className="text-blue-100 text-sm">{point}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="bg-yellow-500/10 p-4 rounded-lg border border-yellow-400/30 mb-4">
                  <p className="font-semibold text-yellow-200 mb-3">⚠️ Risk Factors:</p>
                  <ul className="space-y-2 ml-4">
                    {aiAnalysis.analysis.riskFactors.map((risk: string, idx: number) => (
                      <li key={idx} className="flex items-start gap-2">
                        <span className="text-yellow-400 mt-1">•</span>
                        <span className="text-yellow-100 text-sm">{risk}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="bg-blue-500/10 p-4 rounded-lg border border-blue-400/30 mb-4">
                  <p className="font-semibold text-blue-200 mb-2">📊 Investment Strategy:</p>
                  <p className="text-blue-100 text-sm leading-relaxed">
                    {aiAnalysis.analysis.investmentStrategy}
                  </p>
                </div>

                <div className="flex items-center justify-between p-4 bg-gradient-to-r from-green-500/20 to-emerald-500/20 border border-green-400/30 rounded-xl">
                  <div>
                    <div className="text-sm text-green-200 mb-1">AI Recommendation</div>
                    <div className="text-2xl font-bold text-green-300 flex items-center gap-2">
                      <TrendingUp className="w-6 h-6" />
                      {aiAnalysis.recommendation}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm text-green-200 mb-1">Combined Confidence</div>
                    <div className="text-2xl font-bold text-white">{aiAnalysis.combinedConfidence}%</div>
                    <div className="text-xs text-green-300 mt-1 max-w-xs">{aiAnalysis.analysis.confidenceReasoning}</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Disclaimer */}
        <div className="bg-yellow-500/20 border border-yellow-400/50 rounded-xl p-5">
          <p className="text-xs text-yellow-100 leading-relaxed">
            <strong className="text-yellow-200">⚠️ Important Disclaimer:</strong> This detailed analysis combines sentiment analysis
            of news articles and time series forecasting models for informational and educational purposes only. It should NOT be considered
            as financial advice or investment recommendations. Market conditions can change rapidly, and past performance does not guarantee
            future results. The sentiment scores and price forecasts are based on historical data and AI models that may have limitations.
            Always conduct your own research, consider your risk tolerance and financial situation, and consult with qualified financial
            advisors before making any investment decisions. The creators of this tool are not responsible for any financial losses
            incurred based on this analysis.
          </p>
        </div>

      </div>
    </div>
  );
};

export default DetailedAnalysisPage;
