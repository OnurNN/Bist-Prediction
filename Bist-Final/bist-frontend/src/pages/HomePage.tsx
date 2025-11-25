import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { TrendingUp, TrendingDown, Minus, Search, BarChart3, Brain, Activity, RefreshCw } from 'lucide-react';

interface StockDataPoint {
  date: string;
  open: number;
  close: number;
  high: number;
  low: number;
  volume: number;
}

interface MetaData {
  symbol?: string;
  longName?: string;
  currentPrice: number;
  currency: string;
  dayHigh: number;
  dayLow: number;
  volume: number;
  fiftyTwoWeekLow?: number;
  fiftyTwoWeekHigh?: number;
}

interface Prediction {
  signal: string;
  confidence: number;
  summary: string;
  priceChange: number;
  reasons?: string[];
}

interface TechnicalIndicators {
  rsi: number | null;
  sma7?: number;
  sma30?: number;
  macd: number | null;
  avgVolume?: number;
}

interface AIAnalysis {
  summary: string;
  details: {
    ceo?: string;  // Optional - AI may not always provide CEO info
    sector: string;
    volume: string;
    profitability: string;
  };
  recommendation: string;
  reasoning: string;
}

const timeRanges = [
  { value: '1d', label: '1D' },
  { value: '5d', label: '5D' },
  { value: '1mo', label: '1M' },
  { value: '3mo', label: '3M' },
  { value: '6mo', label: '6M' },
  { value: '1y', label: '1Y' },
  { value: '5y', label: '5Y' },
  { value: '10y', label: '10Y' },
];

const HomePage = () => {
  const navigate = useNavigate();
  const [selectedStock, setSelectedStock] = useState('TUPRS.IS');
  const [customStock, setCustomStock] = useState('');
  const [interval, setInterval] = useState('1d');
  const [range, setRange] = useState('1mo');
  const [stockData, setStockData] = useState<StockDataPoint[]>([]);
  const [metaData, setMetaData] = useState<MetaData | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [technicalIndicators, setTechnicalIndicators] = useState<TechnicalIndicators | null>(null);
  const [aiAnalysis, setAiAnalysis] = useState<AIAnalysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showDetailedAnalysis, setShowDetailedAnalysis] = useState(false);

  // n8n webhook URL - UPDATE THIS WITH YOUR URL
  const N8N_WEBHOOK_URL = 'http://localhost:5678/webhook/stock-data';

  // BIST popular stocks with .IS suffix for Yahoo Finance
  const stocks = [
    { symbol: 'TUPRS.IS', name: 'Tüpraş', sector: 'Energy' },
    { symbol: 'THYAO.IS', name: 'Türk Hava Yolları', sector: 'Transportation' },
    { symbol: 'EREGL.IS', name: 'Ereğli Demir Çelik', sector: 'Manufacturing' },
    { symbol: 'AKBNK.IS', name: 'Akbank', sector: 'Banking' },
    { symbol: 'GARAN.IS', name: 'Garanti BBVA', sector: 'Banking' },
    { symbol: 'SAHOL.IS', name: 'Sabancı Holding', sector: 'Holding' },
    { symbol: 'KCHOL.IS', name: 'Koç Holding', sector: 'Holding' },
    { symbol: 'SISE.IS', name: 'Şişe Cam', sector: 'Manufacturing' },
    { symbol: 'PETKM.IS', name: 'Petkim', sector: 'Chemicals' },
    { symbol: 'TCELL.IS', name: 'Turkcell', sector: 'Telecommunications' }
  ];

  // Fetch real stock data from n8n workflow
  const fetchStockData = async (customRange?: string) => {
    setLoading(true);
    setError(null);

    // Use custom stock if provided, otherwise use selected stock
    let stockToFetch = customStock.trim() || selectedStock;
    
    // Auto-add .IS suffix if not present (for BIST stocks)
    if (stockToFetch && !stockToFetch.includes('.IS')) {
      stockToFetch = stockToFetch.toUpperCase() + '.IS';
    }

    // Use customRange if provided (from button click), otherwise use state
    const rangeToUse = customRange || range;

    try {
      console.log('Fetching data for:', stockToFetch, interval, rangeToUse);
      
      const response = await fetch(N8N_WEBHOOK_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          stockId: stockToFetch,
          interval: interval,
          range: rangeToUse
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      let result = await response.json();
      console.log('Received data:', result);

      // Handle array response (n8n may return array with multiple items)
      if (Array.isArray(result)) {
        console.log('⚠️ Response is an array, taking first item');
        result = result[0];
      }

      if (result.success) {
        // New workflow structure: data is under 'stock' object
        const stockData = result.stock?.historicalData || result.historicalData || [];
        const meta = result.stock?.meta || result.meta || null;
        const statistics = result.stock?.statistics || result.statistics || null;
        const aiAnalysis = result.aiAnalysis || null;
        
        // Keep old prediction/indicators for backward compatibility
        const prediction = result.aiPrediction || result.prediction || null;
        const technicalIndicators = result.technicalIndicators || statistics || null;
        
        console.log('✅ Setting stockData with', stockData.length, 'data points');
        if (stockData.length > 0) {
          console.log('First data point:', stockData[0]);
          console.log('First data point keys:', Object.keys(stockData[0]));
        }
        console.log('✅ Setting meta:', meta);
        console.log('✅ AI Analysis:', aiAnalysis ? 'Found ✅' : 'NOT FOUND ❌');
        if (aiAnalysis) {
          console.log('AI Recommendation:', aiAnalysis.recommendation);
          console.log('AI Summary:', aiAnalysis.summary);
        }
        
        setStockData(stockData);
        setMetaData(meta);
        setPrediction(prediction);
        setTechnicalIndicators(technicalIndicators);
        setAiAnalysis(aiAnalysis);
      } else {
        throw new Error(result.error || 'Failed to fetch data');
      }

    } catch (err) {
      console.error('Error fetching stock data:', err);
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
    } finally {
      setLoading(false);
    }
  };

  // Don't auto-fetch - user must click "Analyze" button
  // useEffect removed to match n8n form workflow pattern

  const getSignalColor = (signal: string | undefined) => {
    switch(signal?.toUpperCase()) {
      case 'BUY': return 'text-green-600 bg-green-50 border-green-300';
      case 'SELL': return 'text-red-600 bg-red-50 border-red-300';
      case 'HOLD': return 'text-yellow-600 bg-yellow-50 border-yellow-300';
      default: return 'text-gray-600 bg-gray-50 border-gray-300';
    }
  };

  const getSignalIcon = (signal: string | undefined) => {
    switch(signal?.toUpperCase()) {
      case 'BUY': return <TrendingUp className="w-6 h-6" />;
      case 'SELL': return <TrendingDown className="w-6 h-6" />;
      case 'HOLD': return <Minus className="w-6 h-6" />;
      default: return <Activity className="w-6 h-6" />;
    }
  };

  const activeStock = customStock.trim() || selectedStock;
  const activeStockFormatted = activeStock.includes('.IS') ? activeStock : activeStock + '.IS';
  const selectedStockInfo = stocks.find(s => s.symbol === activeStockFormatted);
  const latestPrice = metaData?.currentPrice || (stockData.length > 0 ? stockData[stockData.length - 1]?.close : 0) || 0;

  const firstDataPoint = stockData.length > 0 ? stockData[0] : null;
  const lastDataPoint = stockData.length > 0 ? stockData[stockData.length - 1] : null;
  const rangeStartPrice = firstDataPoint?.close ?? firstDataPoint?.open ?? null;
  const rangeEndPrice = lastDataPoint?.close ?? lastDataPoint?.open ?? null;
  const hasRangeData = typeof rangeStartPrice === 'number' && typeof rangeEndPrice === 'number' && rangeStartPrice !== 0;
  const rangeChangeValue = hasRangeData ? rangeEndPrice - rangeStartPrice : 0;
  const rangeChangePercent = hasRangeData ? (rangeChangeValue / rangeStartPrice) * 100 : 0;

  // Calculate high/low from historical data for the selected time range
  const periodHigh = stockData.length > 0 
    ? Math.max(...stockData.map(d => d.high || d.close || 0))
    : (metaData?.dayHigh || 0);
  
  const periodLow = stockData.length > 0 
    ? Math.min(...stockData.map(d => d.low || d.close || 0).filter(v => v > 0))
    : (metaData?.dayLow || 0);

  // Calculate total volume for the period
  const periodVolume = stockData.length > 0
    ? stockData.reduce((sum, d) => sum + (d.volume || 0), 0)
    : (metaData?.volume || 0);

  console.log(`Period stats for ${range}:`, { high: periodHigh, low: periodLow, volume: periodVolume, dataPoints: stockData.length });

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <BarChart3 className="w-10 h-10 text-blue-400" />
              <div>
                <h1 className="text-4xl font-bold text-white">BIST Stock Analyzer</h1>
                <p className="text-blue-200">AI-powered stock analysis for Borsa İstanbul</p>
              </div>
            </div>
            <button
              onClick={() => fetchStockData()}
              disabled={loading}
              className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors disabled:opacity-50"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </button>
          </div>
        </div>

        {/* Stock Selection & Controls - Single Card */}
        <div className="bg-white/10 backdrop-blur-lg rounded-2xl shadow-lg p-6 mb-6 border border-white/20">
          <div className="flex items-center gap-2 mb-4">
            <Search className="w-5 h-5 text-blue-300" />
            <h2 className="text-lg font-semibold text-white">Stock Analysis Search</h2>
          </div>
          
          {/* Stock Selection */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-blue-200 mb-3">Select Popular Stock</label>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
              {stocks.map((stock) => (
                <button
                  key={stock.symbol}
                  onClick={() => {
                    setSelectedStock(stock.symbol);
                    setCustomStock(''); // Clear custom input when clicking a predefined stock
                  }}
                  className={`p-3 rounded-lg border-2 transition-all ${
                    selectedStock === stock.symbol && !customStock
                      ? 'border-blue-400 bg-blue-500/30 shadow-lg'
                      : 'border-white/20 bg-white/5 hover:border-blue-300 hover:bg-white/10'
                  }`}
                >
                  <div className="font-bold text-white">{stock.symbol.replace('.IS', '')}</div>
                  <div className="text-xs text-blue-200 truncate">{stock.name}</div>
                  <div className="text-xs text-blue-300 mt-1">{stock.sector}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Custom Stock Input */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-blue-200 mb-3">Or Enter Any Stock Symbol</label>
            <div className="flex gap-3">
              <input
                type="text"
                value={customStock}
                onChange={(e) => setCustomStock(e.target.value.toUpperCase())}
                placeholder="e.g., ASELS, BIMAS, VESTL (will auto-add .IS)"
                className="flex-1 px-4 py-3 bg-white/20 border border-white/30 rounded-lg text-white placeholder-blue-300/50 focus:outline-none focus:ring-2 focus:ring-purple-400"
              />
              {customStock && (
                <button
                  onClick={() => setCustomStock('')}
                  className="px-4 py-3 bg-red-500/20 border border-red-400/50 text-red-300 rounded-lg hover:bg-red-500/30 transition-all"
                >
                  Clear
                </button>
              )}
            </div>
            {customStock && (
              <p className="mt-2 text-sm text-purple-300">
                Will search for: <strong>{customStock.includes('.IS') ? customStock : customStock + '.IS'}</strong>
              </p>
            )}
          </div>

          {/* Interval & Range Selection */}
          <div className="grid md:grid-cols-3 gap-4 mb-6">
            {/* Interval */}
            <div>
              <label className="block text-sm font-medium text-blue-200 mb-2">Interval</label>
              <select
                value={interval}
                onChange={(e) => setInterval(e.target.value)}
                className="w-full px-4 py-3 bg-white/20 border border-white/30 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-400"
                style={{ colorScheme: 'dark' }}
              >
                <option value="1d" style={{ backgroundColor: '#1e293b', color: 'white' }}>1 Day</option>
                <option value="1wk" style={{ backgroundColor: '#1e293b', color: 'white' }}>1 Week</option>
                <option value="1mo" style={{ backgroundColor: '#1e293b', color: 'white' }}>1 Month</option>
              </select>
            </div>

            {/* Range */}
            <div>
              <label className="block text-sm font-medium text-blue-200 mb-2">Range</label>
              <select
                value={range}
                onChange={(e) => setRange(e.target.value)}
                className="w-full px-4 py-3 bg-white/20 border border-white/30 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-400"
                style={{ colorScheme: 'dark' }}
              >
                <option value="1d" style={{ backgroundColor: '#1e293b', color: 'white' }}>1 Day</option>
                <option value="5d" style={{ backgroundColor: '#1e293b', color: 'white' }}>5 Days</option>
                <option value="1mo" style={{ backgroundColor: '#1e293b', color: 'white' }}>1 Month</option>
                <option value="3mo" style={{ backgroundColor: '#1e293b', color: 'white' }}>3 Months</option>
                <option value="6mo" style={{ backgroundColor: '#1e293b', color: 'white' }}>6 Months</option>
                <option value="1y" style={{ backgroundColor: '#1e293b', color: 'white' }}>1 Year</option>
                <option value="2y" style={{ backgroundColor: '#1e293b', color: 'white' }}>2 Years</option>
                <option value="5y" style={{ backgroundColor: '#1e293b', color: 'white' }}>5 Years</option>
                <option value="10y" style={{ backgroundColor: '#1e293b', color: 'white' }}>10 Years</option>
              </select>
            </div>

            {/* Search Button */}
            <div className="flex items-end">
              <button
                onClick={() => fetchStockData()}
                disabled={loading || !selectedStock}
                className="w-full px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-lg font-semibold hover:from-blue-600 hover:to-purple-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 shadow-lg"
              >
                {loading ? (
                  <>
                    <RefreshCw className="w-5 h-5 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <BarChart3 className="w-5 h-5" />
                    Analyze Stock
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Current Selection Display */}
          <div className="flex items-center justify-center gap-4 p-3 bg-white/5 rounded-lg border border-white/10">
            <span className="text-sm text-blue-200">
              Selected: <strong className="text-white">
                {customStock 
                  ? (customStock.includes('.IS') ? customStock.replace('.IS', '') : customStock)
                  : selectedStock.replace('.IS', '')}
              </strong>
            </span>
            <span className="text-blue-400">•</span>
            <span className="text-sm text-blue-200">
              Interval: <strong className="text-white">{interval}</strong>
            </span>
            <span className="text-blue-400">•</span>
            <span className="text-sm text-blue-200">
              Range: <strong className="text-white">{range}</strong>
            </span>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-500/20 border border-red-500/50 rounded-lg p-4 mb-6">
            <p className="text-red-200">
              <strong>Error:</strong> {error}
              <br />
              <span className="text-sm">Please check your n8n workflow is running at: {N8N_WEBHOOK_URL}</span>
            </p>
          </div>
        )}

        {/* Main Content Grid */}
        <div className="grid lg:grid-cols-3 gap-6">
          {/* Chart Section */}
          <div className="lg:col-span-2 bg-white/10 backdrop-blur-lg rounded-2xl shadow-lg p-6 border border-white/20">
            <div className="flex justify-between items-center mb-6">
      <div>
                <h2 className="text-3xl font-bold text-white">{activeStockFormatted.replace('.IS', '')}</h2>
                <p className="text-sm text-blue-200">{selectedStockInfo?.name || (customStock ? 'Custom Stock' : '')}</p>
                <div className="flex items-center gap-3 mt-2">
                  <span className="text-4xl font-bold text-white">
                    {metaData?.currency || 'TRY'} {latestPrice.toFixed(2)}
                  </span>
                  <span className={`flex items-center gap-1 text-sm font-semibold px-3 py-1 rounded-full ${
                    rangeChangePercent >= 0 ? 'bg-green-500/20 text-green-300' : 'bg-red-500/20 text-red-300'
                  }`}>
                    {rangeChangePercent >= 0 ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                    {rangeChangePercent >= 0 ? '+' : ''}{rangeChangePercent.toFixed(2)}%
                  </span>
                </div>
                {(metaData || stockData.length > 0) && (
                  <div className="flex gap-4 mt-3 text-sm text-blue-200">
                    <span className="flex items-center gap-1">
                      <span className="text-blue-400 font-semibold">{timeRanges.find(tr => tr.value === range)?.label}</span>
                      High: {periodHigh.toFixed(2)}
                    </span>
                    <span className="flex items-center gap-1">
                      <span className="text-blue-400 font-semibold">{timeRanges.find(tr => tr.value === range)?.label}</span>
                      Low: {periodLow.toFixed(2)}
                    </span>
                    <span>Vol: {(periodVolume / 1000000).toFixed(1)}M</span>
                  </div>
                )}
                {hasRangeData && (
                  <p className="text-xs text-blue-300 mt-2">
                    Change over {timeRanges.find(tr => tr.value === range)?.label} range:{' '}
                    {rangeChangeValue >= 0 ? '+' : ''}{rangeChangeValue.toFixed(2)} {metaData?.currency || 'TRY'} (
                    {rangeChangePercent >= 0 ? '+' : ''}{rangeChangePercent.toFixed(2)}%)
                  </p>
                )}
              </div>
              
              <div className="flex gap-2 flex-wrap">
                {timeRanges.map((tr) => (
                  <button
                    key={tr.value}
                    onClick={() => {
                      setRange(tr.value);
                      // Pass the new range value directly to fetchStockData
                      fetchStockData(tr.value);
                    }}
                    disabled={loading}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                      range === tr.value
                        ? 'bg-blue-500 text-white shadow-lg'
                        : 'bg-white/10 text-white hover:bg-white/20'
                    } disabled:opacity-50`}
                  >
                    {tr.label}
                  </button>
                ))}
              </div>
            </div>

            {loading ? (
              <div className="h-96 flex flex-col items-center justify-center">
                <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-blue-400 mb-4"></div>
                <p className="text-white">Analyzing {activeStockFormatted}...</p>
              </div>
            ) : stockData.length > 0 ? (
              <ResponsiveContainer width="100%" height={400}>
                <AreaChart data={stockData}>
                  <defs>
                    <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8}/>
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis 
                    dataKey="date" 
                    stroke="rgba(255,255,255,0.5)"
                    tick={{ fill: 'rgba(255,255,255,0.7)', fontSize: 12 }}
                    tickFormatter={(value) => {
                      const date = new Date(value);
                      return `${date.getMonth() + 1}/${date.getDate()}`;
                    }}
                  />
                  <YAxis 
                    stroke="rgba(255,255,255,0.5)"
                    tick={{ fill: 'rgba(255,255,255,0.7)', fontSize: 12 }}
                    domain={['auto', 'auto']}
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: 'rgba(0, 0, 0, 0.9)', 
                      border: '1px solid rgba(59, 130, 246, 0.5)',
                      borderRadius: '8px',
                      color: 'white'
                    }}
                    labelFormatter={(value) => new Date(value).toLocaleDateString('tr-TR')}
                    formatter={(value) => [`${typeof value === 'number' ? value.toFixed(2) : value} ${metaData?.currency || 'TRY'}`, 'Price']}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="close" 
                    stroke="#3b82f6" 
                    strokeWidth={3}
                    fill="url(#colorPrice)" 
                    name="Close Price"
                  />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-96 flex flex-col items-center justify-center text-white/50">
                <BarChart3 className="w-16 h-16 mb-4 opacity-50" />
                <p className="text-lg font-medium">No data loaded</p>
                <p className="text-sm">Select a stock and click "Analyze Stock" to view data</p>
              </div>
            )}
          </div>

          {/* Right Side Panels */}
          <div className="space-y-6">
            {/* Stock Info Card (AI Analysis) */}
            {aiAnalysis && (
              <div className="bg-white/10 backdrop-blur-lg rounded-2xl shadow-lg p-6 border border-white/20">
                <div className="flex items-center gap-2 mb-5">
                  <Brain className="w-5 h-5 text-purple-400" />
                  <h3 className="text-lg font-semibold text-white">Basic AI Analysis</h3>
                </div>

                {/* 1. Company Full Name */}
                <div className="mb-4">
                  <h4 className="text-xs font-semibold text-blue-300 mb-2 uppercase tracking-wide">Company Full Name</h4>
                  <p className="text-base font-semibold text-white">
                    {selectedStockInfo?.name || metaData?.longName || activeStockFormatted.replace('.IS', '')}
                  </p>
                </div>

                {/* 2. Company Overview */}
                <div className="mb-4">
                  <h4 className="text-xs font-semibold text-blue-300 mb-2 uppercase tracking-wide">Company Overview</h4>
                  <p className="text-sm text-blue-100 leading-relaxed bg-white/5 p-3 rounded-lg">
                    {aiAnalysis.summary}
                  </p>
                </div>

                {/* 3. Sector */}
                <div className="mb-4">
                  <h4 className="text-xs font-semibold text-blue-300 mb-2 uppercase tracking-wide">Sector</h4>
                  <p className="text-base font-semibold text-white bg-white/5 p-3 rounded-lg">
                    {aiAnalysis.details.sector}
                  </p>
                </div>

                {/* 4. Profitability */}
                <div className="mb-4">
                  <h4 className="text-xs font-semibold text-blue-300 mb-2 uppercase tracking-wide">Profitability</h4>
                  <p className="text-base font-semibold text-white bg-white/5 p-3 rounded-lg">
                    {aiAnalysis.details.profitability}
                  </p>
                </div>

                {/* 5. Recommendation */}
                <div className="mb-4">
                  <h4 className="text-xs font-semibold text-blue-300 mb-2 uppercase tracking-wide">Recommendation</h4>
                  <div className={`flex items-center justify-center gap-3 p-4 rounded-xl border-2 ${
                    aiAnalysis.recommendation.includes('BUY') 
                      ? 'bg-green-500/20 border-green-400 text-green-300'
                      : aiAnalysis.recommendation.includes('SELL')
                      ? 'bg-red-500/20 border-red-400 text-red-300'
                      : 'bg-yellow-500/20 border-yellow-400 text-yellow-300'
                  }`}>
                    {aiAnalysis.recommendation.includes('BUY') ? (
                      <TrendingUp className="w-6 h-6" />
                    ) : aiAnalysis.recommendation.includes('SELL') ? (
                      <TrendingDown className="w-6 h-6" />
                    ) : (
                      <Minus className="w-6 h-6" />
                    )}
                    <span className="text-lg font-bold">{aiAnalysis.recommendation}</span>
                  </div>
                </div>

                {/* 6. Investment Reasoning */}
                <div>
                  <h4 className="text-xs font-semibold text-blue-300 mb-2 uppercase tracking-wide">Investment Reasoning</h4>
                  <p className="text-sm text-blue-200 leading-relaxed bg-gradient-to-br from-purple-500/10 to-blue-500/10 border border-purple-400/30 p-4 rounded-lg">
                    {aiAnalysis.reasoning}
                  </p>
                </div>

                {/* Make Detailed Analysis Button */}
                <button
                  onClick={() => navigate('/detailed-analysis', { 
                    state: { 
                      stockData, 
                      selectedStock: activeStockFormatted,
                      selectedStockInfo,
                      metaData, 
                      prediction,
                      technicalIndicators,
                      aiAnalysis
                    }
                  })}
                  disabled={!stockData.length}
                  className="w-full mt-6 px-6 py-4 bg-gradient-to-r from-pink-600 to-purple-600 text-white rounded-xl hover:from-pink-700 hover:to-purple-700 transition-all font-semibold shadow-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-3 text-lg"
                >
                  <Brain className="w-6 h-6" />
                  Make Detailed Analysis
                </button>
              </div>
            )}

            {/* AI Signal */}
            {prediction && (
              <div className="bg-white/10 backdrop-blur-lg rounded-2xl shadow-lg p-6 border border-white/20">
                <div className="flex items-center gap-2 mb-4">
                  <Brain className="w-5 h-5 text-purple-400" />
                  <h3 className="text-lg font-semibold text-white">AI Recommendation</h3>
                </div>
                
                <div className={`flex items-center justify-center gap-3 p-5 rounded-xl border-2 mb-4 ${getSignalColor(prediction.signal)}`}>
                  {getSignalIcon(prediction.signal)}
                  <span className="text-3xl font-bold">{prediction.signal}</span>
                </div>

                <div className="space-y-3 mb-4">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-blue-200">Confidence</span>
                    <span className="font-semibold text-white">{prediction.confidence}%</span>
                  </div>
                  <div className="w-full bg-white/20 rounded-full h-3">
                    <div 
                      className="bg-gradient-to-r from-blue-500 to-purple-500 h-3 rounded-full transition-all duration-500"
                      style={{ width: `${prediction.confidence}%` }}
                    ></div>
                  </div>
                </div>

                <div className="p-4 bg-white/5 rounded-lg text-sm text-blue-100 leading-relaxed mb-4">
                  {prediction.summary}
                </div>

                {prediction.reasons && prediction.reasons.length > 0 && (
                  <div className="space-y-2 mb-4">
                    <h4 className="text-sm font-semibold text-white">Key Factors:</h4>
                    {prediction.reasons.map((reason, idx) => (
                      <div key={idx} className="flex items-start gap-2 text-xs text-blue-200">
                        <span className="text-blue-400 mt-0.5">•</span>
                        <span>{reason}</span>
                      </div>
                    ))}
      </div>
                )}

                <button
                  onClick={() => setShowDetailedAnalysis(!showDetailedAnalysis)}
                  className="w-full mt-4 px-4 py-3 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-lg hover:from-purple-700 hover:to-blue-700 transition-all font-medium shadow-lg"
                >
                  {showDetailedAnalysis ? 'Hide' : 'Show'} Technical Analysis
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Detailed Analysis Section */}
        {showDetailedAnalysis && prediction && technicalIndicators && (
          <div className="mt-6 bg-white/10 backdrop-blur-lg rounded-2xl shadow-lg p-6 border border-white/20">
            <h3 className="text-2xl font-bold text-white mb-6">📊 Detailed Technical Analysis</h3>
            
            <div className="grid md:grid-cols-4 gap-4 mb-6">
              {technicalIndicators.rsi !== null && (
                <div className="p-5 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-xl border border-blue-400/30">
                  <div className="text-sm text-blue-200 mb-2">RSI (14-day)</div>
                  <div className="text-3xl font-bold text-white mb-1">{technicalIndicators.rsi}</div>
                  <div className="text-xs text-blue-300">
                    {technicalIndicators.rsi > 70 ? 'Overbought Zone' : 
                     technicalIndicators.rsi < 30 ? 'Oversold Zone' : 'Neutral Range'}
                  </div>
                </div>
              )}

              {technicalIndicators.macd !== null && technicalIndicators.macd !== undefined && typeof technicalIndicators.macd === 'number' && (
                <div className="p-5 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-xl border border-purple-400/30">
                  <div className="text-sm text-purple-200 mb-2">MACD Signal</div>
                  <div className="text-3xl font-bold text-white mb-1">{technicalIndicators.macd.toFixed(4)}</div>
                  <div className="text-xs text-purple-300">
                    {technicalIndicators.macd > 0 ? 'Bullish Momentum' : 'Bearish Momentum'}
                  </div>
                </div>
              )}

              <div className="p-5 bg-gradient-to-br from-green-500/20 to-emerald-500/20 rounded-xl border border-green-400/30">
                <div className="text-sm text-green-200 mb-2">Price Change</div>
                <div className="text-3xl font-bold text-white mb-1">{prediction.priceChange}%</div>
                <div className="text-xs text-green-300">Over selected period</div>
              </div>

              <div className="p-5 bg-gradient-to-br from-orange-500/20 to-red-500/20 rounded-xl border border-orange-400/30">
                <div className="text-sm text-orange-200 mb-2">Confidence</div>
                <div className="text-3xl font-bold text-white mb-1">{prediction.confidence}%</div>
                <div className="text-xs text-orange-300">AI Model Score</div>
              </div>
            </div>

            <div className="p-6 bg-gradient-to-r from-purple-500/10 to-blue-500/10 rounded-xl border border-purple-400/30">
              <h4 className="text-lg font-semibold text-white mb-4">🤖 AI-Generated Comprehensive Analysis</h4>
              <div className="space-y-4 text-sm text-blue-100 leading-relaxed">
                <p>
                  <strong className="text-white">Time Series Analysis:</strong> {prediction.summary}
                </p>
                
                {prediction.reasons && prediction.reasons.length > 0 && (
                  <div>
                    <strong className="text-white">Technical Factors:</strong>
                    <ul className="mt-2 space-y-1 ml-4">
                      {prediction.reasons.map((reason, idx) => (
                        <li key={idx} className="flex items-start gap-2">
                          <span className="text-blue-400">•</span>
                          <span>{reason}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                <p>
                  <strong className="text-white">Combined Recommendation:</strong> Based on quantitative analysis of price patterns and technical indicators, the model generates a <strong className={`${prediction.signal === 'BUY' ? 'text-green-400' : prediction.signal === 'SELL' ? 'text-red-400' : 'text-yellow-400'}`}>{prediction.signal}</strong> signal with {prediction.confidence}% confidence. This recommendation considers multiple timeframes and market conditions.
                </p>

                {metaData && (
                  <p>
                    <strong className="text-white">Market Context:</strong> {selectedStockInfo?.name} is trading at {metaData.currentPrice?.toFixed(2)} {metaData.currency}
                    {metaData.fiftyTwoWeekLow && metaData.fiftyTwoWeekHigh && (
                      <>, with a 52-week range of {metaData.fiftyTwoWeekLow.toFixed(2)} - {metaData.fiftyTwoWeekHigh.toFixed(2)}</>
                    )}. Current volume suggests {(technicalIndicators?.avgVolume || 0) > metaData.volume ? 'below' : 'above'} average market activity.
                  </p>
                )}
              </div>
            </div>

            <div className="mt-6 p-4 bg-yellow-500/20 border border-yellow-400/50 rounded-xl">
              <p className="text-xs text-yellow-100 leading-relaxed">
                <strong className="text-yellow-200">⚠️ Disclaimer:</strong> This analysis is generated by an AI model for informational and educational purposes only. It should not be considered as financial advice or investment recommendations. Past performance does not guarantee future results. Always conduct your own research, consider your risk tolerance, and consult with qualified financial advisors before making any investment decisions. The creators of this tool are not responsible for any financial losses incurred based on this analysis.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default HomePage;
