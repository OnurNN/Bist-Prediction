# BIST Stock Analyzer

AI-powered stock analysis platform for Borsa Istanbul (BIST). Combines real-time market data with machine learning forecasts and technical indicators to help investors make informed decisions.
https://drive.google.com/drive/folders/1VAtRQzuJqUnclv1PmhNG0-M6KFnxG4Oa?usp=sharing ( Video)

![React](https://img.shields.io/badge/React-19-61DAFB?logo=react)
![TypeScript](https://img.shields.io/badge/TypeScript-5.9-3178C6?logo=typescript)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi)
![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python)

---

## Features

- **Real-Time Stock Data** – Fetches live prices, volume, and historical data for BIST stocks via n8n webhook integration.
- **Dynamic Range Analysis** – View performance over 1D, 5D, 1M, 3M, 6M, 1Y, 5Y, or 10Y with auto-calculated percentage changes.
- **AI Recommendations** – Basic AI analysis including sector info, profitability insights, and buy/hold/sell signals.
- **ML Forecasting** – Prophet, XGBoost, ARIMA, and ensemble models for price predictions.
- **Technical Indicators** – RSI, MACD, SMA, and volume analysis.
- **Detailed Analysis Page** – Deep-dive view with extended charts and model comparison.

---

## Tech Stack

### Frontend
- React 19 + TypeScript
- Vite
- Tailwind CSS
- Recharts
- React Router

### Backend
- FastAPI
- Prophet / XGBoost / scikit-learn / statsmodels
- Pandas / NumPy

### Integration
- n8n (workflow automation for data fetching)

---

## Getting Started

### Prerequisites

- Node.js 18+
- Python 3.11+
- n8n instance (for webhook data flow)

### Frontend

```bash
cd bist-frontend
npm install
npm run dev
```

The app runs at `http://localhost:5173`.

### Backend

```bash
cd bist-backend
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

API available at `http://localhost:8000`.

### n8n Webhook

Configure your n8n workflow to listen at:

```
POST http://localhost:5678/webhook/stock-data
```

The frontend sends `{ stockId, interval, range }` and expects historical data + AI analysis in return.

---

## Project Structure

```
Bist-Final/
├── bist-frontend/          # React + Vite frontend
│   ├── src/
│   │   ├── pages/
│   │   │   ├── HomePage.tsx
│   │   │   └── DetailedAnalysisPage.tsx
│   │   └── ...
│   └── package.json
├── bist-backend/           # FastAPI ML service
│   ├── main.py
│   ├── models/
│   │   ├── forecasting.py
│   │   ├── ensemble.py
│   │   └── ml_models.py
│   ├── utils/
│   │   ├── data_fetcher.py
│   │   └── technical_indicators.py
│   └── requirements.txt
└── README.md
```

---

## Screenshots

https://drive.google.com/drive/folders/1VAtRQzuJqUnclv1PmhNG0-M6KFnxG4Oa?usp=sharing ( Video) 

---

## Disclaimer

This tool is for **educational and informational purposes only**. It does not constitute financial advice. Always do your own research and consult a qualified financial advisor before making investment decisions.

---

## License

MIT

