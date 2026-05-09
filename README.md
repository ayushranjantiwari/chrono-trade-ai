# ChronoTrade AI  
### Multi-Model Time Series Forecasting & Trading System with Walk-Forward Validation

ChronoTrade AI is an end-to-end financial time series forecasting and trading framework built using classical statistical models, machine learning models, ensemble forecasting, walk-forward validation, and trading strategy backtesting.

The project focuses not only on prediction accuracy, but also on evaluating whether predictions can generate profitable trading signals under realistic market conditions.

---

# 🚀 Features

## 📊 Forecasting Models
Implemented multiple forecasting approaches:

- ARIMA
- SARIMA
- SARIMAX
- Prophet
- Random Forest Regressor
- XGBoost Regressor

---

## ⚙️ Feature Engineering
Custom engineered time-series features including:

- Rolling volatility
- High-low spread
- Volume change
- Moving average ratios
- Price momentum
- Lag-based features

---

## 📈 Evaluation Metrics
Model performance evaluated using:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- Direction Accuracy

---

## 🔄 Walk-Forward Validation
Implemented realistic walk-forward validation to simulate real-world forecasting conditions:

- Expanding window retraining
- Sequential prediction
- Out-of-sample evaluation

---

## 💰 Trading Strategy Backtesting
Prediction outputs converted into trading signals and evaluated using:

- Portfolio growth simulation
- Strategy returns
- Sharpe Ratio
- Maximum Drawdown
- Buy & Hold comparison

---

## 🧠 Ensemble Forecasting
Combined predictions from multiple models using ensemble averaging to improve robustness and stability.

---

## 💾 Model Persistence
Implemented model saving/loading pipeline using Joblib:

- Save trained models
- Load production-ready models
- Final retraining on full dataset

---

# 📂 Project Structure

```text
chrono-trade-ai/
│
├── src/
│   ├── backtests/
│   │   ├── walk_forward.py
│   │   ├── strategy.py
│   │   └── backtest.py
│   │
│   ├── data/
│   │   ├── load_data.py
│   │   └── split_data.py
│   │
│   ├── evaluation/
│   │   └── metrics.py
│   │
│   ├── features/
│   │   └── build_features.py
│   │
│   ├── models/
│   │   ├── arima.py
│   │   ├── sarima.py
│   │   ├── sarimax.py
│   │   ├── prophet_model.py
│   │   └── ml_models.py
│   │
│   ├── pipeline/
│   │   ├── pipeline.py
│   │   └── predict_future.py
│   │
│   └── utils/
│       └── model_io.py
│
├── notebooks/
│   ├── walk_forward_analysis.ipynb
│   └── trading_strategy_analysis.ipynb
│
├── models/
├── results/
├── requirements.txt
├── README.md
└── main.py