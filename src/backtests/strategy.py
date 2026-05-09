import numpy as np
import pandas as pd


def run_trading_strategy(df, preds, initial_capital=100, threshold=0.0, cost=0.001):
    """
    Proper trading backtest with:
    - no lookahead bias
    - position management
    - transaction costs
    - portfolio tracking
    """

    df = df.copy().reset_index(drop=True)
    preds = pd.Series(preds).reset_index(drop=True)

    # =====================
    # CLEAN
    # =====================
    preds = preds.clip(-10, 10)
    actual = df["target"].clip(-10, 10)   # use log returns target

    # =====================
    # SIGNAL
    # =====================
    signal = np.where(preds > threshold, 1,
              np.where(preds < -threshold, -1, 0))

    # =====================
    # POSITION (shifted!)
    # =====================
    position = pd.Series(signal).shift(1).fillna(0)

    # =====================
    # RETURNS
    # =====================
    strategy_returns = position * actual

    # =====================
    # COSTS
    # =====================
    trades = position.diff().abs()
    strategy_returns -= trades * cost

    # =====================
    # PORTFOLIO
    # =====================
    cumulative = np.exp(strategy_returns.cumsum())
    portfolio = initial_capital * cumulative

    result = pd.DataFrame({
        "Pred": preds,
        "Actual": actual,
        "Signal": signal,
        "Position": position,
        "Strategy_Return": strategy_returns,
        "Portfolio": portfolio
    })

    return result