import numpy as np
import pandas as pd


def simple_backtest(df, predictions):

    df = df.copy()

    # =====================
    # 1. ALIGN LENGTH
    # =====================
    preds = pd.Series(predictions, index=df.index)

    # =====================
    # 2. CONVERT log → price
    # =====================
    pred_price = np.exp(preds)

    # =====================
    # 3. GENERATE SIGNAL
    # =====================
    # Compare predicted next price vs current actual price
    df['signal'] = np.where(pred_price > df['Close'], 1, -1)

    # =====================
    # 4. STRATEGY RETURNS
    # =====================
    # use actual returns from your dataset
    df['strategy_returns'] = df['signal'] * df['returns']

    # =====================
    # 5. CUMULATIVE RETURNS
    # =====================
    cumulative = np.exp(df['strategy_returns'].cumsum())

    return cumulative