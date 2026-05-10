import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# =====================
# PATH SETUP
# =====================
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# =====================
# PROJECT IMPORTS
# =====================
from src.data.load_data import load_raw_data
from src.features.build_features import create_features
from src.data.split_data import train_test_split_time
from src.config import SPLIT_DATE
from src.utils.model_io import load_model

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="ChronoTrade AI",
    page_icon="📈",
    layout="wide",
)

st.title("ChronoTrade AI")
st.caption("Multi-model time series forecasting, testing, and trading simulation")

# =====================
# CONSTANTS
# =====================
MODEL_NAMES = ["ARIMA", "SARIMA", "SARIMAX", "Prophet", "RandomForest", "XGBoost"]
STOCK_LOADERS = {
    "Reliance": load_raw_data,
}
FORECAST_HORIZON = 5


# =====================
# HELPERS
# =====================
def clean_array(values):
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr)
    arr = np.clip(arr, -10, 10)
    return arr


def to_price(values, index=None):
    arr = np.exp(clean_array(values))
    if index is not None:
        return pd.Series(arr[:len(index)], index=index)
    return pd.Series(arr)


def load_stock_data(stock_name: str) -> pd.DataFrame:
    if stock_name not in STOCK_LOADERS:
        raise ValueError(f"Unknown stock: {stock_name}")

    df = STOCK_LOADERS[stock_name]()
    df = create_features(df)
    df = df.sort_index().dropna()
    return df


def get_model(model_name: str):
    return load_model(f"{model_name}_final")


def predict_test(model_name: str, model, test_df: pd.DataFrame):
    if model_name in ["ARIMA", "SARIMA"]:
        preds = model.predict(len(test_df))
    elif model_name in ["SARIMAX", "Prophet"]:
        preds = model.predict(test=test_df)
    elif model_name in ["RandomForest", "XGBoost"]:
        preds = model.predict(test_df)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return clean_array(preds)


def predict_future(model_name: str, model, full_df: pd.DataFrame):
    if model_name in ["ARIMA", "SARIMA"]:
        preds = model.predict(FORECAST_HORIZON)
    elif model_name in ["SARIMAX", "Prophet"]:
        preds = model.predict(test=full_df.tail(FORECAST_HORIZON), steps=FORECAST_HORIZON)
    elif model_name in ["RandomForest", "XGBoost"]:
        preds = model.predict_future(full_df, steps=FORECAST_HORIZON)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return clean_array(preds)


def ensemble_pred(pred_dict: dict):
    if not pred_dict:
        raise ValueError("No predictions available for ensemble.")
    preds_df = pd.DataFrame(pred_dict)
    return preds_df.mean(axis=1).to_numpy()


def eval_metrics(actual_log: pd.Series, pred_log) -> dict:
    actual_log = pd.Series(actual_log).reset_index(drop=True)
    pred_log = pd.Series(pred_log).reset_index(drop=True)

    n = min(len(actual_log), len(pred_log))
    actual_log = actual_log.iloc[:n]
    pred_log = pred_log.iloc[:n]

    actual_price = to_price(actual_log)
    pred_price = to_price(pred_log)

    mae = float(np.mean(np.abs(actual_price - pred_price)))
    rmse = float(np.sqrt(np.mean((actual_price - pred_price) ** 2)))

    direction = ((pred_price.diff() > 0) == (actual_price.diff() > 0)).mean()

    return {
        "MAE": mae,
        "RMSE": rmse,
        "Direction_Accuracy": float(direction),
    }


def plot_test_prediction(actual_price: pd.Series, pred_price: pd.Series, model_name: str):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=actual_price.index,
            y=actual_price.values,
            mode="lines",
            name="Actual",
            line=dict(color="#1f77b4", width=2),  # blue
        )
    )

    fig.add_trace(
        go.Scatter(
            x=pred_price.index,
            y=pred_price.values,
            mode="lines+markers",
            name=f"{model_name} Prediction",
            line=dict(color="#008080", width=2),  # teal
        )
    )

    fig.update_layout(
        template="plotly_white",
        height=520,
        title=f"{model_name} — Test Prediction",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Series",
    )
    return fig


def plot_future_prediction(pred_price: pd.Series, model_name: str):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=pred_price.index,
            y=pred_price.values,
            mode="lines+markers",
            name=f"{model_name} Forecast",
            line=dict(color="#2ca02c", width=3),  # green
        )
    )

    fig.update_layout(
        template="plotly_white",
        height=520,
        title=f"{model_name} — Future Forecast ({FORECAST_HORIZON} days)",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Series",
    )
    return fig


def investment_projection(pred_price: pd.Series, initial_investment: float, last_close: float):
    shares = initial_investment / last_close
    projected_value = shares * pred_price

    result = pd.DataFrame({
        "Date": pred_price.index,
        "Predicted Price": pred_price.values,
        "Projected Value": projected_value.values,
    })

    return result


# =====================
# SIDEBAR
# =====================
st.sidebar.header("Controls")
st.sidebar.info("Forecast horizon is fixed at 5 business days.")

stock_name = st.sidebar.selectbox("Select stock", list(STOCK_LOADERS.keys()), index=0)
selected_model = st.sidebar.selectbox("Select model", MODEL_NAMES + ["Ensemble"], index=0)
initial_investment = st.sidebar.number_input(
    "Initial investment (₹)",
    min_value=100.0,
    value=1000.0,
    step=100.0,
)


# =====================
# LOAD DATA
# =====================
try:
    df = load_stock_data(stock_name)
except Exception as e:
    st.error(f"Failed to load data for {stock_name}: {e}")
    st.stop()

train, test = train_test_split_time(df, SPLIT_DATE)

actual_test_price = to_price(test["target"])
full_price = to_price(df["target"])
last_close = float(full_price.iloc[-1])

st.subheader(f"Stock: {stock_name}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Train size", f"{len(train):,}")
c3.metric("Test size", f"{len(test):,}")
c4.metric("Last close", f"₹{last_close:,.2f}")


# =====================
# LOAD MODELS + GENERATE PREDICTIONS
# =====================
test_preds = {}
future_preds = {}
errors = {}

with st.spinner("Loading final models and generating predictions..."):
    for model_name in MODEL_NAMES:
        try:
            model = get_model(model_name)

            test_pred = predict_test(model_name, model, test)
            future_pred = predict_future(model_name, model, df)

            test_preds[model_name] = test_pred
            future_preds[model_name] = future_pred

        except Exception as e:
            errors[model_name] = str(e)

    if test_preds:
        try:
            test_preds["Ensemble"] = ensemble_pred(test_preds)
        except Exception as e:
            errors["Ensemble (test)"] = str(e)

    if future_preds:
        try:
            future_preds["Ensemble"] = ensemble_pred(future_preds)
        except Exception as e:
            errors["Ensemble (future)"] = str(e)

available_models = list(test_preds.keys())
if not available_models:
    st.error("No models could be loaded or predicted.")
    st.stop()

if selected_model not in available_models:
    st.warning(f"{selected_model} is not available. Switching to first available model.")
    selected_model = available_models[0]


# =====================
# METRICS TABLE
# =====================
metrics_rows = []
for name, pred_log in test_preds.items():
    metrics = eval_metrics(test["target"], pred_log)
    metrics_rows.append({"Model": name, **metrics})

metrics_df = pd.DataFrame(metrics_rows).sort_values("RMSE").reset_index(drop=True)


# =====================
# TABS
# =====================
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Model Performance",
    "Future Forecast",
    "Investment Simulator"
])


# =====================
# TAB 1 — OVERVIEW
# =====================
with tab1:
    st.subheader("Train vs Test split")

    split_fig = go.Figure()
    split_fig.add_trace(
        go.Scatter(
            x=train.index,
            y=to_price(train["target"]),
            mode="lines",
            name="Train",
            line=dict(color="#1f77b4", width=2),
        )
    )
    split_fig.add_trace(
        go.Scatter(
            x=test.index,
            y=actual_test_price,
            mode="lines",
            name="Test",
            line=dict(color="#2ca02c", width=2),
        )
    )
    split_fig.update_layout(
        template="plotly_white",
        height=500,
        title="Train vs Test",
        xaxis_title="Date",
        yaxis_title="Price",
    )
    st.plotly_chart(split_fig, use_container_width=True)

    st.subheader(f"{selected_model}: Actual vs prediction on test data")
    model_test_pred = pd.Series(
        to_price(test_preds[selected_model]),
        index=test.index[:len(test_preds[selected_model])]
    )

    test_fig = plot_test_prediction(
        actual_price=actual_test_price.iloc[:len(model_test_pred)],
        pred_price=model_test_pred,
        model_name=selected_model,
    )
    st.plotly_chart(test_fig, use_container_width=True)

    sel_metrics = eval_metrics(
        test["target"].iloc[:len(test_preds[selected_model])],
        test_preds[selected_model]
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{sel_metrics['MAE']:.4f}")
    m2.metric("RMSE", f"{sel_metrics['RMSE']:.4f}")
    m3.metric("Direction accuracy", f"{sel_metrics['Direction_Accuracy']:.2%}")

    if errors:
        with st.expander("Prediction / loading issues"):
            for k, v in errors.items():
                st.write(f"**{k}**: {v}")


# =====================
# TAB 2 — PERFORMANCE
# =====================
with tab2:
    st.subheader("Model comparison on test set")
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    perf_fig = go.Figure()
    perf_fig.add_trace(
        go.Bar(x=metrics_df["Model"], y=metrics_df["RMSE"], name="RMSE")
    )
    perf_fig.add_trace(
        go.Bar(x=metrics_df["Model"], y=metrics_df["MAE"], name="MAE")
    )
    perf_fig.update_layout(
        barmode="group",
        template="plotly_white",
        height=450,
        title="RMSE / MAE comparison",
    )
    st.plotly_chart(perf_fig, use_container_width=True)


# =====================
# TAB 3 — FUTURE FORECAST
# =====================
with tab3:
    st.subheader(f"{selected_model}: Next {FORECAST_HORIZON} business days forecast")

    future_dates = pd.bdate_range(start=df.index[-1], periods=FORECAST_HORIZON + 1)[1:]

    if selected_model not in future_preds:
        st.error(f"No future prediction available for {selected_model}")
    else:
        selected_future_pred = pd.Series(
            to_price(future_preds[selected_model]),
            index=future_dates[:len(future_preds[selected_model])]
        )

        # Only predictions here — no actual line
        future_fig = plot_future_prediction(selected_future_pred, selected_model)
        st.plotly_chart(future_fig, use_container_width=True)

        st.markdown("### Selected model forecast table")
        future_table = pd.DataFrame({
            "Date": selected_future_pred.index,
            "Forecast Price": selected_future_pred.values,
        })
        st.dataframe(future_table, use_container_width=True, hide_index=True)

    st.markdown("### All model future forecasts")
    if future_preds:
        all_future_df = pd.DataFrame(
            {
                name: to_price(preds)
                for name, preds in future_preds.items()
            },
            index=future_dates
        )
        st.dataframe(all_future_df, use_container_width=True)

        all_future_fig = go.Figure()
        for name in all_future_df.columns:
            all_future_fig.add_trace(
                go.Scatter(
                    x=all_future_df.index,
                    y=all_future_df[name],
                    mode="lines+markers",
                    name=name,
                )
            )
        all_future_fig.update_layout(
            template="plotly_white",
            height=500,
            title="All models — future forecast",
            xaxis_title="Date",
            yaxis_title="Price",
        )
        st.plotly_chart(all_future_fig, use_container_width=True)
    else:
        st.warning("No future forecasts were generated.")

# =====================
# TAB 4 — INVESTMENT SIMULATOR
# =====================
with tab4:
    st.subheader("Investment projection")

    if selected_model not in future_preds:
        st.error(f"No future prediction available for {selected_model}")
    else:
        selected_future_price = pd.Series(
            to_price(future_preds[selected_model]),
            index=future_dates[:len(future_preds[selected_model])]
        )

        sim_df = investment_projection(
            pred_price=selected_future_price,
            initial_investment=initial_investment,
            last_close=last_close,
        )

        final_value = float(sim_df["Projected Value"].iloc[-1])
        pnl = final_value - initial_investment
        pct = (pnl / initial_investment) * 100

        d1, d2, d3 = st.columns(3)
        d1.metric("Initial investment", f"₹{initial_investment:,.2f}")
        d2.metric("Projected final value", f"₹{final_value:,.2f}", f"{pct:+.2f}%")
        d3.metric("Projected P/L", f"₹{pnl:,.2f}")

        st.dataframe(sim_df, use_container_width=True, hide_index=True)

        invest_fig = go.Figure()
        invest_fig.add_trace(
            go.Scatter(
                x=sim_df["Date"],
                y=sim_df["Projected Value"],
                mode="lines+markers",
                name="Projected value",
                line=dict(color="#2ca02c", width=3),
            )
        )
        invest_fig.update_layout(
            template="plotly_white",
            height=500,
            title=f"₹{initial_investment:,.0f} investment projection ({selected_model})",
            xaxis_title="Date",
            yaxis_title="Portfolio value (₹)",
        )
        st.plotly_chart(invest_fig, use_container_width=True)