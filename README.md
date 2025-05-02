import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import hashlib

# ---------------------- Streamlit Setup ---------------------- #
st.set_page_config(layout="wide")
st.title("📊 AI-Driven Financial Anomaly & Market Sentiment Dashboard")

# ---------------------- MOCK DATA GENERATION ---------------------- #

@st.cache_data
def generate_stock_data(symbol: str, days: int = 180):
    np.random.seed(abs(hash(symbol)) % 2**32)  # consistent per symbol
    base_price = np.random.uniform(90, 150)
    volatility = np.random.uniform(1, 2)
    drift = np.random.normal(0.01, 0.02)
    prices = [base_price]
    for _ in range(days - 1):
        change = np.random.normal(drift, volatility)
        prices.append(prices[-1] * (1 + change / 100))
    dates = pd.date_range(end=datetime.today(), periods=days)
    return pd.DataFrame({'Date': dates, 'Close': prices})

def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['rolling_mean'] = df['Close'].rolling(window=10, min_periods=1).mean()
    df['rolling_std'] = df['Close'].rolling(window=10, min_periods=1).std()
    df['anomaly_score'] = (df['Close'] - df['rolling_mean']).abs() / (df['rolling_std'] + 1e-6)

    # Quantile-based thresholding
    threshold = df['anomaly_score'].quantile(0.95)
    df['is_anomaly'] = df['anomaly_score'] > threshold

    # Fallback: ensure at least 3 anomalies
    if df['is_anomaly'].sum() == 0:
        top_n = df['anomaly_score'].nlargest(3).index
        df.loc[top_n, 'is_anomaly'] = True

    df['anomaly'] = np.where(df['is_anomaly'], -1, 1)
    return df

def generate_sentiment_score(symbol: str) -> float:
    h = int(hashlib.sha256(symbol.encode()).hexdigest(), 16)
    np.random.seed(h % 2**32)
    return round(np.random.uniform(-1.0, 1.0), 2)

# ---------------------- DASHBOARD UI ---------------------- #

symbol = st.text_input("Enter Stock Symbol (Mocked)", "AAPL").upper()
days = st.slider("Select Date Range (Past Days)", 30, 365, 180)

if symbol:
    df = generate_stock_data(symbol, days)
    df = detect_anomalies(df)
    sentiment = generate_sentiment_score(symbol)

    # Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close Price'))
    anomalies = df[df['is_anomaly']]
    fig.add_trace(go.Scatter(
        x=anomalies['Date'],
        y=anomalies['Close'],
        mode='markers',
        marker=dict(color='red', size=8),
        name='Anomalies'
    ))
    fig.update_layout(title=f"{symbol} Price with Anomalies", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

    # Sentiment
    st.subheader("📉 Market Sentiment Score")
    st.metric(label="Market Sentiment Score", value=sentiment)

    # Raw data
    if st.checkbox("Show raw data"):
        st.dataframe(df[['Date', 'Close', 'anomaly', 'is_anomaly']])
