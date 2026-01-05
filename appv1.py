import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# =========================================================
# 1. PAGE CONFIG & CSS
# =========================================================
st.set_page_config(page_title="IDX Pro Analyst System", layout="wide")

# Styling agar tampilan Metrics rapi
st.markdown("""
<style>
    .card {
        padding: 20px; border-radius: 12px; margin-bottom: 10px;
        color: #ffffff; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .buy { background: #1e3a2f; border-left: 5px solid #2ecc71; }
    .sell { background: #3a1e1e; border-left: 5px solid #e74c3c; }
    .hold { background: #2c2c2c; border-left: 5px solid #f1c40f; }
    .metric-val { font-size: 24px; font-weight: bold; }
    .label { font-size: 12px; color: #b0b3b8; }
</style>
""", unsafe_allow_html=True)

# Set Style Matplotlib ke Dark Mode agar cocok dengan Streamlit
plt.style.use('dark_background')

# =========================================================
# 2. SIDEBAR
# =========================================================
st.sidebar.header("⚙️ Analysis Mode")
mode = st.sidebar.radio(
    "Select Mode",
    ["Decision Mode", "Trading Mode"]
)

st.sidebar.header("🔍 Market Scanner")
ticker_input = st.sidebar.text_input("Ticker IDX", value="BBCA.JK").upper()
horizon = st.sidebar.slider("Projection Horizon (Days)", 5, 60, 14)

# =========================================================
# 3. DATA ENGINE
# =========================================================
@st.cache_data(ttl=3600)
def get_stock_data(symbol):
    # Ambil data dari 2020 agar cukup untuk Moving Average
    data = yf.download(symbol, start="2020-01-01", progress=False)
    info = yf.Ticker(symbol).info
    return data, info

try:
    df_raw, stock_info = get_stock_data(ticker_input)
    if df_raw.empty:
        st.error("Ticker tidak ditemukan atau data kosong.")
        st.stop()

    df = df_raw.copy()

    # Handling MultiIndex columns dari yfinance versi baru
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    # Standardisasi nama kolom jadi huruf kecil (date, open, close, dll)
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    
    # Pastikan kolom date adalah datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    # =========================================================
    # 4. TECHNICAL INDICATORS & LOGIC
    # =========================================================
    df["ma50"] = df["close"].rolling(50).mean()
    df["ma200"] = df["close"].rolling(200).mean()

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    latest = df.iloc[-1]
    prev_close = df.iloc[-2]["close"]

    # Decision Engine
    decision = "HOLD"
    reasons = []

    if mode == "Decision Mode":
        if latest["rsi"] >= 70:
            decision = "HOLD"
            reasons.append("RSI Overbought (Risk Elevated)")
        elif latest["close"] > latest["ma50"] > latest["ma200"]:
            decision = "BUY"
            reasons.append("Bullish Trend Structure")
        else:
            reasons.append("Market Neutral")
    else:
        if latest["rsi"] > 75:
            decision = "SELL"
            reasons.append("RSI Overbought (>75)")
        elif latest["close"] < latest["ma200"]:
            decision = "SELL"
            reasons.append("Below MA200 (Bearish)")
        elif latest["rsi"] < 35:
            decision = "BUY"
            reasons.append("RSI Oversold (<35)")
        elif latest["close"] > latest["ma50"] > latest["ma200"]:
            decision = "STRONG BUY"
            reasons.append("Bullish MA Alignment")
        else:
            reasons.append("Market Neutral")

    # ML Projection
    model_df = df.tail(60).dropna()
    X = np.arange(len(model_df)).reshape(-1, 1)
    y = model_df["close"].values
    model = LinearRegression().fit(X, y)
    reg_pred = model.predict([[len(model_df) + horizon]])[0]

    stop_loss = latest["close"] * 0.95
    take_profit = latest["close"] * 1.10

    # =========================================================
    # 5. UI DISPLAY
    # =========================================================
    st.title(f"📈 {stock_info.get('shortName', ticker_input)}")

    # -- Metric Cards --
    m1, m2, m3, m4 = st.columns(4)
    color = "buy" if "BUY" in decision else ("sell" if "SELL" in decision else "hold")

    with m1:
        st.markdown(f"<div class='card {color}'><div class='label'>Signal</div><div class='metric-val'>{decision}</div></div>", unsafe_allow_html=True)
    with m2:
        diff = ((latest['close'] - prev_close) / prev_close) * 100
        st.markdown(f"<div class='card hold'><div class='label'>Last Price</div><div class='metric-val'>{latest['close']:,.0f} ({diff:+.2f}%)</div></div>", unsafe_allow_html=True)
    with m3:
        st.markdown(f"<div class='card hold'><div class='label'>RSI</div><div class='metric-val'>{latest['rsi']:.1f}</div></div>", unsafe_allow_html=True)
    with m4:
        pe = stock_info.get("trailingPE", "N/A")
        pe_str = f"{pe:.2f}x" if isinstance(pe, (float, int)) else "N/A"
        st.markdown(f"<div class='card hold'><div class='label'>P/E</div><div class='metric-val'>{pe_str}</div></div>", unsafe_allow_html=True)

    # -- SIMPLE CHARTS (MATPLOTLIB) --
    st.write("---")
    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("Price & Trends")
        # Filter data 1 tahun terakhir agar grafik tidak terlalu padat
        chart_df = df.tail(365)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        # Plot Close Price
        ax.plot(chart_df["date"], chart_df["close"], label="Close Price", color="cyan", linewidth=1.5)
        # Plot Moving Averages
        ax.plot(chart_df["date"], chart_df["ma50"], label="MA50", color="yellow", linestyle="--", linewidth=1)
        ax.plot(chart_df["date"], chart_df["ma200"], label="MA200", color="red", linestyle="--", linewidth=1)
        
        ax.set_title(f"{ticker_input} Daily Chart (1 Year)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (IDR)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        st.subheader("RSI Indicator")
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.plot(chart_df["date"], chart_df["rsi"], label="RSI", color="lime")
        ax2.axhline(70, color="red", linestyle="--", alpha=0.5)
        ax2.axhline(30, color="green", linestyle="--", alpha=0.5)
        ax2.fill_between(chart_df["date"], 70, 30, color='gray', alpha=0.2)
        ax2.legend(loc='upper left')
        ax2.set_ylim(0, 100)
        st.pyplot(fig2)

    with c2:
        st.subheader("📋 Analysis Summary")
        
        # Tampilkan Alasan
        st.info(f"**Signal Reason:**")
        for r in reasons:
            st.write(f"• {r}")
            
        st.markdown("---")
        
        # Tampilkan Trading Plan
        if mode == "Trading Mode":
            st.subheader("🎯 Plan")
            st.success(f"Take Profit: **{take_profit:,.0f}**")
            st.error(f"Stop Loss: **{stop_loss:,.0f}**")
        
        st.markdown("---")
        
        # Tampilkan Prediksi ML
        st.subheader(f"🤖 AI Forecast ({horizon} Days)")
        change_pct = ((reg_pred - latest['close']) / latest['close']) * 100
        color_pred = "green" if change_pct > 0 else "red"
        
        st.metric("Predicted Price", f"{reg_pred:,.0f}", f"{change_pct:+.2f}%")
        st.caption("Prediksi menggunakan Linear Regression sederhana pada tren 60 hari terakhir.")

except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")

st.markdown("---")
st.caption("Disclaimer On: Segala keputusan investasi (beli/jual) sepenuhnya menjadi tanggung jawab investor (Do Your Own Research). Data disajikan sebagai alat bantu analisis.")