import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# =========================================================
# 1. PAGE CONFIG & STYLE
# =========================================================
st.set_page_config(page_title="IDX Pro Analyst System", layout="wide")

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

plt.style.use("dark_background")

# =========================================================
# 2. SIDEBAR
# =========================================================
st.sidebar.header("⚙️ Analysis Mode")
mode = st.sidebar.radio("Select Mode", ["Decision Mode", "Trading Mode"])

st.sidebar.header("🔍 Market Scanner")
ticker_input = st.sidebar.text_input("Ticker IDX", value="BBCA.JK").upper()
horizon = st.sidebar.slider("Projection Horizon (Days)", 5, 60, 14)

# =========================================================
# 3. DATA ENGINE (CLEAN & STANDARD)
# =========================================================
@st.cache_data(ttl=3600)
def get_stock_data(symbol):
    # Menggunakan standar yfinance tanpa session manual agar tidak konflik
    # yfinance versi baru sudah menangani header secara otomatis
    
    # 1. Download data history
    data = yf.download(symbol, start="2020-01-01", progress=False)
    
    # 2. Download info profile perusahaan
    try:
        ticker_obj = yf.Ticker(symbol)
        info = ticker_obj.info
    except Exception:
        # Jika gagal ambil info (karena rate limit/koneksi), return kosong agar tidak crash
        info = {}
        
    return data, info

# =========================================================
# 4. MAIN EXECUTION
# =========================================================
try:
    # 1. Ambil Data
    df_raw, stock_info = get_stock_data(ticker_input)
    
    if df_raw.empty:
        st.warning(f"Data untuk {ticker_input} tidak ditemukan atau kosong.")
        st.stop()

    # 2. Bersihkan Data (Data Cleaning)
    df = df_raw.copy()
    
    # Handle MultiIndex (Eror umum di yfinance baru)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Reset index agar 'Date' menjadi kolom biasa
    df = df.reset_index()
    
    # Ubah nama kolom jadi huruf kecil semua (open, close, high, low)
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    
    # Pastikan kolom date adalah datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    # =====================================================
    # TECHNICAL INDICATORS
    # =====================================================
    # Cek jumlah data
    if len(df) < 200:
        st.error(f"Data historis {ticker_input} kurang dari 200 hari. Tidak bisa hitung MA200.")
        st.stop()

    df["ma50"] = df["close"].rolling(50).mean()
    df["ma200"] = df["close"].rolling(200).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    
    # Hindari pembagian dengan nol
    rs = gain / loss.replace(0, 0.001) 
    df["rsi"] = 100 - (100 / (1 + rs))

    latest = df.iloc[-1]
    prev_close = df.iloc[-2]["close"]

    # =====================================================
    # DECISION ENGINE
    # =====================================================
    decision = "HOLD"
    reasons = []

    if mode == "Decision Mode":
        if latest["rsi"] >= 70:
            decision = "HOLD"
            reasons.append("RSI Overbought (Risk Elevated)")
        elif (latest["close"] > latest["ma50"] and latest["ma50"] > latest["ma200"]):
            decision = "BUY"
            reasons.append("Bullish Trend Structure")
        else:
            decision = "HOLD"
            reasons.append("Market Neutral")

    else:  # Trading Mode
        if latest["rsi"] > 75:
            decision = "SELL"
            reasons.append("RSI Overbought (>75)")
        elif latest["close"] < latest["ma200"]:
            decision = "SELL"
            reasons.append("Below MA200 (Bearish)")
        elif latest["rsi"] < 35:
            decision = "BUY"
            reasons.append("RSI Oversold (<35)")
        elif (latest["close"] > latest["ma50"] and latest["ma50"] > latest["ma200"]):
            decision = "STRONG BUY"
            reasons.append("Bullish MA Alignment")
        else:
            decision = "HOLD"
            reasons.append("Market Neutral")

    # =====================================================
    # ML PROJECTION (Linear Regression)
    # =====================================================
    model_df = df.tail(60).dropna()
    X = np.arange(len(model_df)).reshape(-1, 1)
    y = model_df["close"].values

    model = LinearRegression()
    model.fit(X, y)

    reg_pred = model.predict([[len(model_df) + horizon]])[0]

    stop_loss = latest["close"] * 0.95
    take_profit = latest["close"] * 1.10

    # =====================================================
    # UI DISPLAY
    # =====================================================
    title_name = stock_info.get("shortName", ticker_input)
    # Fallback name jika info gagal diambil
    if not title_name: 
        title_name = ticker_input

    st.title(f"📈 {title_name}")

    m1, m2, m3, m4 = st.columns(4)
    color = "buy" if "BUY" in decision else ("sell" if "SELL" in decision else "hold")

    with m1:
        st.markdown(f"<div class='card {color}'><div class='label'>Signal</div><div class='metric-val'>{decision}</div></div>", unsafe_allow_html=True)
    with m2:
        diff = ((latest["close"] - prev_close) / prev_close) * 100
        st.markdown(f"<div class='card hold'><div class='label'>Last Price</div><div class='metric-val'>{latest['close']:,.0f} ({diff:+.2f}%)</div></div>", unsafe_allow_html=True)
    with m3:
        st.markdown(f"<div class='card hold'><div class='label'>RSI</div><div class='metric-val'>{latest['rsi']:.1f}</div></div>", unsafe_allow_html=True)
    with m4:
        pe = stock_info.get("trailingPE")
        pe_display = f"{pe:.2f}x" if isinstance(pe, (float, int)) else "—"
        st.markdown(f"<div class='card hold'><div class='label'>P/E</div><div class='metric-val'>{pe_display}</div></div>", unsafe_allow_html=True)

    # CHARTS
    st.divider()
    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("Price & Trends")
        chart_df = df.tail(365)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(chart_df["date"], chart_df["close"], label="Close", color="cyan")
        ax.plot(chart_df["date"], chart_df["ma50"], label="MA50", linestyle="--", color="yellow")
        ax.plot(chart_df["date"], chart_df["ma200"], label="MA200", linestyle="--", color="red")
        
        # Styling Plot
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        ax.tick_params(colors='white')
        ax.yaxis.label.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.legend(facecolor='black', labelcolor='white')
        ax.grid(alpha=0.2, color='gray')
        
        st.pyplot(fig)

        st.subheader("RSI Momentum")
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.plot(chart_df["date"], chart_df["rsi"], color="lime")
        ax2.axhline(70, color="red", linestyle="--", alpha=0.5)
        ax2.axhline(30, color="green", linestyle="--", alpha=0.5)
        
        # Styling Plot RSI
        ax2.set_facecolor('black')
        fig2.patch.set_facecolor('black')
        ax2.tick_params(colors='white')
        ax2.set_ylim(0, 100)
        
        st.pyplot(fig2)

    with c2:
        st.subheader("📋 Analysis Summary")
        for r in reasons:
            st.write(f"• {r}")

        if mode == "Trading Mode":
            st.divider()
            st.success(f"Take Profit: {take_profit:,.0f}")
            st.error(f"Stop Loss: {stop_loss:,.0f}")

        st.divider()
        change_pct = ((reg_pred - latest["close"]) / latest["close"]) * 100
        st.metric("🤖 AI Forecast", f"{reg_pred:,.0f}", f"{change_pct:+.2f}%")
        st.caption(f"Projection for next {horizon} days (Linear Reg).")

except Exception as e:
    # Error Handling Terpusat
    st.error("Terjadi kendala saat memproses data.")
    st.warning("Jika ini 'Rate Limit' atau 'No Data', coba ganti Ticker atau tunggu sejenak.")
    with st.expander("Lihat Detail Error (Untuk Debugging)"):
        st.code(str(e))

st.divider()
st.caption("Disclaimer: Semua keputusan investasi sepenuhnya menjadi tanggung jawab investor (DYOR).")

# Footer Credit
st.markdown(
    "<div style='text-align:center; color:#8b8b8b; font-size:12px; margin-top:20px;'>"
    "Made with ❤️ by <b>Yoga Adi Tandanu</b>"
    "</div>",
    unsafe_allow_html=True
)