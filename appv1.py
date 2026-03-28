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
.dividend { background: #1a237e; border-left: 5px solid #5c6bc0; } 
.fundamental { background: #283593; border-left: 5px solid #8e24aa; } 
.metric-val { font-size: 24px; font-weight: bold; }
.label { font-size: 12px; color: #b0b3b8; }
.analyst-text { font-size: 15px; line-height: 1.6; color: #e0e0e0; margin-bottom: 10px;}
</style>
""", unsafe_allow_html=True)

plt.style.use("dark_background")

# =========================================================
# 2. SIDEBAR
# =========================================================
st.sidebar.header("⚙️ Analysis Mode")
mode = st.sidebar.radio("Select Mode", ["Investasi Jangka Panjang", "Trading"])

st.sidebar.header("🔍 Market Scanner")
ticker_input = st.sidebar.text_input("Ticker IDX (Gunakan .JK)", value="BBCA.JK").upper()
horizon = st.sidebar.slider("Projection Horizon (Days)", 5, 60, 14)

# =========================================================
# 3. DATA ENGINE (CLEAN & STANDARD)
# =========================================================
@st.cache_data(ttl=3600)
def get_stock_data(symbol):
    data = yf.download(symbol, start="2010-01-01", actions=True, progress=False)
    try:
        ticker_obj = yf.Ticker(symbol)
        info = ticker_obj.info
    except Exception:
        info = {}
    return data, info

# =========================================================
# 4. MAIN EXECUTION
# =========================================================
try:
    df_raw, stock_info = get_stock_data(ticker_input)
    
    if df_raw.empty:
        st.warning(f"Data untuk {ticker_input} tidak ditemukan atau kosong.")
        st.stop()

    # Data Cleaning
    df = df_raw.copy()
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.reset_index()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    # =====================================================
    # TECHNICAL INDICATORS
    # =====================================================
    if len(df) < 200:
        st.error(f"Data historis {ticker_input} kurang dari 200 hari. Tidak bisa memproses semua indikator.")
        st.stop()

    # 1. Moving Averages
    df["ma50"] = df["close"].rolling(50).mean()
    df["ma200"] = df["close"].rolling(200).mean()

    # 2. RSI (Relative Strength Index)
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, 0.001) 
    df["rsi"] = 100 - (100 / (1 + rs))

    # 3. MACD
    df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema12"] - df["ema26"]
    df["signal_line"] = df["macd"].ewm(span=9, adjust=False).mean()

    # 4. Bollinger Bands (Periode 20, StdDev 2)
    df["bb_middle"] = df["close"].rolling(20).mean()
    df["bb_upper"] = df["bb_middle"] + 2 * df["close"].rolling(20).std()
    df["bb_lower"] = df["bb_middle"] - 2 * df["close"].rolling(20).std()

    latest = df.iloc[-1]
    prev_close = df.iloc[-2]["close"]

    # =====================================================
    # LONG-TERM RETURN & LATEST DIVIDEND
    # =====================================================
    price_col = 'adj_close' if 'adj_close' in df.columns else 'close'
    
    harga_awal = df[price_col].iloc[0]
    harga_akhir = df[price_col].iloc[-1]
    
    harga_awal = float(harga_awal) if not isinstance(harga_awal, pd.Series) else float(harga_awal.iloc[0])
    harga_akhir = float(harga_akhir) if not isinstance(harga_akhir, pd.Series) else float(harga_akhir.iloc[0])
    
    total_return_pct = ((harga_akhir - harga_awal) / harga_awal) * 100
    
    dividen_terakhir = 0
    tanggal_dividen = "-"
    
    if 'dividends' in df.columns and df['dividends'].sum() > 0:
        df_div = df[df['dividends'] > 0]
        if not df_div.empty:
            dividen_terakhir = float(df_div.iloc[-1]['dividends'])
            tanggal_dividen = df_div.iloc[-1]['date'].strftime('%d %b %Y')

    # =====================================================
    # FUNDAMENTAL METRICS EXTRACTION
    # =====================================================
    pe_ratio = stock_info.get("trailingPE", None)
    pb_ratio = stock_info.get("priceToBook", None)
    roe = stock_info.get("returnOnEquity", None)
    market_cap = stock_info.get("marketCap", None)
    div_yield = stock_info.get("dividendYield", None)

    pe_display = f"{pe_ratio:.2f}x" if pe_ratio else "—"
    pb_display = f"{pb_ratio:.2f}x" if pb_ratio else "—"
    roe_display = f"{(roe * 100):.2f}%" if roe else "—"
    
    if div_yield is not None:
        if div_yield > 1: 
            div_yield_display = f"{div_yield:.2f}%"
        else:             
            div_yield_display = f"{(div_yield * 100):.2f}%"
    else:
        div_yield_display = "—"
   
    if market_cap:
        if market_cap >= 1e12:
            mcap_display = f"Rp {market_cap / 1e12:.2f} T"
        elif market_cap >= 1e9:
            mcap_display = f"Rp {market_cap / 1e9:.2f} M"
        else:
            mcap_display = f"Rp {market_cap:,.0f}"
    else:
        mcap_display = "—"

    # =====================================================
    # DECISION ENGINE (CALCULATION & ANALYST TEXT)
    # =====================================================
    decision = "HOLD"
    reasons = []

    if mode == "Investasi Jangka Panjang":
        if roe and roe > 0.15:
            reasons.append("Fundamental Kuat: Tingkat profitabilitas perusahaan sangat baik (ROE > 15%). Manajemen terbukti mampu memutar modal ekuitas secara efisien untuk menghasilkan laba.")
        elif roe and roe < 0.05:
            reasons.append("Fundamental Kurang Efisien: Tingkat profitabilitas tergolong lemah (ROE < 5%). Ada tantangan bagi perusahaan dalam mencetak laba maksimal dari modal yang ditanamkan.")
            
        if pe_ratio and pe_ratio < 15:
            reasons.append("Valuasi Atraktif: Berdasarkan P/E ratio di bawah 15x, harga saham tergolong cukup murah (Undervalued) dibandingkan kinerja labanya. Memberikan *margin of safety* yang baik.")
            
        if latest["close"] > latest["ma200"]:
            reasons.append("Tren Harga Positif: Secara historis jangka panjang, saham berada di fase Uptrend karena posisinya konsisten di atas MA200. Mengonfirmasi adanya akumulasi bertahap.")
            decision = "AKUMULASI / BUY"
        else:
            reasons.append("Tren Harga Negatif: Saham dalam bayang-bayang Downtrend jangka panjang karena tertekan di bawah MA200. Disarankan menunggu konfirmasi pembalikan arah.")
            decision = "WAIT & SEE / HOLD"
            
    else: # Trading Mode
        if (latest["close"] > latest["ma50"] and latest["ma50"] > latest["ma200"]):
            decision = "STRONG BUY"
            reasons.append("Struktur Bullish: Susunan teknikal solid (*Golden Alignment*). Harga di atas MA50, dan MA50 di atas MA200. Momentum pergerakan naik sangat kuat.")
        elif latest["close"] < latest["ma200"]:
            decision = "SELL"
            reasons.append("Struktur Bearish: Harga diperdagangkan di bawah MA200 yang bertindak sebagai resistance dinamis kuat. Menandakan tekanan jual yang dominan.")
        else:
            decision = "HOLD"
            reasons.append("Fase Konsolidasi: Pergerakan harga tidak menunjukkan kecenderungan arah tren yang ekstrem di antara garis Moving Average.")

        if latest["rsi"] > 75:
            decision = "SELL"
            reasons.append("Peringatan RSI: Momentum masuk area *Overbought* (>75). Probabilitas tinggi terjadinya aksi *profit-taking* yang dapat memicu koreksi jangka pendek.")
        elif latest["rsi"] < 35:
            if decision == "SELL": decision = "HOLD" 
            elif decision == "HOLD": decision = "BUY"
            reasons.append("Peluang RSI: Momentum menyentuh area *Oversold* (<35). Sering kali membuka ruang bagi pembeli spekulatif untuk mendorong *technical rebound*.")

        if latest["macd"] > latest["signal_line"]:
            reasons.append("MACD Positif: Garis MACD memotong ke atas *Signal Line* (Golden Cross), mengonfirmasi akselerasi momentum kenaikan harga.")
        else:
            reasons.append("MACD Negatif: Garis MACD berada di bawah *Signal Line* (Death Cross), memberikan sinyal awal pelemahan momentum dan tekanan distribusi.")

        if latest["close"] < latest["bb_lower"]:
            if decision == "HOLD": decision = "BUY"
            reasons.append("Bollinger Extreme Low: Harga telah menembus batas bawah (*Lower Band*). Mengindikasikan kepanikan jual berlebih dan probabilitas kuat untuk memantul naik.")
        elif latest["close"] > latest["bb_upper"]:
            if decision == "BUY" or decision == "STRONG BUY": decision = "HOLD"
            reasons.append("Bollinger Extreme High: Harga menembus batas atas (*Upper Band*). Saham sudah terlalu 'mahal' dalam jangka pendek, waspadai potensi *pullback* ke harga rata-rata.")

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
    if not title_name: 
        title_name = ticker_input

    st.title(f"📈 {title_name} - {mode}")

    m1, m2, m3, m4 = st.columns(4)
    color = "buy" if "BUY" in decision or "AKUMULASI" in decision else ("sell" if "SELL" in decision else "hold")

    with m1:
        st.markdown(f"<div class='card {color}'><div class='label'>Signal ({mode})</div><div class='metric-val'>{decision}</div></div>", unsafe_allow_html=True)
    with m2:
        diff = ((latest["close"] - prev_close) / prev_close) * 100
        st.markdown(f"<div class='card hold'><div class='label'>Last Price</div><div class='metric-val'>Rp {latest['close']:,.0f} ({diff:+.2f}%)</div></div>", unsafe_allow_html=True)
    with m3:
        st.markdown(f"<div class='card hold'><div class='label'>RSI (Momentum)</div><div class='metric-val'>{latest['rsi']:.1f}</div></div>", unsafe_allow_html=True)
    with m4:
        change_pct = ((reg_pred - latest["close"]) / latest["close"]) * 100
        st.markdown(f"<div class='card hold'><div class='label'>AI Forecast ({horizon}D)</div><div class='metric-val'>Rp {reg_pred:,.0f}</div></div>", unsafe_allow_html=True)

    # FUNDAMENTAL SECTION
    st.divider()
    st.subheader("📊 Analisa Fundamental & Valuasi")
    f1, f2, f3, f4, f5 = st.columns(5)
    
    with f1:
        st.markdown(f"<div class='card fundamental'><div class='label'>P/E Ratio</div><div class='metric-val'>{pe_display}</div></div>", unsafe_allow_html=True)
    with f2:
        st.markdown(f"<div class='card fundamental'><div class='label'>Price to Book (PBV)</div><div class='metric-val'>{pb_display}</div></div>", unsafe_allow_html=True)
    with f3:
        st.markdown(f"<div class='card fundamental'><div class='label'>Return on Equity (ROE)</div><div class='metric-val'>{roe_display}</div></div>", unsafe_allow_html=True)
    with f4:
        st.markdown(f"<div class='card fundamental'><div class='label'>Dividend Yield</div><div class='metric-val'>{div_yield_display}</div></div>", unsafe_allow_html=True)
    with f5:
        st.markdown(f"<div class='card fundamental'><div class='label'>Market Cap</div><div class='metric-val'>{mcap_display}</div></div>", unsafe_allow_html=True)

    # LONG-TERM SUMMARY UI
    st.divider()
    st.subheader("💰 Kinerja Jangka Panjang")
    lt1, lt2, lt3 = st.columns(3)
    
    with lt1:
        st.markdown(f"<div class='card dividend'><div class='label'>Dividen Terakhir ({tanggal_dividen})</div><div class='metric-val'>Rp {dividen_terakhir:,.0f}</div></div>", unsafe_allow_html=True)
    with lt2:
        return_color = "#2ecc71" if total_return_pct > 0 else "#e74c3c"
        st.markdown(f"<div class='card hold'><div class='label'>All-Time Return (Capital Gain)</div><div class='metric-val' style='color:{return_color};'>{total_return_pct:+.2f}%</div></div>", unsafe_allow_html=True)
    with lt3:
        st.markdown(f"<div class='card hold'><div class='label'>Initial Price (2010 Adj)</div><div class='metric-val'>Rp {harga_awal:,.0f}</div></div>", unsafe_allow_html=True)

# =====================================================
    # DCA SIMULATOR (DATA-DRIVEN CAGR)
    # =====================================================
    if mode == "Investasi Jangka Panjang":
        st.divider()
        st.subheader("💸 Simulasi Dollar Cost Averaging (DCA)")
        
        # 1. Hitung Umur Data (Tahun)
        umur_hari = (df['date'].iloc[-1] - df['date'].iloc[0]).days
        umur_tahun = umur_hari / 365.25
        
        # 2. Hitung CAGR Historis Asli
        if umur_tahun > 0 and harga_awal > 0:
            cagr_historis = ((harga_akhir / harga_awal) ** (1 / umur_tahun)) - 1
            cagr_persen = cagr_historis * 100
        else:
            cagr_persen = 0.0

        st.caption(f"Proyeksi ini menggunakan setoran rutin bulanan. Return default diisi otomatis berdasarkan CAGR historis saham **{ticker_input}** selama **{umur_tahun:.1f} tahun** terakhir ({cagr_persen:.2f}%).")

        dca_c1, dca_c2, dca_c3 = st.columns(3)
        with dca_c1:
            monthly_invest = st.number_input("Setoran Bulanan (Rp)", min_value=100000, value=1500000, step=100000)
        with dca_c2:
            dca_years = st.slider("Durasi Investasi Ke Depan (Tahun)", min_value=1, max_value=20, value=5)
        with dca_c3:
            # Gunakan CAGR historis sebagai default value (dibulatkan 2 desimal)
            # Jika performa minus, kita batasi nilai minimum input agar sistem tidak error, atau biarkan user melihat realitanya
            default_cagr = float(round(cagr_persen, 2))
            annual_cagr = st.number_input("Asumsi Return Tahunan (%)", value=default_cagr, step=1.0)

        # 3. Hitung DCA Future Value menggunakan rumus Compound Interest untuk Anuitas
        months = dca_years * 12
        monthly_rate = (annual_cagr / 100) / 12
        total_invested = monthly_invest * months

        if monthly_rate > 0:
            future_value = monthly_invest * (((1 + monthly_rate)**months - 1) / monthly_rate)
        elif monthly_rate < 0:
            # Jika return negatif, kita gunakan hitungan eksponensial menurun
            future_value = monthly_invest * (((1 + monthly_rate)**months - 1) / monthly_rate) 
        else:
            future_value = total_invested
            
        profit = future_value - total_invested

        # 4. Tampilkan Hasil DCA
        res1, res2, res3 = st.columns(3)
        with res1:
            st.markdown(f"<div class='card hold'><div class='label'>Total Modal Disetor ({dca_years} Thn)</div><div class='metric-val'>Rp {total_invested:,.0f}</div></div>", unsafe_allow_html=True)
        with res2:
            st.markdown(f"<div class='card buy'><div class='label'>Estimasi Nilai Akhir Portofolio</div><div class='metric-val'>Rp {future_value:,.0f}</div></div>", unsafe_allow_html=True)
        with res3:
            # Warna dinamis: Hijau jika untung, Merah jika rugi
            profit_color = "#2ecc71" if profit >= 0 else "#e74c3c"
            profit_sign = "+" if profit >= 0 else ""
            st.markdown(f"<div class='card fundamental'><div class='label'>Potensi Return (Capital Gain)</div><div class='metric-val' style='color:{profit_color};'>{profit_sign} Rp {profit:,.0f}</div></div>", unsafe_allow_html=True)
    # CHARTS
    st.divider()
    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("Price & Trends (with Bollinger Bands)")
        chart_df = df.tail(365) 
        
        fig, ax = plt.subplots(figsize=(10, 5))
        # Plot Harga Utama dan Moving Averages
        ax.plot(chart_df["date"], chart_df["close"], label="Close", color="cyan")
        ax.plot(chart_df["date"], chart_df["ma50"], label="MA50", linestyle="--", color="yellow")
        ax.plot(chart_df["date"], chart_df["ma200"], label="MA200", linestyle="--", color="red")
        
        # Plot Bollinger Bands
        ax.plot(chart_df["date"], chart_df["bb_upper"], label="BB Upper", linestyle=":", color="magenta", alpha=0.5)
        ax.plot(chart_df["date"], chart_df["bb_lower"], label="BB Lower", linestyle=":", color="magenta", alpha=0.5)
        
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
        
        ax2.set_facecolor('black')
        fig2.patch.set_facecolor('black')
        ax2.tick_params(colors='white')
        ax2.set_ylim(0, 100)
        
        st.pyplot(fig2)

    with c2:
        st.subheader("📋 Ringkasan Analisa")
        for r in reasons:
            st.markdown(f"<div class='analyst-text'>📌 {r}</div>", unsafe_allow_html=True)

        if mode == "Trading":
            st.divider()
            st.success(f"🎯 Take Profit (Target Ideal): Rp {take_profit:,.0f}")
            st.error(f"🛑 Stop Loss (Batas Risiko): Rp {stop_loss:,.0f}")
            st.caption("Catatan: Angka dikalkulasi secara mekanis. Sesuaikan dengan rasio Risk/Reward dan level support/resistance.")

except Exception as e:
    st.error("Terjadi kendala saat memproses data.")
    st.warning("Pastikan Anda menggunakan .JK untuk saham Indonesia (misal: TLKM.JK) dan koneksi internet stabil.")
    with st.expander("Lihat Detail Error (Untuk Debugging)"):
        st.code(str(e))

st.divider()
st.caption("Disclaimer: Analisis ini digenerasi oleh sistem berbasis aturan teknikal dan fundamental dasar. Semua keputusan investasi sepenuhnya menjadi tanggung jawab investor (DYOR).")

st.markdown(
    "<div style='text-align:center; color:#8b8b8b; font-size:12px; margin-top:20px;'>"
    "Made with ❤️ by <b>Yoga Adi Tandanu</b>"
    "</div>",
    unsafe_allow_html=True
)
