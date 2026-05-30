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
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 10px;
    color: #ffffff;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
}
.buy  { background: #1e3a2f; border-left: 5px solid #2ecc71; }
.sell { background: #3a1e1e; border-left: 5px solid #e74c3c; }
.hold { background: #2c2c2c; border-left: 5px solid #f1c40f; }
.dividend   { background: #1a237e; border-left: 5px solid #5c6bc0; }
.fundamental{ background: #283593; border-left: 5px solid #8e24aa; }
.metric-val { font-size: 24px; font-weight: bold; }
.label      { font-size: 12px; color: #b0b3b8; }
.analyst-text { font-size: 15px; line-height: 1.6; color: #e0e0e0; margin-bottom: 10px; }
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
# 3. DATA ENGINE
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
        st.error(f"Data historis {ticker_input} kurang dari 200 hari.")
        st.stop()

    df["ma50"]  = df["close"].rolling(50).mean()
    df["ma200"] = df["close"].rolling(200).mean()

    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = -delta.clip(upper=0).rolling(14).mean()
    rs    = gain / loss.replace(0, 0.001)
    df["rsi"] = 100 - (100 / (1 + rs))

    df["ema12"]       = df["close"].ewm(span=12, adjust=False).mean()
    df["ema26"]       = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"]        = df["ema12"] - df["ema26"]
    df["signal_line"] = df["macd"].ewm(span=9, adjust=False).mean()

    df["bb_middle"] = df["close"].rolling(20).mean()
    df["bb_upper"]  = df["bb_middle"] + 2 * df["close"].rolling(20).std()
    df["bb_lower"]  = df["bb_middle"] - 2 * df["close"].rolling(20).std()

    latest     = df.iloc[-1]
    prev_close = df.iloc[-2]["close"]

    # =====================================================
    # PRICE & DIVIDEND BASELINE
    # =====================================================
    price_col  = 'adj_close' if 'adj_close' in df.columns else 'close'
    harga_awal = df[price_col].iloc[0]
    harga_akhir= df[price_col].iloc[-1]
    harga_awal  = float(harga_awal)  if not isinstance(harga_awal,  pd.Series) else float(harga_awal.iloc[0])
    harga_akhir = float(harga_akhir) if not isinstance(harga_akhir, pd.Series) else float(harga_akhir.iloc[0])

    dividen_terakhir = 0
    tanggal_dividen  = "-"
    if 'dividends' in df.columns and df['dividends'].sum() > 0:
        df_div = df[df['dividends'] > 0]
        if not df_div.empty:
            dividen_terakhir = float(df_div.iloc[-1]['dividends'])
            tanggal_dividen  = df_div.iloc[-1]['date'].strftime('%d %b %Y')

    # =====================================================
    # FUNDAMENTAL METRICS
    # =====================================================
    pe_ratio   = stock_info.get("trailingPE",      None)
    pb_ratio   = stock_info.get("priceToBook",     None)
    roe        = stock_info.get("returnOnEquity",  None)
    market_cap = stock_info.get("marketCap",       None)
    div_yield  = stock_info.get("dividendYield",   None)

    pe_display  = f"{pe_ratio:.2f}x"           if pe_ratio   else "—"
    pb_display  = f"{pb_ratio:.2f}x"           if pb_ratio   else "—"
    roe_display = f"{(roe * 100):.2f}%"        if roe        else "—"

    if div_yield is not None:
        div_yield_display = f"{div_yield:.2f}%" if div_yield > 1 else f"{(div_yield * 100):.2f}%"
    else:
        div_yield_display = "—"

    if market_cap:
        if   market_cap >= 1e12: mcap_display = f"Rp {market_cap/1e12:.2f} T"
        elif market_cap >= 1e9:  mcap_display = f"Rp {market_cap/1e9:.2f} M"
        else:                    mcap_display = f"Rp {market_cap:,.0f}"
    else:
        mcap_display = "—"

    # =====================================================
    # DECISION ENGINE
    # =====================================================
    decision = "HOLD"
    reasons  = []

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
    else:
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
            if decision == "SELL":  decision = "HOLD"
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
            if decision in ("BUY", "STRONG BUY"): decision = "HOLD"
            reasons.append("Bollinger Extreme High: Harga menembus batas atas (*Upper Band*). Saham sudah terlalu 'mahal' dalam jangka pendek, waspadai potensi *pullback* ke harga rata-rata.")

    # =====================================================
    # ML PROJECTION (Linear Regression)
    # =====================================================
    model_df = df.tail(60).dropna()
    X = np.arange(len(model_df)).reshape(-1, 1)
    y = model_df["close"].values
    model = LinearRegression()
    model.fit(X, y)
    reg_pred   = model.predict([[len(model_df) + horizon]])[0]
    stop_loss  = latest["close"] * 0.95
    take_profit= latest["close"] * 1.10

    # =====================================================
    # CAGR HISTORIS
    # =====================================================
    umur_hari_total  = (df['date'].iloc[-1] - df['date'].iloc[0]).days
    umur_tahun_total = umur_hari_total / 365.25
    if umur_tahun_total > 0 and harga_awal > 0:
        cagr_historis = ((harga_akhir / harga_awal) ** (1 / umur_tahun_total)) - 1
        cagr_persen   = cagr_historis * 100
    else:
        cagr_persen = 0.0

    avg_div_per_tahun = 0
    if 'dividends' in df.columns and umur_tahun_total > 0:
        avg_div_per_tahun = float(df['dividends'].sum()) / umur_tahun_total

    # =====================================================
    # UI: HEADER & SIGNAL CARDS
    # =====================================================
    title_name = stock_info.get("shortName", ticker_input) or ticker_input
    st.title(f"📈 {title_name} - {mode}")

    m1, m2, m3, m4 = st.columns(4)
    color = "buy" if ("BUY" in decision or "AKUMULASI" in decision) else ("sell" if "SELL" in decision else "hold")

    with m1:
        st.markdown(f"<div class='card {color}'><div class='label'>Signal ({mode})</div><div class='metric-val'>{decision}</div></div>", unsafe_allow_html=True)
    with m2:
        diff = ((latest["close"] - prev_close) / prev_close) * 100
        st.markdown(f"<div class='card hold'><div class='label'>Last Price</div><div class='metric-val'>Rp {latest['close']:,.0f} ({diff:+.2f}%)</div></div>", unsafe_allow_html=True)
    with m3:
        st.markdown(f"<div class='card hold'><div class='label'>RSI (Momentum)</div><div class='metric-val'>{latest['rsi']:.1f}</div></div>", unsafe_allow_html=True)
    with m4:
        st.markdown(f"<div class='card hold'><div class='label'>AI Forecast ({horizon}D)</div><div class='metric-val'>Rp {reg_pred:,.0f}</div></div>", unsafe_allow_html=True)

    # =====================================================
    # FUNDAMENTAL CARDS
    # =====================================================
    st.divider()
    st.subheader("📊 Analisa Fundamental & Valuasi")
    f1, f2, f3, f4, f5 = st.columns(5)
    with f1: st.markdown(f"<div class='card fundamental'><div class='label'>P/E Ratio</div><div class='metric-val'>{pe_display}</div></div>", unsafe_allow_html=True)
    with f2: st.markdown(f"<div class='card fundamental'><div class='label'>Price to Book (PBV)</div><div class='metric-val'>{pb_display}</div></div>", unsafe_allow_html=True)
    with f3: st.markdown(f"<div class='card fundamental'><div class='label'>Return on Equity (ROE)</div><div class='metric-val'>{roe_display}</div></div>", unsafe_allow_html=True)
    with f4: st.markdown(f"<div class='card fundamental'><div class='label'>Dividend Yield</div><div class='metric-val'>{div_yield_display}</div></div>", unsafe_allow_html=True)
    with f5: st.markdown(f"<div class='card fundamental'><div class='label'>Market Cap</div><div class='metric-val'>{mcap_display}</div></div>", unsafe_allow_html=True)

    # =====================================================
    # FUNDAMENTAL ANALYST TEXT
    # =====================================================
    st.subheader("🧠 Analisa Fundamental")
    fundamental_texts = []

    # P/E
    if pe_ratio:
        if pe_ratio < 10:
            fundamental_texts.append(f"📌 **Valuasi Sangat Murah (P/E {pe_ratio:.1f}x):** Saham diperdagangkan di bawah 10x earnings. Ini adalah level *deep value* yang menarik — kemungkinan pasar sedang underpricing kualitas bisnis ini, atau ada sentimen negatif sementara yang bisa jadi peluang.")
        elif pe_ratio < 15:
            fundamental_texts.append(f"📌 **Valuasi Wajar-Murah (P/E {pe_ratio:.1f}x):** Harga saham masih terbilang terjangkau dibandingkan profitnya. Memberikan *margin of safety* yang nyaman bagi investor jangka panjang.")
        elif pe_ratio < 25:
            fundamental_texts.append(f"📌 **Valuasi Moderat (P/E {pe_ratio:.1f}x):** Pasar membayar harga yang wajar. Saham tidak murah, namun juga tidak mahal — perlu pertumbuhan laba yang konsisten untuk membenarkan valuasi ini ke depan.")
        else:
            fundamental_texts.append(f"📌 **Valuasi Premium (P/E {pe_ratio:.1f}x):** Pasar mempricing saham ini dengan ekspektasi pertumbuhan tinggi. Cocok jika bisnis memang tumbuh pesat, namun berisiko koreksi keras jika laba sewaktu-waktu meleset dari ekspektasi.")
    else:
        fundamental_texts.append("📌 **P/E Ratio:** Data tidak tersedia. Kemungkinan perusahaan sedang merugi atau data belum terupdate di sumber data.")

    # PBV
    if pb_ratio:
        if pb_ratio < 1:
            fundamental_texts.append(f"📌 **PBV di Bawah 1x ({pb_ratio:.2f}x):** Harga saham lebih rendah dari nilai bukunya — secara teori kamu 'membeli aset seharga diskon'. Menarik, namun perlu dicek apakah ada alasan struktural mengapa bisnis dihargai serendah ini oleh pasar.")
        elif pb_ratio < 2:
            fundamental_texts.append(f"📌 **PBV Wajar ({pb_ratio:.2f}x):** Harga relatif proporsional terhadap nilai aset bersih perusahaan. Umumnya masih kategori menarik, terutama untuk sektor perbankan dan keuangan.")
        elif pb_ratio < 5:
            fundamental_texts.append(f"📌 **PBV Moderat ({pb_ratio:.2f}x):** Pasar memberi premium atas nilai buku, mencerminkan ekspektasi profitabilitas yang baik ke depan. Wajar selama ROE juga tinggi.")
        else:
            fundamental_texts.append(f"📌 **PBV Tinggi ({pb_ratio:.2f}x):** Premium valuasi sangat tinggi. Biasanya hanya terjustifikasi jika ROE konsisten tinggi dan bisnis memiliki *economic moat* yang kuat dan sulit ditiru pesaing.")
    else:
        fundamental_texts.append("📌 **Price to Book (PBV):** Data tidak tersedia.")

    # ROE
    if roe:
        roe_pct = roe * 100
        if roe_pct >= 20:
            fundamental_texts.append(f"📌 **ROE Excellent ({roe_pct:.1f}%):** Kemampuan perusahaan menghasilkan laba dari modal ekuitas tergolong kelas atas. Ini adalah tanda bisnis berkualitas tinggi dengan *competitive advantage* yang nyata dan konsisten.")
        elif roe_pct >= 15:
            fundamental_texts.append(f"📌 **ROE Kuat ({roe_pct:.1f}%):** Manajemen terbukti efisien dalam memutar ekuitas menjadi profit. Layak jadi kandidat investasi jangka panjang dengan fundamental yang solid.")
        elif roe_pct >= 8:
            fundamental_texts.append(f"📌 **ROE Cukup ({roe_pct:.1f}%):** Profitabilitas ada, namun masih di bawah benchmark ideal. Pantau apakah tren ROE membaik atau justru memburuk dalam beberapa kuartal ke belakang.")
        elif roe_pct >= 0:
            fundamental_texts.append(f"📌 **ROE Lemah ({roe_pct:.1f}%):** Tingkat pengembalian ekuitas sangat rendah. Perlu investigasi lebih dalam: apakah ini efek siklus industri, atau masalah struktural bisnis yang lebih dalam?")
        else:
            fundamental_texts.append(f"📌 **ROE Negatif ({roe_pct:.1f}%):** Perusahaan saat ini sedang merugi dan menggerus ekuitas pemegang saham. Hati-hati dan pastikan memahami penyebab kerugian sebelum berinvestasi.")
    else:
        fundamental_texts.append("📌 **Return on Equity (ROE):** Data tidak tersedia.")

    # Dividend Yield
    if div_yield is not None:
        dy = div_yield * 100 if div_yield <= 1 else div_yield
        if dy >= 5:
            fundamental_texts.append(f"📌 **Dividend Yield Tinggi ({dy:.2f}%):** Imbal hasil dividen sangat menarik, melebihi rata-rata deposito. Cocok untuk investor yang fokus pada *passive income* dari dividen rutin.")
        elif dy >= 2:
            fundamental_texts.append(f"📌 **Dividend Yield Moderat ({dy:.2f}%):** Perusahaan rutin membagikan dividen dengan yield yang wajar. Menjadi bonus tambahan di atas potensi capital gain jangka panjang.")
        elif dy > 0:
            fundamental_texts.append(f"📌 **Dividend Yield Rendah ({dy:.2f}%):** Perusahaan membagi dividen, namun nilainya kecil. Kemungkinan sebagian besar laba diputar kembali untuk pertumbuhan bisnis (*reinvestment*).")
    else:
        fundamental_texts.append("📌 **Dividend Yield:** Tidak ada data dividen atau perusahaan tidak membagikan dividen saat ini.")

    for text in fundamental_texts:
        st.markdown(f"<div class='analyst-text'>{text}</div>", unsafe_allow_html=True)

    # =====================================================
    # SIMULATOR: HISTORIS & MASA DEPAN
    # =====================================================
    st.divider()
    st.subheader("🕰️ Simulator Investasi Historis & Proyeksi")

    tab_hist, tab_future = st.tabs(["📜 Jika Beli X Tahun Lalu", "🔮 Jika Beli Sekarang & Tahan X Tahun"])

    # --- TAB 1: HISTORIS ---
    with tab_hist:
        st.caption(f"Simulasi berdasarkan data historis nyata **{ticker_input}**. Dividen dihitung kumulatif dari tanggal beli sampai hari ini.")
        h1, h2, h3 = st.columns(3)
        with h1:
            max_tahun = max(1, int(umur_tahun_total))
            tahun_lalu = st.slider("Invest berapa tahun lalu?", min_value=1, max_value=max_tahun, value=min(5, max_tahun), key="hist_years")
        with h2:
            modal_hist = st.number_input("Modal Awal (Rp)", min_value=1_000_000, value=10_000_000, step=1_000_000, key="modal_hist")
        with h3:
            st.markdown("<br>", unsafe_allow_html=True)

        target_date     = df['date'].iloc[-1] - pd.DateOffset(years=tahun_lalu)
        df_hist_point   = df[df['date'] >= target_date]

        if not df_hist_point.empty:
            harga_beli_hist = float(df_hist_point.iloc[0][price_col])
            lot_beli        = int(modal_hist / (harga_beli_hist * 100))
            lembar_beli     = lot_beli * 100
            modal_efektif   = lembar_beli * harga_beli_hist

            df_dividen_hist      = df[df['date'] >= df_hist_point.iloc[0]['date']]
            total_div_per_lembar = float(df_dividen_hist['dividends'].sum()) if 'dividends' in df_dividen_hist.columns else 0
            total_dividen_diterima = total_div_per_lembar * lembar_beli

            nilai_sekarang  = lembar_beli * harga_akhir
            capital_gain    = nilai_sekarang - modal_efektif
            total_return_hist = capital_gain + total_dividen_diterima

            rh1, rh2, rh3, rh4 = st.columns(4)
            with rh1:
                st.markdown(f"<div class='card hold'><div class='label'>Harga Beli ({tahun_lalu} thn lalu)</div><div class='metric-val'>Rp {harga_beli_hist:,.0f}</div></div>", unsafe_allow_html=True)
            with rh2:
                cap_color = "#2ecc71" if capital_gain >= 0 else "#e74c3c"
                st.markdown(f"<div class='card hold'><div class='label'>Capital Gain s/d Sekarang</div><div class='metric-val' style='color:{cap_color};'>Rp {capital_gain:+,.0f}</div></div>", unsafe_allow_html=True)
            with rh3:
                st.markdown(f"<div class='card dividend'><div class='label'>Total Dividen Diterima</div><div class='metric-val'>Rp {total_dividen_diterima:,.0f}</div></div>", unsafe_allow_html=True)
            with rh4:
                total_color = "#2ecc71" if total_return_hist >= 0 else "#e74c3c"
                st.markdown(f"<div class='card buy'><div class='label'>Total Return (CG + Dividen)</div><div class='metric-val' style='color:{total_color};'>Rp {total_return_hist:+,.0f}</div></div>", unsafe_allow_html=True)

            st.caption(
                f"Asumsi: Beli **{lembar_beli:,} lembar ({lot_beli} lot)** @ Rp {harga_beli_hist:,.0f} → modal efektif **Rp {modal_efektif:,.0f}**. "
                f"Sisa modal Rp {modal_hist - modal_efektif:,.0f} tidak diinvestasikan (tidak mencukupi 1 lot). "
                f"Nilai portofolio saat ini **Rp {nilai_sekarang:,.0f}**."
            )
        else:
            st.warning("Data historis tidak cukup untuk simulasi tahun yang dipilih.")

    # --- TAB 2: PROYEKSI MASA DEPAN ---
    with tab_future:
        st.caption(
            f"Proyeksi menggunakan CAGR historis saham **{ticker_input}** sebesar **{cagr_persen:.2f}%/thn** "
            f"selama {umur_tahun_total:.1f} tahun. Dividen diestimasi dari rata-rata historis per lembar. Ini adalah estimasi, bukan jaminan."
        )

        ft1, ft2, ft3 = st.columns(3)
        with ft1:
            modal_future = st.number_input("Modal Awal (Rp)", min_value=1_000_000, value=10_000_000, step=1_000_000, key="modal_future")
        with ft2:
            tahun_future = st.slider("Durasi Investasi (Tahun)", min_value=1, max_value=30, value=10, key="future_years")
        with ft3:
            cagr_input = st.number_input("Asumsi CAGR (%/thn)", value=round(cagr_persen, 2), step=1.0, key="future_cagr")

        lot_future         = int(modal_future / (harga_akhir * 100))
        lembar_future      = lot_future * 100
        modal_efektif_future = lembar_future * harga_akhir

        # Proyeksi harga akhir
        harga_proyeksi     = harga_akhir * ((1 + cagr_input / 100) ** tahun_future)
        nilai_akhir_future = lembar_future * harga_proyeksi
        capital_gain_future= nilai_akhir_future - modal_efektif_future

        # Proyeksi dividen: asumsi tumbuh 5%/thn dari rata-rata historis
        total_div_future = 0
        div_growth_rate  = 0.05
        for y in range(1, tahun_future + 1):
            div_tahun_ini = avg_div_per_tahun * ((1 + div_growth_rate) ** (y - 1))
            total_div_future += div_tahun_ini * lembar_future

        total_return_future = capital_gain_future + total_div_future

        rf1, rf2, rf3, rf4 = st.columns(4)
        with rf1:
            st.markdown(f"<div class='card hold'><div class='label'>Harga Beli Sekarang</div><div class='metric-val'>Rp {harga_akhir:,.0f}</div></div>", unsafe_allow_html=True)
        with rf2:
            cap_f_color = "#2ecc71" if capital_gain_future >= 0 else "#e74c3c"
            st.markdown(f"<div class='card hold'><div class='label'>Est. Capital Gain ({tahun_future} Thn)</div><div class='metric-val' style='color:{cap_f_color};'>Rp {capital_gain_future:+,.0f}</div></div>", unsafe_allow_html=True)
        with rf3:
            st.markdown(f"<div class='card dividend'><div class='label'>Est. Total Dividen ({tahun_future} Thn)</div><div class='metric-val'>Rp {total_div_future:,.0f}</div></div>", unsafe_allow_html=True)
        with rf4:
            total_f_color = "#2ecc71" if total_return_future >= 0 else "#e74c3c"
            st.markdown(f"<div class='card buy'><div class='label'>Total Return Proyeksi</div><div class='metric-val' style='color:{total_f_color};'>Rp {total_return_future:+,.0f}</div></div>", unsafe_allow_html=True)

        st.caption(
            f"Asumsi: Beli **{lembar_future:,} lembar ({lot_future} lot)** @ Rp {harga_akhir:,.0f} → modal efektif **Rp {modal_efektif_future:,.0f}**. "
            f"Proyeksi dividen tumbuh {div_growth_rate*100:.0f}%/thn dari rata-rata historis Rp {avg_div_per_tahun:,.1f}/lembar/thn. "
            f"Estimasi nilai portofolio akhir: **Rp {nilai_akhir_future:,.0f}**."
        )

    # =====================================================
    # DCA SIMULATOR (hanya Investasi Jangka Panjang)
    # =====================================================
    if mode == "Investasi Jangka Panjang":
        st.divider()
        st.subheader("💸 Simulasi Dollar Cost Averaging (DCA)")
        st.caption(
            f"Proyeksi ini menggunakan setoran rutin bulanan. Return default diisi otomatis berdasarkan CAGR historis saham "
            f"**{ticker_input}** selama **{umur_tahun_total:.1f} tahun** terakhir ({cagr_persen:.2f}%)."
        )

        dca_c1, dca_c2, dca_c3 = st.columns(3)
        with dca_c1:
            monthly_invest = st.number_input("Setoran Bulanan (Rp)", min_value=100_000, value=1_500_000, step=100_000)
        with dca_c2:
            dca_years = st.slider("Durasi Investasi Ke Depan (Tahun)", min_value=1, max_value=20, value=5)
        with dca_c3:
            default_cagr = float(round(cagr_persen, 2))
            annual_cagr  = st.number_input("Asumsi Return Tahunan (%)", value=default_cagr, step=1.0)

        months       = dca_years * 12
        monthly_rate = (annual_cagr / 100) / 12
        total_invested = monthly_invest * months

        if monthly_rate != 0:
            future_value = monthly_invest * (((1 + monthly_rate) ** months - 1) / monthly_rate)
        else:
            future_value = total_invested

        profit = future_value - total_invested

        res1, res2, res3 = st.columns(3)
        with res1:
            st.markdown(f"<div class='card hold'><div class='label'>Total Modal Disetor ({dca_years} Thn)</div><div class='metric-val'>Rp {total_invested:,.0f}</div></div>", unsafe_allow_html=True)
        with res2:
            st.markdown(f"<div class='card buy'><div class='label'>Estimasi Nilai Akhir Portofolio</div><div class='metric-val'>Rp {future_value:,.0f}</div></div>", unsafe_allow_html=True)
        with res3:
            profit_color = "#2ecc71" if profit >= 0 else "#e74c3c"
            profit_sign  = "+" if profit >= 0 else ""
            st.markdown(f"<div class='card fundamental'><div class='label'>Potensi Return (Capital Gain)</div><div class='metric-val' style='color:{profit_color};'>{profit_sign} Rp {profit:,.0f}</div></div>", unsafe_allow_html=True)

    # =====================================================
    # CHARTS
    # =====================================================
    st.divider()
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Price & Trends (with Bollinger Bands)")
        chart_df = df.tail(365)
        fig, ax  = plt.subplots(figsize=(10, 5))
        ax.plot(chart_df["date"], chart_df["close"],     label="Close",    color="cyan")
        ax.plot(chart_df["date"], chart_df["ma50"],      label="MA50",     linestyle="--", color="yellow")
        ax.plot(chart_df["date"], chart_df["ma200"],     label="MA200",    linestyle="--", color="red")
        ax.plot(chart_df["date"], chart_df["bb_upper"],  label="BB Upper", linestyle=":",  color="magenta", alpha=0.5)
        ax.plot(chart_df["date"], chart_df["bb_lower"],  label="BB Lower", linestyle=":",  color="magenta", alpha=0.5)
        ax.set_facecolor('black'); fig.patch.set_facecolor('black')
        ax.tick_params(colors='white'); ax.yaxis.label.set_color('white'); ax.xaxis.label.set_color('white')
        ax.legend(facecolor='black', labelcolor='white'); ax.grid(alpha=0.2, color='gray')
        st.pyplot(fig)

        st.subheader("RSI Momentum")
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.plot(chart_df["date"], chart_df["rsi"], color="lime")
        ax2.axhline(70, color="red",   linestyle="--", alpha=0.5)
        ax2.axhline(30, color="green", linestyle="--", alpha=0.5)
        ax2.set_facecolor('black'); fig2.patch.set_facecolor('black')
        ax2.tick_params(colors='white'); ax2.set_ylim(0, 100)
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
