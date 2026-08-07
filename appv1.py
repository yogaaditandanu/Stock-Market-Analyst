import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
.buy        { background: #1e3a2f; border-left: 5px solid #2ecc71; }
.sell       { background: #3a1e1e; border-left: 5px solid #e74c3c; }
.hold       { background: #2c2c2c; border-left: 5px solid #f1c40f; }
.dividend   { background: #1a237e; border-left: 5px solid #5c6bc0; }
.fundamental{ background: #283593; border-left: 5px solid #8e24aa; }
.pvb-band   { background: #1a2e1a; border-left: 5px solid #00e676; }
.divyield   { background: #1a1a2e; border-left: 5px solid #7c4dff; }
.metric-val { font-size: 24px; font-weight: bold; }
.label      { font-size: 12px; color: #b0b3b8; }
.analyst-text { font-size: 15px; line-height: 1.6; color: #e0e0e0; margin-bottom: 10px; }
.signal-badge { display:inline-block; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:bold; margin:2px; }
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

st.sidebar.header("🎛️ Advanced Settings")
pvb_window_years = st.sidebar.slider("PBV Band — Lookback (Tahun)", 1, 5, 3)
div_yield_window_years = st.sidebar.slider("Div Yield Band — Lookback (Tahun)", 1, 5, 5)

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

@st.cache_data(ttl=3600)
def get_balance_sheet(symbol):
    try:
        t = yf.Ticker(symbol)
        bs = t.quarterly_balance_sheet
        return bs
    except Exception:
        return pd.DataFrame()

# =========================================================
# 4. HELPER: PBV Band (Historical)
# =========================================================
def compute_pbv_band(df_close, balance_sheet_df, window_years=3, currency=None, ticker=None, fx_rate=16000):
    """
    Hitung PBV harian dari close price dan book value per share historis.
    Returns: Series pbv_series, mean, std, current_pbv
    """
    if balance_sheet_df.empty:
        return None, None, None, None

    # Cari total equity & shares outstanding dari balance sheet kuartalan
    equity_rows = [r for r in balance_sheet_df.index if 'Stockholders' in str(r) or 'stockholders' in str(r) or 'Equity' in str(r)]
    share_rows  = [r for r in balance_sheet_df.index if 'Share' in str(r) and 'Issued' in str(r)]

    if not equity_rows:
        equity_rows = [r for r in balance_sheet_df.index if 'equity' in str(r).lower()]
    if not share_rows:
        share_rows = [r for r in balance_sheet_df.index if 'ordinary' in str(r).lower() or 'common' in str(r).lower()]

    if not equity_rows or not share_rows:
        return None, None, None, None

    equity_row = equity_rows[0]
    share_row  = share_rows[0]

    equity_series = balance_sheet_df.loc[equity_row].dropna().sort_index()
    shares_series = balance_sheet_df.loc[share_row].dropna().sort_index()

    bvps_dict = {}
    for col in equity_series.index:
        if col in shares_series.index:
            eq  = float(equity_series[col])
            sh  = float(shares_series[col])
            if sh and sh != 0:
                bvps_dict[col] = eq / sh

    if not bvps_dict:
        return None, None, None, None

    bvps_series = pd.Series(bvps_dict).sort_index()

    # FIX: Normalisasi BVPS jika laporan keuangan (balance sheet) dalam USD
    # sementara harga saham (df_close) dalam IDR. Tanpa ini, PBV akan
    # membengkak ~kurs USD/IDR kali lipat (mis. 16.000x lebih besar).
    if currency == "USD" and ticker and ticker.endswith(".JK"):
        bvps_series = bvps_series * fx_rate

    # Reindex ke harian dan forward-fill
    date_range = df_close.index
    bvps_daily = bvps_series.reindex(date_range, method='ffill').ffill().bfill()

    # Hitung PBV harian
    pbv_daily = df_close / bvps_daily
    pbv_daily = pbv_daily.dropna()

    cutoff = pd.Timestamp.now() - pd.DateOffset(years=window_years)
    pbv_window = pbv_daily[pbv_daily.index >= cutoff]

    if len(pbv_window) < 30:
        pbv_window = pbv_daily

    mean_pbv = float(pbv_window.mean())
    std_pbv  = float(pbv_window.std())
    curr_pbv = float(pbv_daily.iloc[-1]) if len(pbv_daily) > 0 else None

    return pbv_daily, mean_pbv, std_pbv, curr_pbv

# =========================================================
# 5. HELPER: Dividend Yield Band (Historical)
# =========================================================
def compute_div_yield_band(df, window_years=5):
    """
    Hitung Dividend Yield Band dari DPS TTM (Trailing 12M) / Close.
    """
    if 'dividends' not in df.columns or df['dividends'].sum() == 0:
        return None, None, None, None

    df2 = df.set_index('date') if 'date' in df.columns else df.copy()
    div_ttm = df2['dividends'].rolling(252, min_periods=1).sum()
    yield_series = (div_ttm / df2['close']).dropna()
    yield_series = yield_series[yield_series > 0]

    if len(yield_series) < 30:
        return None, None, None, None

    cutoff = pd.Timestamp.now() - pd.DateOffset(years=window_years)
    yield_window = yield_series[yield_series.index >= cutoff]
    if len(yield_window) < 30:
        yield_window = yield_series

    mean_y = float(yield_window.mean())
    std_y  = float(yield_window.std())
    curr_y = float(yield_series.iloc[-1])

    return yield_series, mean_y, std_y, curr_y

# =========================================================
# 6. HELPER: Bullish MACD Divergence
# =========================================================
def detect_macd_divergence(df, lookback=60):
    """
    Deteksi Bullish Divergence: harga Lower Low tapi MACD Higher Low.
    Returns: bool, description string
    """
    sub = df.tail(lookback).copy()
    if len(sub) < 20:
        return False, ""

    prices = sub['close'].values
    macds  = sub['macd'].values

    # Cari 2 lembah (trough) terakhir pada harga
    from scipy.signal import argrelmin
    price_troughs = argrelmin(prices, order=5)[0]
    macd_troughs  = argrelmin(macds,  order=5)[0]

    if len(price_troughs) < 2 or len(macd_troughs) < 2:
        return False, ""

    # Bandingkan 2 trough terakhir
    pt1, pt2 = price_troughs[-2], price_troughs[-1]
    mt1, mt2 = macd_troughs[-2],  macd_troughs[-1]

    price_lower_low = prices[pt2] < prices[pt1]
    macd_higher_low = macds[mt2]  > macds[mt1]

    if price_lower_low and macd_higher_low:
        desc = (f"✅ **Bullish MACD Divergence Terdeteksi!** Harga membentuk Lower Low "
                f"(Rp {prices[pt1]:,.0f} → Rp {prices[pt2]:,.0f}), namun MACD membentuk Higher Low "
                f"({macds[mt1]:.2f} → {macds[mt2]:.2f}). Tekanan jual melemah — potensi reversal naik meningkat signifikan.")
        return True, desc
    return False, ""

# =========================================================
# 7. HELPER: Volume Profile / Point of Control (POC)
# =========================================================
def compute_volume_poc(df, n_bins=50, lookback=252):
    """
    Hitung Volume Profile dan Point of Control (POC) dari data historis.
    Returns: poc_price, vol_at_each_bin (dict)
    """
    sub = df.tail(lookback).copy()
    if 'volume' not in sub.columns or sub['volume'].sum() == 0:
        return None, None, None

    price_min = sub['close'].min()
    price_max = sub['close'].max()
    bins = np.linspace(price_min, price_max, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    vol_profile = np.zeros(n_bins)

    for _, row in sub.iterrows():
        for i in range(n_bins):
            if bins[i] <= row['close'] < bins[i+1]:
                vol_profile[i] += row['volume']
                break

    poc_idx   = np.argmax(vol_profile)
    poc_price = bin_centers[poc_idx]
    poc_vol   = vol_profile[poc_idx]

    return poc_price, vol_profile, bin_centers

# =========================================================
# 8. HELPER: Average Price (DCA Weighted)
# =========================================================
def compute_weighted_avg(prices, quantities):
    total_qty = sum(quantities)
    if total_qty == 0:
        return 0
    return sum(p * q for p, q in zip(prices, quantities)) / total_qty

# =========================================================
# 9. MAIN EXECUTION
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
    currency   = stock_info.get("financialCurrency", "IDR") # Cek mata uang laporan

# FIX: Normalisasi PBV jika laporan keuangan dalam USD
    usd_idr_rate = 16000  # Asumsi kurs kasar (bisa disesuaikan atau ditarik via API)
    if currency == "USD" and ticker_input.endswith(".JK"):
        if pb_ratio:
            pb_ratio = pb_ratio / usd_idr_rate

    roe        = stock_info.get("returnOnEquity",  None)
    market_cap = stock_info.get("marketCap",       None)
    div_yield  = stock_info.get("dividendYield",   None)

    pe_display  = f"{pe_ratio:.2f}x"    if pe_ratio   else "—"
    pb_display  = f"{pb_ratio:.2f}x"    if pb_ratio   else "—"
    roe_display = f"{(roe*100):.2f}%"   if roe        else "—"

    if div_yield is not None:
        div_yield_display = f"{div_yield:.2f}%" if div_yield > 1 else f"{(div_yield*100):.2f}%"
    else:
        div_yield_display = "—"

    if market_cap:
        if   market_cap >= 1e12: mcap_display = f"Rp {market_cap/1e12:.2f} T"
        elif market_cap >= 1e9:  mcap_display = f"Rp {market_cap/1e9:.2f} M"
        else:                    mcap_display = f"Rp {market_cap:,.0f}"
    else:
        mcap_display = "—"

    # =====================================================
    # PBV BAND COMPUTATION
    # =====================================================
    df_close_indexed = df.set_index('date')['close'] if 'date' in df.columns else df['close']
    balance_sheet_df = get_balance_sheet(ticker_input)
    pbv_series, pbv_mean, pbv_std, current_pbv = compute_pbv_band(
        df_close_indexed, balance_sheet_df, window_years=pvb_window_years,
        currency=currency, ticker=ticker_input, fx_rate=usd_idr_rate
    )

    # Fallback: gunakan pb_ratio dari yfinance jika komputasi gagal
    if current_pbv is None and pb_ratio:
        current_pbv = pb_ratio
        pbv_mean    = pb_ratio
        pbv_std     = pb_ratio * 0.3  # estimasi kasar

    # Guard tambahan: kalau hasil komputasi masih tidak masuk akal
    # (mis. beda >50x dari pb_ratio resmi yfinance), pakai pb_ratio sebagai gantinya.
    if current_pbv is not None and pb_ratio and pb_ratio > 0:
        if current_pbv > pb_ratio * 50 or current_pbv < pb_ratio / 50:
            current_pbv = pb_ratio
            pbv_mean    = pb_ratio
            pbv_std     = pb_ratio * 0.3
            pbv_series  = None

    # PBV SD Level
    pbv_signal = None
    pbv_signal_text = ""
    if current_pbv is not None and pbv_mean is not None and pbv_std is not None and pbv_std > 0:
        pbv_z = (current_pbv - pbv_mean) / pbv_std
        if pbv_z <= -2:
            pbv_signal = "DEEP VALUE (-2 SD)"
            pbv_signal_text = (f"🔥 **PBV Deep Value (-2 SD):** PBV saat ini {current_pbv:.2f}x jauh di bawah rata-rata historis "
                               f"{pbv_mean:.2f}x (z-score: {pbv_z:.2f}). Secara statistik ini adalah level langka — probabilitas "
                               f"harga turun lebih dalam sudah sangat mengecil. **Sinyal eksekusi beli terkuat.**")
        elif pbv_z <= -1:
            pbv_signal = "VALUE ZONE (-1 SD)"
            pbv_signal_text = (f"✅ **PBV Value Zone (-1 SD):** PBV {current_pbv:.2f}x berada di bawah -1 SD dari rata-rata historis "
                               f"{pbv_mean:.2f}x (z-score: {pbv_z:.2f}). Saham sedang diskon — area akumulasi yang menarik.")
        elif pbv_z >= 2:
            pbv_signal = "OVERVALUED (+2 SD)"
            pbv_signal_text = (f"⚠️ **PBV Overvalued (+2 SD):** PBV {current_pbv:.2f}x melampaui +2 SD di atas rata-rata historis "
                               f"{pbv_mean:.2f}x (z-score: {pbv_z:.2f}). Saham diperdagangkan sangat premium — risiko mean reversion tinggi.")
        elif pbv_z >= 1:
            pbv_signal = "PREMIUM (+1 SD)"
            pbv_signal_text = (f"⚠️ **PBV Premium (+1 SD):** PBV {current_pbv:.2f}x di atas rata-rata historis {pbv_mean:.2f}x "
                               f"(z-score: {pbv_z:.2f}). Bukan momen terbaik untuk menambah posisi.")
        else:
            pbv_signal = "FAIR VALUE"
            pbv_signal_text = (f"📊 **PBV Fair Value:** PBV {current_pbv:.2f}x mendekati rata-rata historis {pbv_mean:.2f}x "
                               f"(z-score: {pbv_z:.2f}). Valuasi wajar, bukan diskon namun juga tidak mahal.")
    else:
        pbv_z = None

    # =====================================================
    # DIVIDEND YIELD BAND COMPUTATION
    # =====================================================
    dy_series, dy_mean, dy_std, current_dy = compute_div_yield_band(df, window_years=div_yield_window_years)

    dy_signal = None
    dy_signal_text = ""
    if current_dy is not None and dy_mean is not None and dy_std is not None and dy_std > 0:
        dy_z = (current_dy - dy_mean) / dy_std
        if dy_z >= 2:
            dy_signal = "YIELD SANGAT TINGGI (+2 SD)"
            dy_signal_text = (f"🔥 **Dividend Yield Ekstrem Tinggi (+2 SD):** Yield saat ini {current_dy*100:.2f}% vs rata-rata historis "
                              f"{dy_mean*100:.2f}% (z-score: {dy_z:.2f}). Harga anjlok sementara dividen tetap — ini adalah sinyal beli "
                              f"paling langka bagi investor dividen. Yield melebihi deposito bank jauh.")
        elif dy_z >= 1:
            dy_signal = "YIELD TINGGI (+1 SD)"
            dy_signal_text = (f"✅ **Dividend Yield Tinggi (+1 SD):** Yield {current_dy*100:.2f}% di atas rata-rata historis "
                              f"{dy_mean*100:.2f}% (z-score: {dy_z:.2f}). Harga diskon relatif terhadap dividen — area menarik untuk akumulasi.")
        elif dy_z <= -1:
            dy_signal = "YIELD RENDAH (-1 SD)"
            dy_signal_text = (f"⚠️ **Dividend Yield Rendah (-1 SD):** Yield {current_dy*100:.2f}% di bawah rata-rata historis "
                              f"{dy_mean*100:.2f}% (z-score: {dy_z:.2f}). Harga terlalu mahal relatif terhadap dividen.")
        else:
            dy_signal = "YIELD NORMAL"
            dy_signal_text = (f"📊 **Dividend Yield Normal:** Yield {current_dy*100:.2f}% mendekati rata-rata historis "
                              f"{dy_mean*100:.2f}% (z-score: {dy_z:.2f}).")
    else:
        dy_z = None

    # =====================================================
    # MACD BULLISH DIVERGENCE
    # =====================================================
    try:
        from scipy.signal import argrelmin as _arm
        macd_div, macd_div_text = detect_macd_divergence(df, lookback=90)
    except ImportError:
        macd_div      = False
        macd_div_text = "⚠️ scipy tidak tersedia — deteksi divergence dinonaktifkan."

    # =====================================================
    # VOLUME PROFILE / POC
    # =====================================================
    poc_price, vol_profile, bin_centers = compute_volume_poc(df, lookback=252)
    poc_signal_text = ""
    if poc_price is not None:
        distance_pct = (float(latest["close"]) - poc_price) / poc_price * 100
        if abs(distance_pct) <= 3:
            poc_signal_text = (f"🎯 **Harga Mendekati POC (Point of Control):** Harga saat ini Rp {latest['close']:,.0f} berada "
                               f"hanya {abs(distance_pct):.1f}% dari POC di Rp {poc_price:,.0f} — area dengan transaksi institusional "
                               f"paling tebal dalam 1 tahun terakhir. Support sangat kuat.")
        elif distance_pct < -3:
            poc_signal_text = (f"📌 **POC Support di Atas:** Harga Rp {latest['close']:,.0f} berada {abs(distance_pct):.1f}% "
                               f"di bawah POC Rp {poc_price:,.0f}. Jika harga kembali ke POC, area tersebut akan menjadi resistance kuat.")
        else:
            poc_signal_text = (f"📌 **POC Support Valid:** POC di Rp {poc_price:,.0f} — harga saat ini {distance_pct:.1f}% di atasnya. "
                               f"Jika terjadi koreksi ke level POC, itu adalah kesempatan akumulasi bagi institusi.")

    # =====================================================
    # RSI MOMENTUM SIGNAL
    # =====================================================
    rsi_val = float(latest["rsi"])
    rsi_signal = ""
    if rsi_val > 70:
        rsi_signal = "⛔ OVERBOUGHT — Tunda DCA bulan ini, simpan di RDN"
    elif rsi_val < 20:
        rsi_signal = "🔥 EXTREME OVERSOLD (<20) — Sinyal beli paling presisi"
    elif rsi_val < 30:
        rsi_signal = "✅ OVERSOLD — Momentum masuk DCA sangat baik"
    else:
        rsi_signal = "📊 NORMAL — Tidak ada sinyal ekstrem"

    # =====================================================
    # DECISION ENGINE
    # =====================================================
    decision = "HOLD"
    reasons  = []

    # --- Skor komposit sinyal baru ---
    bullish_signals = 0
    bearish_signals = 0

    if pbv_z is not None:
        if pbv_z <= -1: bullish_signals += 1
        if pbv_z >= 1:  bearish_signals += 1

    if dy_z is not None:
        if dy_z >= 1:   bullish_signals += 1
        if dy_z <= -1:  bearish_signals += 1

    if macd_div:
        bullish_signals += 1

    if poc_price is not None and abs((float(latest["close"]) - poc_price) / poc_price * 100) <= 3:
        bullish_signals += 1

    # FIX: RSI ekstrem sekarang ikut masuk skor komposit untuk mode
    # Investasi Jangka Panjang. Sebelumnya RSI hanya dipakai untuk
    # tampilan/DCA calculator, sehingga sinyal BUY yang murni berasal
    # dari trend MA200 tidak pernah "ditahan" walau RSI overbought
    # dan PBV sudah premium secara bersamaan.
    if rsi_val > 70:
        bearish_signals += 1
    elif rsi_val < 30:
        bullish_signals += 1

    if mode == "Investasi Jangka Panjang":
        if roe and roe > 0.15:
            reasons.append("Fundamental Kuat: Tingkat profitabilitas perusahaan sangat baik (ROE > 15%). Manajemen terbukti mampu memutar modal ekuitas secara efisien untuk menghasilkan laba.")
        elif roe and roe < 0.05:
            reasons.append("Fundamental Kurang Efisien: Tingkat profitabilitas tergolong lemah (ROE < 5%). Ada tantangan bagi perusahaan dalam mencetak laba maksimal dari modal yang ditanamkan.")

        if pe_ratio and pe_ratio < 15:
            reasons.append("Valuasi Atraktif: Berdasarkan P/E ratio di bawah 15x, harga saham tergolong cukup murah (Undervalued) dibandingkan kinerja labanya. Memberikan margin of safety yang baik.")

        if latest["close"] > latest["ma200"]:
            reasons.append("Tren Harga Positif: Secara historis jangka panjang, saham berada di fase Uptrend karena posisinya konsisten di atas MA200. Mengonfirmasi adanya akumulasi bertahap.")
            decision = "AKUMULASI / BUY"
        else:
            reasons.append("Tren Harga Negatif: Saham dalam bayang-bayang Downtrend jangka panjang karena tertekan di bawah MA200. Disarankan menunggu konfirmasi pembalikan arah.")
            decision = "WAIT & SEE / HOLD"

        # Upgrade/downgrade berdasarkan sinyal komposit baru
        if bullish_signals >= 3:
            decision = "STRONG AKUMULASI ⚡"
        elif bullish_signals >= 2 and decision == "AKUMULASI / BUY":
            decision = "AKUMULASI AGRESIF 🚀"
        elif bearish_signals >= 2:
            decision = "TAHAN / WAIT & SEE ⚠️"

        # FIX: override eksplisit — RSI overbought tunggal saja cukup untuk
        # menahan entry baru, terlepas dari skor komposit lainnya. Trend
        # jangka panjang boleh tetap positif, tapi entry/tambah posisi
        # sebaiknya ditunda saat momentum jangka pendek sudah jenuh beli.
        if rsi_val > 70 and decision in ("AKUMULASI / BUY", "STRONG AKUMULASI ⚡", "AKUMULASI AGRESIF 🚀"):
            decision = "TUNGGU KOREKSI / HOLD ⚠️ (RSI Overbought)"
            reasons.append(f"Peringatan Momentum: RSI berada di {rsi_val:.1f} (>70, Overbought). Trend jangka panjang masih positif, namun entry/menambah posisi sebaiknya ditunda sampai terjadi koreksi harga atau RSI mendingin.")

    else:
        if (latest["close"] > latest["ma50"] and latest["ma50"] > latest["ma200"]):
            decision = "STRONG BUY"
            reasons.append("Struktur Bullish: Susunan teknikal solid (Golden Alignment). Harga di atas MA50, dan MA50 di atas MA200. Momentum pergerakan naik sangat kuat.")
        elif latest["close"] < latest["ma200"]:
            decision = "SELL"
            reasons.append("Struktur Bearish: Harga diperdagangkan di bawah MA200 yang bertindak sebagai resistance dinamis kuat. Menandakan tekanan jual yang dominan.")
        else:
            decision = "HOLD"
            reasons.append("Fase Konsolidasi: Pergerakan harga tidak menunjukkan kecenderungan arah tren yang ekstrem di antara garis Moving Average.")

        if latest["rsi"] > 75:
            decision = "SELL"
            reasons.append("Peringatan RSI: Momentum masuk area Overbought (>75). Probabilitas tinggi terjadinya aksi profit-taking yang dapat memicu koreksi jangka pendek.")
        elif latest["rsi"] < 35:
            if decision == "SELL":   decision = "HOLD"
            elif decision == "HOLD": decision = "BUY"
            reasons.append("Peluang RSI: Momentum menyentuh area Oversold (<35). Sering kali membuka ruang bagi pembeli spekulatif untuk mendorong technical rebound.")

        if latest["macd"] > latest["signal_line"]:
            reasons.append("MACD Positif: Garis MACD memotong ke atas Signal Line (Golden Cross), mengonfirmasi akselerasi momentum kenaikan harga.")
        else:
            reasons.append("MACD Negatif: Garis MACD berada di bawah Signal Line (Death Cross), memberikan sinyal awal pelemahan momentum dan tekanan distribusi.")

        if latest["close"] < latest["bb_lower"]:
            if decision == "HOLD": decision = "BUY"
            reasons.append("Bollinger Extreme Low: Harga telah menembus batas bawah (Lower Band). Mengindikasikan kepanikan jual berlebih dan probabilitas kuat untuk memantul naik.")
        elif latest["close"] > latest["bb_upper"]:
            if decision in ("BUY", "STRONG BUY"): decision = "HOLD"
            reasons.append("Bollinger Extreme High: Harga menembus batas atas (Upper Band). Saham sudah terlalu 'mahal' dalam jangka pendek, waspadai potensi pullback ke harga rata-rata.")

        # Override bullish sinyal komposit
        if macd_div and decision in ("SELL", "HOLD"):
            decision = "WATCH / POTENTIAL BUY"

    # =====================================================
    # ML PROJECTION (Linear Regression)
    # =====================================================
    model_df = df.tail(60).dropna()
    X = np.arange(len(model_df)).reshape(-1, 1)
    y = model_df["close"].values
    model = LinearRegression()
    model.fit(X, y)
    reg_pred    = model.predict([[len(model_df) + horizon]])[0]
    stop_loss   = latest["close"] * 0.95
    take_profit = latest["close"] * 1.10

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
    st.title(f"📈 {title_name} — {mode}")

    # Composite Signal Score Badge
    total_signals = bullish_signals + bearish_signals
    score_pct = (bullish_signals / total_signals * 100) if total_signals > 0 else 50
    score_color = "#2ecc71" if score_pct >= 60 else ("#e74c3c" if score_pct <= 40 else "#f1c40f")
    st.markdown(
        f"<div style='padding:10px 16px; border-radius:10px; background:#1e1e1e; margin-bottom:10px; "
        f"border-left:5px solid {score_color};'>"
        f"<span style='color:#b0b3b8; font-size:13px;'>Composite Signal Score (Bullish Signals): </span>"
        f"<span style='color:{score_color}; font-size:22px; font-weight:bold;'>{bullish_signals}/{bullish_signals+bearish_signals}</span>"
        f"<span style='color:#b0b3b8; font-size:13px;'> dari indikator kuantitatif (PBV Band, Div Yield Band, MACD Divergence, POC, RSI)</span>"
        f"</div>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    color = "buy" if ("BUY" in decision or "AKUMULASI" in decision or "STRONG" in decision) else ("sell" if "SELL" in decision else "hold")

    with m1:
        st.markdown(f"<div class='card {color}'><div class='label'>Signal ({mode})</div><div class='metric-val'>{decision}</div></div>", unsafe_allow_html=True)
    with m2:
        diff = ((latest["close"] - prev_close) / prev_close) * 100
        st.markdown(f"<div class='card hold'><div class='label'>Last Price</div><div class='metric-val'>Rp {latest['close']:,.0f} ({diff:+.2f}%)</div></div>", unsafe_allow_html=True)
    with m3:
        rsi_color = "sell" if rsi_val > 70 else ("buy" if rsi_val < 30 else "hold")
        st.markdown(f"<div class='card {rsi_color}'><div class='label'>RSI (Momentum)</div><div class='metric-val'>{rsi_val:.1f}</div><div class='label'>{rsi_signal}</div></div>", unsafe_allow_html=True)
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
    # [NEW] PBV BAND + DIV YIELD BAND CARDS
    # =====================================================
    st.divider()
    st.subheader("📐 Analisa Valuasi Statistik — PBV Band & Dividend Yield Band")

    pv1, pv2 = st.columns(2)
    with pv1:
        if current_pbv is not None and pbv_mean is not None:
            pbv_color = "buy" if (pbv_z is not None and pbv_z <= -1) else ("sell" if (pbv_z is not None and pbv_z >= 1) else "hold")
            sd_bands = ""
            if pbv_std:
                sd_bands = (f"−2SD: {pbv_mean-2*pbv_std:.2f}x | −1SD: {pbv_mean-pbv_std:.2f}x | "
                            f"Mean: {pbv_mean:.2f}x | +1SD: {pbv_mean+pbv_std:.2f}x | +2SD: {pbv_mean+2*pbv_std:.2f}x")
            st.markdown(
                f"<div class='card pvb-band'>"
                f"<div class='label'>PBV Saat Ini vs Band Historis ({pvb_window_years} Thn)</div>"
                f"<div class='metric-val'>{current_pbv:.2f}x"
                f"{'  🔥' if pbv_z is not None and pbv_z <=-2 else ('  ✅' if pbv_z is not None and pbv_z<=-1 else ('  ⚠️' if pbv_z is not None and pbv_z>=1 else ''))}"
                f"</div>"
                f"<div class='label'>{pbv_signal or '—'}</div>"
                f"<div class='label' style='margin-top:6px; font-size:11px; color:#80cfa0;'>{sd_bands}</div>"
                f"</div>", unsafe_allow_html=True)
            if pbv_signal_text:
                st.markdown(f"<div class='analyst-text'>{pbv_signal_text}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='card hold'><div class='label'>PBV Band</div><div class='metric-val'>Data tidak cukup</div><div class='label'>Balance sheet historis tidak tersedia dari sumber data</div></div>", unsafe_allow_html=True)

    with pv2:
        if current_dy is not None and dy_mean is not None:
            dy_color = "buy" if (dy_z is not None and dy_z >= 1) else ("sell" if (dy_z is not None and dy_z <= -1) else "hold")
            dy_bands = ""
            if dy_std:
                dy_bands = (f"−1SD: {(dy_mean-dy_std)*100:.2f}% | Mean: {dy_mean*100:.2f}% | "
                            f"+1SD: {(dy_mean+dy_std)*100:.2f}% | +2SD: {(dy_mean+2*dy_std)*100:.2f}%")
            st.markdown(
                f"<div class='card divyield'>"
                f"<div class='label'>Dividend Yield Saat Ini vs Band Historis ({div_yield_window_years} Thn)</div>"
                f"<div class='metric-val'>{current_dy*100:.2f}%"
                f"{'  🔥' if dy_z is not None and dy_z>=2 else ('  ✅' if dy_z is not None and dy_z>=1 else ('  ⚠️' if dy_z is not None and dy_z<=-1 else ''))}"
                f"</div>"
                f"<div class='label'>{dy_signal or '—'}</div>"
                f"<div class='label' style='margin-top:6px; font-size:11px; color:#b39ddb;'>{dy_bands}</div>"
                f"</div>", unsafe_allow_html=True)
            if dy_signal_text:
                st.markdown(f"<div class='analyst-text'>{dy_signal_text}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='card hold'><div class='label'>Dividend Yield Band</div><div class='metric-val'>Data tidak tersedia</div><div class='label'>Perusahaan tidak membagikan dividen atau data tidak cukup</div></div>", unsafe_allow_html=True)

    # =====================================================
    # [NEW] MACD DIVERGENCE + POC CARDS
    # =====================================================
    st.divider()
    st.subheader("🔬 Analisa Teknikal Kuantitatif — Divergence & Volume Profile")

    td1, td2 = st.columns(2)
    with td1:
        div_card_color = "buy" if macd_div else "hold"
        div_icon = "✅" if macd_div else "—"
        st.markdown(
            f"<div class='card {div_card_color}'>"
            f"<div class='label'>Bullish MACD Divergence (90 Hari Terakhir)</div>"
            f"<div class='metric-val'>{div_icon} {'TERDETEKSI' if macd_div else 'Tidak Ada'}</div>"
            f"</div>", unsafe_allow_html=True)
        if macd_div_text:
            st.markdown(f"<div class='analyst-text'>{macd_div_text}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='analyst-text' style='color:#888;'>Tidak ditemukan pola Bullish Divergence pada periode 90 hari terakhir. Pantau terus grafik MACD untuk potensi pembalikan arah.</div>", unsafe_allow_html=True)

    with td2:
        poc_color = "buy" if poc_price and abs((float(latest["close"]) - poc_price) / poc_price * 100) <= 3 else "hold"
        poc_display = f"Rp {poc_price:,.0f}" if poc_price else "—"
        st.markdown(
            f"<div class='card {poc_color}'>"
            f"<div class='label'>Volume Profile — Point of Control (POC) 252 Hari</div>"
            f"<div class='metric-val'>{poc_display}</div>"
            f"<div class='label'>Area support/resistance terkuat berbasis volume institusional</div>"
            f"</div>", unsafe_allow_html=True)
        if poc_signal_text:
            st.markdown(f"<div class='analyst-text'>{poc_signal_text}</div>", unsafe_allow_html=True)

    # =====================================================
    # FUNDAMENTAL ANALYST TEXT
    # =====================================================
    st.divider()
    st.subheader("🧠 Analisa Fundamental")
    fundamental_texts = []

    if pe_ratio:
        if pe_ratio < 10:
            fundamental_texts.append(f"📌 **Valuasi Sangat Murah (P/E {pe_ratio:.1f}x):** Saham diperdagangkan di bawah 10x earnings. Ini adalah level deep value yang menarik — kemungkinan pasar sedang underpricing kualitas bisnis ini, atau ada sentimen negatif sementara yang bisa jadi peluang.")
        elif pe_ratio < 15:
            fundamental_texts.append(f"📌 **Valuasi Wajar-Murah (P/E {pe_ratio:.1f}x):** Harga saham masih terbilang terjangkau dibandingkan profitnya. Memberikan margin of safety yang nyaman bagi investor jangka panjang.")
        elif pe_ratio < 25:
            fundamental_texts.append(f"📌 **Valuasi Moderat (P/E {pe_ratio:.1f}x):** Pasar membayar harga yang wajar. Saham tidak murah, namun juga tidak mahal — perlu pertumbuhan laba yang konsisten untuk membenarkan valuasi ini ke depan.")
        else:
            fundamental_texts.append(f"📌 **Valuasi Premium (P/E {pe_ratio:.1f}x):** Pasar mempricing saham ini dengan ekspektasi pertumbuhan tinggi. Cocok jika bisnis memang tumbuh pesat, namun berisiko koreksi keras jika laba sewaktu-waktu meleset dari ekspektasi.")
    else:
        fundamental_texts.append("📌 **P/E Ratio:** Data tidak tersedia. Kemungkinan perusahaan sedang merugi atau data belum terupdate di sumber data.")

    if pb_ratio:
        if pb_ratio < 1:
            fundamental_texts.append(f"📌 **PBV di Bawah 1x ({pb_ratio:.2f}x):** Harga saham lebih rendah dari nilai bukunya — secara teori kamu 'membeli aset seharga diskon'. Menarik, namun perlu dicek apakah ada alasan struktural mengapa bisnis dihargai serendah ini oleh pasar.")
        elif pb_ratio < 2:
            fundamental_texts.append(f"📌 **PBV Wajar ({pb_ratio:.2f}x):** Harga relatif proporsional terhadap nilai aset bersih perusahaan. Umumnya masih kategori menarik, terutama untuk sektor perbankan dan keuangan.")
        elif pb_ratio < 5:
            fundamental_texts.append(f"📌 **PBV Moderat ({pb_ratio:.2f}x):** Pasar memberi premium atas nilai buku, mencerminkan ekspektasi profitabilitas yang baik ke depan. Wajar selama ROE juga tinggi.")
        else:
            fundamental_texts.append(f"📌 **PBV Tinggi ({pb_ratio:.2f}x):** Premium valuasi sangat tinggi. Biasanya hanya terjustifikasi jika ROE konsisten tinggi dan bisnis memiliki economic moat yang kuat dan sulit ditiru pesaing.")
    else:
        fundamental_texts.append("📌 **Price to Book (PBV):** Data tidak tersedia.")

    if roe:
        roe_pct = roe * 100
        if roe_pct >= 20:
            fundamental_texts.append(f"📌 **ROE Excellent ({roe_pct:.1f}%):** Kemampuan perusahaan menghasilkan laba dari modal ekuitas tergolong kelas atas. Ini adalah tanda bisnis berkualitas tinggi dengan competitive advantage yang nyata dan konsisten.")
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

    if div_yield is not None:
        dy_val = div_yield * 100 if div_yield <= 1 else div_yield
        if dy_val >= 5:
            fundamental_texts.append(f"📌 **Dividend Yield Tinggi ({dy_val:.2f}%):** Imbal hasil dividen sangat menarik, melebihi rata-rata deposito. Cocok untuk investor yang fokus pada passive income dari dividen rutin.")
        elif dy_val >= 2:
            fundamental_texts.append(f"📌 **Dividend Yield Moderat ({dy_val:.2f}%):** Perusahaan rutin membagikan dividen dengan yield yang wajar. Menjadi bonus tambahan di atas potensi capital gain jangka panjang.")
        elif dy_val > 0:
            fundamental_texts.append(f"📌 **Dividend Yield Rendah ({dy_val:.2f}%):** Perusahaan membagi dividen, namun nilainya kecil. Kemungkinan sebagian besar laba diputar kembali untuk pertumbuhan bisnis (reinvestment).")
    else:
        fundamental_texts.append("📌 **Dividend Yield:** Tidak ada data dividen atau perusahaan tidak membagikan dividen saat ini.")

    for text in fundamental_texts:
        st.markdown(f"<div class='analyst-text'>{text}</div>", unsafe_allow_html=True)

    # =====================================================
    # [NEW] DCA SMART — TAKTIK PEMBOBOTAN LOT
    # =====================================================
    if mode == "Investasi Jangka Panjang":
        st.divider()
        st.subheader("🧮 DCA Smart — Kalkulator Taktik Pembobotan Lot")
        st.caption(
            "Masukkan histori pembelian sahammu untuk mendapatkan rekomendasi jumlah lot bulan ini berdasarkan "
            "perbandingan harga saat ini vs average price, kondisi RSI, dan PBV Band."
        )

        dca_c1, dca_c2 = st.columns([1, 1])
        with dca_c1:
            st.markdown("**Histori Pembelian (Tambah Baris)**")
            n_entries = st.number_input("Jumlah Transaksi Sebelumnya", min_value=1, max_value=20, value=3, step=1)

            buy_prices = []
            buy_lots   = []
            for i in range(int(n_entries)):
                cols = st.columns([2, 1])
                with cols[0]:
                    bp = st.number_input(f"Harga Beli #{i+1} (Rp)", min_value=1, value=int(latest["close"]), step=10, key=f"bp_{i}")
                with cols[1]:
                    bl = st.number_input(f"Lot #{i+1}", min_value=1, value=1, step=1, key=f"bl_{i}")
                buy_prices.append(bp)
                buy_lots.append(bl)

        with dca_c2:
            if buy_prices and buy_lots:
                avg_price = compute_weighted_avg(buy_prices, [l*100 for l in buy_lots])
                total_lot = sum(buy_lots)
                total_lembar = total_lot * 100
                harga_skrg = float(latest["close"])
                diff_vs_avg = (harga_skrg - avg_price) / avg_price * 100

                # Rekomendasi lot
                rec_lot = 1
                rec_reason = []
                rec_color  = "#f1c40f"

                if rsi_val > 70:
                    rec_lot = 0
                    rec_reason.append("RSI Overbought (>70): **Simpan uang di RDN, jangan beli bulan ini.**")
                    rec_color = "#e74c3c"
                elif diff_vs_avg <= -10 and (pbv_z is not None and pbv_z <= -1):
                    rec_lot = 3
                    rec_reason.append(f"Harga {abs(diff_vs_avg):.1f}% di bawah avg price DAN PBV murah secara statistik ({pbv_signal}): **Beli 2-3 lot agresif untuk membanting average price.**")
                    rec_color = "#2ecc71"
                elif diff_vs_avg <= -10:
                    rec_lot = 2
                    rec_reason.append(f"Harga {abs(diff_vs_avg):.1f}% di bawah avg price: **Beli 2 lot untuk menurunkan average price.**")
                    rec_color = "#2ecc71"
                elif diff_vs_avg > 0:
                    rec_lot = 1
                    rec_reason.append(f"Harga {diff_vs_avg:.1f}% di atas avg price: **Beli 1 lot saja untuk menjaga kedisiplinan DCA.**")
                    rec_color = "#f1c40f"
                else:
                    rec_lot = 1
                    rec_reason.append("Harga mendekati avg price: **Beli 1 lot standar.**")
                    rec_color = "#f1c40f"

                if rsi_val < 20:
                    if rec_lot < 3:
                        rec_lot += 1
                    rec_reason.append(f"RSI Extreme Oversold ({rsi_val:.1f}): Tambah 1 lot ekstra — ini momen panik pasar yang langka.")
                    rec_color = "#2ecc71"

                # Proyeksi avg price baru
                new_total_lembar = total_lembar + rec_lot * 100
                new_modal        = sum(p * l * 100 for p, l in zip(buy_prices, buy_lots)) + harga_skrg * rec_lot * 100
                new_avg_price    = new_modal / new_total_lembar if new_total_lembar > 0 else avg_price

                st.markdown(
                    f"<div style='padding:16px; border-radius:12px; background:#1e1e1e; border-left:5px solid {rec_color};'>"
                    f"<div class='label'>Average Price Saat Ini</div>"
                    f"<div class='metric-val'>Rp {avg_price:,.0f}</div>"
                    f"<div class='label' style='color:#b0b3b8;'>Harga Saat Ini: Rp {harga_skrg:,.0f} "
                    f"({'▼' if diff_vs_avg < 0 else '▲'} {abs(diff_vs_avg):.1f}% vs avg)</div>"
                    f"</div>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                st.markdown(
                    f"<div style='padding:16px; border-radius:12px; background:#1e1e1e; border-left:5px solid {rec_color}; margin-top:8px;'>"
                    f"<div class='label'>Rekomendasi Pembelian Bulan Ini</div>"
                    f"<div class='metric-val' style='color:{rec_color};'>{rec_lot} Lot{'  🚫' if rec_lot==0 else ('  🔥' if rec_lot>=3 else '')}</div>"
                    f"<div class='label' style='margin-top:6px;'>Avg Price Baru Proyeksi: <b style='color:{rec_color};'>Rp {new_avg_price:,.0f}</b></div>"
                    f"</div>", unsafe_allow_html=True)

                for r in rec_reason:
                    st.markdown(f"<div class='analyst-text'>📌 {r}</div>", unsafe_allow_html=True)

                # Formula Average Price
                st.latex(r"\text{Average Price} = \frac{\sum (P_i \times Q_i)}{\sum Q_i} = \frac{" +
                         f"{new_modal:,.0f}" + r"}{" + f"{new_total_lembar:,}" + r"} = \text{Rp } " +
                         f"{new_avg_price:,.0f}")

    # =====================================================
    # SIMULATOR: HISTORIS & MASA DEPAN
    # =====================================================
    st.divider()
    st.subheader("🕰️ Simulator Investasi Historis & Proyeksi")

    tab_hist, tab_future = st.tabs(["📜 Jika Beli X Tahun Lalu", "🔮 Jika Beli Sekarang & Tahan X Tahun"])

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

        target_date   = df['date'].iloc[-1] - pd.DateOffset(years=tahun_lalu)
        df_hist_point = df[df['date'] >= target_date]

        if not df_hist_point.empty:
            harga_beli_hist   = float(df_hist_point.iloc[0]['close'])
            harga_skrg_aktual = float(df.iloc[-1]['close'])
            lot_beli          = int(modal_hist / (harga_beli_hist * 100))
            lembar_beli       = lot_beli * 100
            modal_efektif     = lembar_beli * harga_beli_hist

            df_dividen_hist        = df[df['date'] >= df_hist_point.iloc[0]['date']]
            total_div_per_lembar   = float(df_dividen_hist['dividends'].sum()) if 'dividends' in df_dividen_hist.columns else 0
            total_dividen_diterima = total_div_per_lembar * lembar_beli

            nilai_sekarang    = lembar_beli * harga_skrg_aktual
            capital_gain      = nilai_sekarang - modal_efektif
            capital_gain_pct  = (capital_gain / modal_efektif * 100) if modal_efektif > 0 else 0
            total_return_hist = capital_gain + total_dividen_diterima
            total_return_pct  = (total_return_hist / modal_efektif * 100) if modal_efektif > 0 else 0

            rh1, rh2, rh3, rh4 = st.columns(4)
            with rh1:
                st.markdown(f"<div class='card hold'><div class='label'>Harga Beli ({tahun_lalu} thn lalu)</div><div class='metric-val'>Rp {harga_beli_hist:,.0f}</div><div class='label'>Harga Sekarang: Rp {harga_skrg_aktual:,.0f}</div></div>", unsafe_allow_html=True)
            with rh2:
                cap_color = "#2ecc71" if capital_gain >= 0 else "#e74c3c"
                st.markdown(f"<div class='card hold'><div class='label'>Capital Gain s/d Sekarang</div><div class='metric-val' style='color:{cap_color};'>Rp {capital_gain:+,.0f}</div><div class='label' style='color:{cap_color};'>{capital_gain_pct:+.2f}% dari modal efektif</div></div>", unsafe_allow_html=True)
            with rh3:
                st.markdown(f"<div class='card dividend'><div class='label'>Total Dividen Tunai Diterima</div><div class='metric-val'>Rp {total_dividen_diterima:,.0f}</div><div class='label'>Rp {total_div_per_lembar:,.0f}/lembar kumulatif</div></div>", unsafe_allow_html=True)
            with rh4:
                total_color = "#2ecc71" if total_return_hist >= 0 else "#e74c3c"
                st.markdown(f"<div class='card buy'><div class='label'>Total Return (CG + Dividen)</div><div class='metric-val' style='color:{total_color};'>Rp {total_return_hist:+,.0f}</div><div class='label' style='color:{total_color};'>{total_return_pct:+.2f}% dari modal efektif</div></div>", unsafe_allow_html=True)

            st.caption(f"Asumsi: Beli **{lembar_beli:,} lembar ({lot_beli} lot)** @ Rp {harga_beli_hist:,.0f} → modal efektif **Rp {modal_efektif:,.0f}**. Sisa modal Rp {modal_hist - modal_efektif:,.0f} tidak diinvestasikan.")
        else:
            st.warning("Data historis tidak cukup untuk simulasi tahun yang dipilih.")

    with tab_future:
        st.caption(f"Proyeksi menggunakan CAGR historis saham **{ticker_input}** sebesar **{cagr_persen:.2f}%/thn** selama {umur_tahun_total:.1f} tahun.")

        ft1, ft2, ft3 = st.columns(3)
        with ft1:
            modal_future = st.number_input("Modal Awal (Rp)", min_value=1_000_000, value=10_000_000, step=1_000_000, key="modal_future")
        with ft2:
            tahun_future = st.slider("Durasi Investasi (Tahun)", min_value=1, max_value=30, value=10, key="future_years")
        with ft3:
            cagr_input = st.number_input("Asumsi CAGR (%/thn)", value=round(cagr_persen, 2), step=1.0, key="future_cagr")

        lot_future           = int(modal_future / (harga_akhir * 100))
        lembar_future        = lot_future * 100
        modal_efektif_future = lembar_future * harga_akhir
        harga_proyeksi       = harga_akhir * ((1 + cagr_input / 100) ** tahun_future)
        nilai_akhir_future   = lembar_future * harga_proyeksi
        capital_gain_future  = nilai_akhir_future - modal_efektif_future

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

        st.caption(f"Asumsi: Beli **{lembar_future:,} lembar ({lot_future} lot)** @ Rp {harga_akhir:,.0f} → modal efektif **Rp {modal_efektif_future:,.0f}**. Estimasi nilai portofolio akhir: **Rp {nilai_akhir_future:,.0f}**.")

    # =====================================================
    # DCA SIMULATOR — MONTE CARLO
    # =====================================================
    if mode == "Investasi Jangka Panjang":
        st.divider()
        st.subheader("💸 Simulasi DCA — Monte Carlo")
        st.caption(
            f"Simulasi menggunakan **1.000 skenario acak** berbasis distribusi return harian historis saham **{ticker_input}** "
            f"(mean harian: {df['close'].pct_change().mean()*100:.3f}%, volatilitas: {df['close'].pct_change().std()*100:.3f}%/hari)."
        )

        dca_c1, dca_c2 = st.columns(2)
        with dca_c1:
            monthly_invest = st.number_input("Setoran Bulanan (Rp)", min_value=100_000, value=1_500_000, step=100_000)
        with dca_c2:
            dca_years = st.slider("Durasi Investasi Ke Depan (Tahun)", min_value=1, max_value=20, value=5)

        daily_returns  = df['close'].pct_change().dropna()
        mu_daily       = float(daily_returns.mean())
        sigma_daily    = float(daily_returns.std())
        trading_days   = 252
        months_total   = dca_years * 12
        total_invested = monthly_invest * months_total

        np.random.seed(42)
        n_sim          = 1000
        n_days         = dca_years * trading_days
        days_per_month = trading_days // 12

        final_values = []
        sample_paths = []

        for i in range(n_sim):
            harga             = float(df['close'].iloc[-1])
            portofolio_lembar = 0
            kas_dividen       = 0
            monthly_counter   = 0
            path_value        = []

            for d in range(n_days):
                shock = np.random.normal(mu_daily, sigma_daily)
                harga = max(harga * (1 + shock), 1)

                if d % days_per_month == 0 and monthly_counter < months_total:
                    lot_bulan = int(monthly_invest / (harga * 100))
                    portofolio_lembar += lot_bulan * 100
                    monthly_counter   += 1

                if i < 50 and d % days_per_month == 0:
                    path_value.append(portofolio_lembar * harga)

            for y in range(1, dca_years + 1):
                kas_dividen += avg_div_per_tahun * (1.05 ** (y - 1)) * portofolio_lembar

            final_values.append(portofolio_lembar * harga + kas_dividen)
            if i < 50:
                sample_paths.append(path_value)

        final_values = np.array(final_values)
        p10 = np.percentile(final_values, 10)
        p50 = np.percentile(final_values, 50)
        p90 = np.percentile(final_values, 90)
        prob_profit = (final_values > total_invested).mean() * 100

        res1, res2, res3, res4 = st.columns(4)
        with res1:
            st.markdown(f"<div class='card hold'><div class='label'>Total Modal Disetor ({dca_years} Thn)</div><div class='metric-val'>Rp {total_invested:,.0f}</div></div>", unsafe_allow_html=True)
        with res2:
            p10_color = "#2ecc71" if p10 > total_invested else "#e74c3c"
            st.markdown(f"<div class='card sell'><div class='label'>😰 Skenario Buruk (10%)</div><div class='metric-val'>Rp {p10:,.0f}</div><div class='label' style='color:{p10_color};'>{p10-total_invested:+,.0f} ({(p10-total_invested)/total_invested*100:+.1f}%)</div></div>", unsafe_allow_html=True)
        with res3:
            p50_color = "#2ecc71" if p50 > total_invested else "#e74c3c"
            st.markdown(f"<div class='card hold'><div class='label'>😐 Skenario Median (50%)</div><div class='metric-val'>Rp {p50:,.0f}</div><div class='label' style='color:{p50_color};'>{p50-total_invested:+,.0f} ({(p50-total_invested)/total_invested*100:+.1f}%)</div></div>", unsafe_allow_html=True)
        with res4:
            p90_color = "#2ecc71" if p90 > total_invested else "#e74c3c"
            st.markdown(f"<div class='card buy'><div class='label'>🚀 Skenario Bagus (90%)</div><div class='metric-val'>Rp {p90:,.0f}</div><div class='label' style='color:{p90_color};'>{p90-total_invested:+,.0f} ({(p90-total_invested)/total_invested*100:+.1f}%)</div></div>", unsafe_allow_html=True)

        prob_color = "#2ecc71" if prob_profit >= 60 else ("#f1c40f" if prob_profit >= 40 else "#e74c3c")
        st.markdown(
            f"<div style='margin-top:10px; padding:12px 18px; border-radius:8px; background:#1e1e1e; border-left: 4px solid {prob_color};'>"
            f"<span style='color:#b0b3b8; font-size:13px;'>Probabilitas Untung dari 1.000 Simulasi: </span>"
            f"<span style='color:{prob_color}; font-size:20px; font-weight:bold;'>{prob_profit:.1f}%</span>"
            f"</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        fig_mc, ax_mc = plt.subplots(figsize=(10, 4))
        x_axis = list(range(len(sample_paths[0]))) if sample_paths else []
        if x_axis:
            for path in sample_paths:
                if len(path) == len(x_axis):
                    ax_mc.plot(x_axis, path, color='cyan', alpha=0.05, linewidth=0.8)
            path_matrix = np.array([p for p in sample_paths if len(p) == len(x_axis)])
            if len(path_matrix) > 0:
                ax_mc.plot(x_axis, np.percentile(path_matrix, 50, axis=0), color='yellow',  linewidth=2,   label='Median')
                ax_mc.plot(x_axis, np.percentile(path_matrix, 10, axis=0), color='#e74c3c', linewidth=1.5, linestyle='--', label='Skenario Buruk (10%)')
                ax_mc.plot(x_axis, np.percentile(path_matrix, 90, axis=0), color='#2ecc71', linewidth=1.5, linestyle='--', label='Skenario Bagus (90%)')
                ax_mc.fill_between(x_axis, np.percentile(path_matrix, 10, axis=0), np.percentile(path_matrix, 90, axis=0), color='cyan', alpha=0.08)
            ax_mc.axhline(total_invested, color='white', linestyle=':', linewidth=1.2, label=f'Modal Disetor')
            ax_mc.set_xlabel('Bulan ke-', color='white'); ax_mc.set_ylabel('Nilai Portofolio (Rp)', color='white')
            ax_mc.set_title(f'Fan Chart DCA {dca_years} Tahun — 50 Sampel dari 1.000 Simulasi', color='white')
            ax_mc.set_facecolor('black'); fig_mc.patch.set_facecolor('black')
            ax_mc.tick_params(colors='white'); ax_mc.legend(facecolor='black', labelcolor='white', fontsize=9)
            ax_mc.grid(alpha=0.15, color='gray')
            st.pyplot(fig_mc)

    # =====================================================
    # CHARTS
    # =====================================================
    st.divider()
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Price & Trends (with Bollinger Bands)")
        chart_df = df.tail(365)
        fig, ax  = plt.subplots(figsize=(10, 5))
        ax.plot(chart_df["date"], chart_df["close"],    label="Close",    color="cyan")
        ax.plot(chart_df["date"], chart_df["ma50"],     label="MA50",     linestyle="--", color="yellow")
        ax.plot(chart_df["date"], chart_df["ma200"],    label="MA200",    linestyle="--", color="red")
        ax.plot(chart_df["date"], chart_df["bb_upper"], label="BB Upper", linestyle=":",  color="magenta", alpha=0.5)
        ax.plot(chart_df["date"], chart_df["bb_lower"], label="BB Lower", linestyle=":",  color="magenta", alpha=0.5)

        # Tampilkan POC line di chart
        if poc_price:
            ax.axhline(poc_price, color='#00e676', linestyle='-.', linewidth=1.2, alpha=0.8, label=f'POC Rp {poc_price:,.0f}')

        ax.set_facecolor('black'); fig.patch.set_facecolor('black')
        ax.tick_params(colors='white'); ax.yaxis.label.set_color('white'); ax.xaxis.label.set_color('white')
        ax.legend(facecolor='black', labelcolor='white', fontsize=8); ax.grid(alpha=0.2, color='gray')
        st.pyplot(fig)

        # Volume Profile Chart
        if vol_profile is not None and bin_centers is not None:
            st.subheader("📊 Volume Profile (252 Hari Terakhir)")
            fig_vp, ax_vp = plt.subplots(figsize=(10, 3))
            bar_colors = ['#00e676' if bc == bin_centers[np.argmax(vol_profile)] else '#2196F3' for bc in bin_centers]
            ax_vp.barh(bin_centers, vol_profile, height=(bin_centers[1]-bin_centers[0])*0.85,
                       color=bar_colors, alpha=0.85)
            ax_vp.axhline(float(latest["close"]), color='yellow', linestyle='--', linewidth=1.5, label=f'Harga Saat Ini')
            ax_vp.axhline(poc_price, color='#00e676', linestyle='-', linewidth=2, label=f'POC')
            ax_vp.set_facecolor('black'); fig_vp.patch.set_facecolor('black')
            ax_vp.tick_params(colors='white'); ax_vp.set_xlabel('Volume', color='white'); ax_vp.set_ylabel('Harga (Rp)', color='white')
            ax_vp.legend(facecolor='black', labelcolor='white', fontsize=9)
            ax_vp.set_title('Volume Profile — Distribusi Transaksi per Level Harga', color='white')
            ax_vp.grid(alpha=0.15, color='gray', axis='x')
            st.pyplot(fig_vp)

        st.subheader("RSI Momentum")
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.plot(chart_df["date"], chart_df["rsi"], color="lime")
        ax2.axhline(70, color="red",   linestyle="--", alpha=0.7, label="Overbought 70")
        ax2.axhline(30, color="green", linestyle="--", alpha=0.7, label="Oversold 30")
        ax2.axhline(20, color="#00e676", linestyle=":",  alpha=0.7, label="Extreme Oversold 20")
        ax2.set_facecolor('black'); fig2.patch.set_facecolor('black')
        ax2.tick_params(colors='white'); ax2.set_ylim(0, 100)
        ax2.legend(facecolor='black', labelcolor='white', fontsize=8)
        st.pyplot(fig2)

        # PBV Band Chart (jika data tersedia)
        if pbv_series is not None and pbv_mean is not None and pbv_std is not None:
            st.subheader(f"📐 PBV Band Chart ({pvb_window_years} Tahun Terakhir)")
            cutoff_chart = pd.Timestamp.now() - pd.DateOffset(years=pvb_window_years)
            pbv_chart = pbv_series[pbv_series.index >= cutoff_chart]
            if len(pbv_chart) > 10:
                fig3, ax3 = plt.subplots(figsize=(10, 3))
                ax3.plot(pbv_chart.index, pbv_chart.values, color='cyan', linewidth=1.2, label='PBV')
                ax3.axhline(pbv_mean,            color='white',   linestyle='-',  linewidth=1.2, label=f'Mean {pbv_mean:.2f}x')
                ax3.axhline(pbv_mean + pbv_std,  color='#f1c40f', linestyle='--', linewidth=1,   label=f'+1SD {pbv_mean+pbv_std:.2f}x')
                ax3.axhline(pbv_mean - pbv_std,  color='#2ecc71', linestyle='--', linewidth=1,   label=f'-1SD {pbv_mean-pbv_std:.2f}x')
                ax3.axhline(pbv_mean + 2*pbv_std,color='#e74c3c', linestyle=':',  linewidth=1,   label=f'+2SD {pbv_mean+2*pbv_std:.2f}x')
                ax3.axhline(pbv_mean - 2*pbv_std,color='#00e676', linestyle=':',  linewidth=1,   label=f'-2SD {pbv_mean-2*pbv_std:.2f}x')
                ax3.fill_between(pbv_chart.index, pbv_mean - pbv_std, pbv_mean + pbv_std, color='cyan', alpha=0.05)
                ax3.scatter([pbv_chart.index[-1]], [current_pbv], color='yellow', s=60, zorder=5, label=f'Sekarang {current_pbv:.2f}x')
                ax3.set_facecolor('black'); fig3.patch.set_facecolor('black')
                ax3.tick_params(colors='white'); ax3.set_ylabel('PBV', color='white')
                ax3.legend(facecolor='black', labelcolor='white', fontsize=8, ncol=3)
                ax3.grid(alpha=0.15, color='gray')
                ax3.set_title(f'PBV Band — {ticker_input} ({pvb_window_years} Tahun)', color='white')
                st.pyplot(fig3)

        # Dividend Yield Band Chart (jika data tersedia)
        if dy_series is not None and dy_mean is not None and dy_std is not None:
            st.subheader(f"💰 Dividend Yield Band Chart ({div_yield_window_years} Tahun Terakhir)")
            cutoff_dy = pd.Timestamp.now() - pd.DateOffset(years=div_yield_window_years)
            dy_chart  = dy_series[dy_series.index >= cutoff_dy] * 100  # ke persen
            if len(dy_chart) > 10:
                dy_m_pct  = dy_mean * 100
                dy_s_pct  = dy_std  * 100
                fig4, ax4 = plt.subplots(figsize=(10, 3))
                ax4.plot(dy_chart.index, dy_chart.values, color='#7c4dff', linewidth=1.2, label='Div Yield TTM')
                ax4.axhline(dy_m_pct,             color='white',   linestyle='-',  linewidth=1.2, label=f'Mean {dy_m_pct:.2f}%')
                ax4.axhline(dy_m_pct + dy_s_pct,  color='#2ecc71', linestyle='--', linewidth=1,   label=f'+1SD {dy_m_pct+dy_s_pct:.2f}%')
                ax4.axhline(dy_m_pct + 2*dy_s_pct,color='#00e676', linestyle=':',  linewidth=1,   label=f'+2SD {dy_m_pct+2*dy_s_pct:.2f}%')
                ax4.axhline(dy_m_pct - dy_s_pct,  color='#f1c40f', linestyle='--', linewidth=1,   label=f'-1SD {dy_m_pct-dy_s_pct:.2f}%')
                ax4.fill_between(dy_chart.index, dy_m_pct + dy_s_pct, dy_m_pct + 2*dy_s_pct, color='#2ecc71', alpha=0.07)
                ax4.scatter([dy_chart.index[-1]], [current_dy*100], color='yellow', s=60, zorder=5, label=f'Sekarang {current_dy*100:.2f}%')
                ax4.set_facecolor('black'); fig4.patch.set_facecolor('black')
                ax4.tick_params(colors='white'); ax4.set_ylabel('Div Yield (%)', color='white')
                ax4.legend(facecolor='black', labelcolor='white', fontsize=8, ncol=3)
                ax4.grid(alpha=0.15, color='gray')
                ax4.set_title(f'Dividend Yield Band — {ticker_input} ({div_yield_window_years} Tahun)', color='white')
                st.pyplot(fig4)

    with c2:
        st.subheader("📋 Ringkasan Analisa")
        for r in reasons:
            st.markdown(f"<div class='analyst-text'>📌 {r}</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**🔬 Sinyal Kuantitatif Baru**")
        if pbv_signal_text:
            st.markdown(f"<div class='analyst-text'>{pbv_signal_text}</div>", unsafe_allow_html=True)
        if dy_signal_text:
            st.markdown(f"<div class='analyst-text'>{dy_signal_text}</div>", unsafe_allow_html=True)
        if macd_div_text:
            st.markdown(f"<div class='analyst-text'>{macd_div_text}</div>", unsafe_allow_html=True)
        if poc_signal_text:
            st.markdown(f"<div class='analyst-text'>{poc_signal_text}</div>", unsafe_allow_html=True)

    if mode == "Trading":
        st.divider()
        st.success(f"🎯 Take Profit (Target Ideal): Rp {take_profit:,.0f}")
        st.error(f"🛑 Stop Loss (Batas Risiko): Rp {stop_loss:,.0f}")
        if poc_price:
            st.info(f"📌 POC Support/Resistance: Rp {poc_price:,.0f} — pertimbangkan sebagai level entry alternatif atau target profit.")
        st.caption("Catatan: Angka dikalkulasi secara mekanis. Sesuaikan dengan rasio Risk/Reward dan level support/resistance.")

except Exception as e:
    st.error("Terjadi kendala saat memproses data.")
    st.warning("Pastikan Anda menggunakan .JK untuk saham Indonesia (misal: TLKM.JK) dan koneksi internet stabil.")
    with st.expander("Lihat Detail Error (Untuk Debugging)"):
        import traceback
        st.code(traceback.format_exc())

st.divider()
st.caption("Disclaimer: Analisis ini digenerasi oleh sistem berbasis aturan teknikal dan fundamental dasar. Semua keputusan investasi sepenuhnya menjadi tanggung jawab investor (DYOR).")
st.markdown(
    "<div style='text-align:center; color:#8b8b8b; font-size:12px; margin-top:20px;'>"
    "Made with ❤️ by <b>Yoga Adi Tandanu</b>"
    "</div>",
    unsafe_allow_html=True
)
