import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time

# ═══════════════════════════════════════════════════════
# ⚙️ CONFIGURAZIONE
# ═══════════════════════════════════════════════════════
st.set_page_config(page_title="Scanner Ultra V6", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
.stApp { background-color: #050a10; }
.card {
    padding: 14px; border-radius: 8px;
    margin-bottom: 10px; border: 1px solid #1a2535;
    font-family: 'Share Tech Mono', monospace;
}
.buy  { background-color: #0a1f14; border-left: 4px solid #00ff88; }
.sell { background-color: #1f0a0a; border-left: 4px solid #ff3355; }
.wait { background-color: #0d1520; border-left: 4px solid #1a2535; }
.label { color: #3a5070; font-size: 0.72em; text-transform: uppercase; letter-spacing: 2px; }
.value { color: white; font-weight: bold; font-size: 1em; }
.score-high { color: #00ff88; font-size: 1.3em; font-weight: bold; }
.score-mid  { color: #ffcc00; font-size: 1.3em; font-weight: bold; }
.score-low  { color: #ff3355; font-size: 1.3em; font-weight: bold; }
.warn { color: #ff8800; font-size: 0.75em; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# 📋 UNIVERSO
# ═══════════════════════════════════════════════════════
UNIVERSE = {
    'SPY':      {'name': 'S&P 500',    'cat': 'equity',    'forex': False},
    'QQQ':      {'name': 'Nasdaq 100', 'cat': 'equity',    'forex': False},
    'IWM':      {'name': 'Small Cap',  'cat': 'equity',    'forex': False},
    'XLK':      {'name': 'Tech',       'cat': 'equity',    'forex': False},
    'XLE':      {'name': 'Energy',     'cat': 'equity',    'forex': False},
    'XLF':      {'name': 'Finance',    'cat': 'equity',    'forex': False},
    'XLV':      {'name': 'Healthcare', 'cat': 'equity',    'forex': False},
    'GLD':      {'name': 'Gold ETF',   'cat': 'commodity', 'forex': False},
    'GC=F':     {'name': 'Gold Fut',   'cat': 'commodity', 'forex': False},
    'BZ=F':     {'name': 'Brent',      'cat': 'commodity', 'forex': False},
    'USO':      {'name': 'WTI Oil',    'cat': 'commodity', 'forex': False},
    'UNG':      {'name': 'Nat Gas',    'cat': 'commodity', 'forex': False},
    'EURUSD=X': {'name': 'EUR/USD',    'cat': 'forex',     'forex': True},
    'GBPUSD=X': {'name': 'GBP/USD',    'cat': 'forex',     'forex': True},
}

# ═══════════════════════════════════════════════════════
# 🔧 INDICATORI
# ═══════════════════════════════════════════════════════

def calc_rsi(series, period=14):
    """RSI corretto su dati giornalieri."""
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).iloc[-1]

def calc_atr(df, period=14):
    """ATR per calcolo SL/TP dinamici."""
    hl  = df['High'] - df['Low']
    hpc = (df['High'] - df['Close'].shift()).abs()
    lpc = (df['Low']  - df['Close'].shift()).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    return tr.rolling(period).mean().iloc[-1]

def calc_adx(df, period=14):
    """ADX per misurare forza del trend."""
    try:
        high, low, close = df['High'], df['Low'], df['Close']
        dm_plus  = (high.diff()).clip(lower=0)
        dm_minus = (-low.diff()).clip(lower=0)
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr14    = tr.rolling(period).mean()
        di_plus  = 100 * dm_plus.rolling(period).mean()  / atr14
        di_minus = 100 * dm_minus.rolling(period).mean() / atr14
        dx       = (100 * (di_plus - di_minus).abs() / (di_plus + di_minus)).rolling(period).mean()
        return dx.iloc[-1]
    except:
        return 0

def get_signal(ticker, info):
    try:
        # Dati giornalieri — RSI più affidabile
        df = yf.download(ticker, period='120d', interval='1d',
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 50: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        price   = float(df['Close'].iloc[-1])
        prev_h  = float(df['High'].iloc[-2])
        prev_l  = float(df['Low'].iloc[-2])

        # Medie mobili
        ma20  = float(df['Close'].rolling(20).mean().iloc[-1])
        ma50  = float(df['Close'].rolling(50).mean().iloc[-1])
        ma200 = float(df['Close'].rolling(100).mean().iloc[-1])  # 100 come proxy 200

        # Indicatori
        rsi  = calc_rsi(df['Close'])
        atr  = calc_atr(df)
        adx  = calc_adx(df)

        # Volume (solo equity/commodity)
        vol_ratio = 0.0
        if not info['forex']:
            vol     = float(df['Volume'].iloc[-1])
            vol_avg = float(df['Volume'].rolling(20).mean().iloc[-1])
            vol_ratio = vol / vol_avg if vol_avg > 0 else 0

        # ── FILTRI QUALITÀ ──
        # RSI non estremo (zona operativa)
        rsi_buy_ok  = 40 < rsi < 65   # non ipercomprato e non collassato
        rsi_sell_ok = 35 < rsi < 60   # non ipervenduto e non in rally
        # Trend
        uptrend   = price > ma50 and ma50 > ma200
        downtrend = price < ma50 and ma50 < ma200
        # Volume confermato
        vol_ok = info['forex'] or vol_ratio > 1.2
        # Trend forte (ADX > 20)
        strong_trend = adx > 20
        # Breakout
        bull_break = price > prev_h
        bear_break = price < prev_l

        # ── PUNTEGGIO BUY (max 10) ──
        score_buy = 0
        if bull_break:      score_buy += 3  # breakout confermato
        if uptrend:         score_buy += 2  # trend allineato
        if vol_ratio > 1.5: score_buy += 2  # volume forte
        elif vol_ok:        score_buy += 1  # volume sufficiente
        if rsi_buy_ok:      score_buy += 1  # RSI in zona sana
        if strong_trend:    score_buy += 1  # ADX trend forte
        if price > ma20:    score_buy += 1  # sopra MA breve

        # Penalità BUY
        if rsi > 70:        score_buy -= 3  # ipercomprato → NON comprare
        if rsi > 65:        score_buy -= 1  # quasi ipercomprato
        if vol_ratio < 0.8: score_buy -= 1  # volume molto basso
        if not uptrend:     score_buy -= 1  # trend non allineato

        # ── PUNTEGGIO SELL (max 10) ──
        score_sell = 0
        if bear_break:      score_sell += 3
        if downtrend:       score_sell += 2
        if vol_ratio > 1.5: score_sell += 2
        elif vol_ok:        score_sell += 1
        if rsi_sell_ok:     score_sell += 1
        if strong_trend:    score_sell += 1
        if price < ma20:    score_sell += 1

        # Penalità SELL
        if rsi < 30:        score_sell -= 3  # ipervenduto → NON vendere
        if rsi < 35:        score_sell -= 1
        if vol_ratio < 0.8: score_sell -= 1
        if not downtrend:   score_sell -= 1

        # Clamp 0-10
        score_buy  = max(0, min(10, score_buy))
        score_sell = max(0, min(10, score_sell))

        # ── SL/TP basati su ATR (R/R = 2:1) ──
        sl_long  = price - atr * 1.5
        tp_long  = price + atr * 3.0
        sl_short = price + atr * 1.5
        tp_short = price - atr * 3.0

        # ── DECISIONE FINALE ──
        # Richiede score >= 7 E filtri RSI/volume OK
        if score_buy >= 7 and bull_break and rsi_buy_ok and vol_ok:
            status = 'BUY'
        elif score_sell >= 7 and bear_break and rsi_sell_ok and vol_ok:
            status = 'SELL'
        else:
            status = 'WAIT'

        # Warning speciali
        warnings = []
        if rsi > 70: warnings.append('⚠️ RSI IPERCOMPRATO')
        if rsi < 30: warnings.append('⚠️ RSI IPERVENDUTO')
        if vol_ratio > 0 and vol_ratio < 0.8: warnings.append('⚠️ VOLUME BASSO')
        if adx < 15: warnings.append('⚠️ TREND DEBOLE')

        return {
            'ticker':     ticker,
            'name':       info['name'],
            'cat':        info['cat'],
            'forex':      info['forex'],
            'price':      price,
            'rsi':        round(rsi, 1),
            'adx':        round(adx, 1),
            'vol_ratio':  round(vol_ratio, 1),
            'ma20':       round(ma20, 4),
            'ma50':       round(ma50, 4),
            'atr':        round(atr, 4),
            'uptrend':    uptrend,
            'downtrend':  downtrend,
            'score_buy':  score_buy,
            'score_sell': score_sell,
            'sl_long':    round(sl_long, 4),
            'tp_long':    round(tp_long, 4),
            'sl_short':   round(sl_short, 4),
            'tp_short':   round(tp_short, 4),
            'status':     status,
            'warnings':   warnings,
        }
    except Exception as e:
        return None

# ═══════════════════════════════════════════════════════
# 🖥️ RENDER CARD
# ═══════════════════════════════════════════════════════

def score_color(s):
    if s >= 8: return 'score-high'
    if s >= 6: return 'score-mid'
    return 'score-low'

def render_card(r):
    css       = 'buy' if r['status']=='BUY' else 'sell' if r['status']=='SELL' else 'wait'
    sig_color = '#00ff88' if r['status']=='BUY' else '#ff3355' if r['status']=='SELL' else '#3a5070'
    sig_icon  = '▲ BUY'  if r['status']=='BUY' else '▼ SELL'  if r['status']=='SELL' else '⚪ WAIT'
    score     = r['score_buy'] if r['status']=='BUY' else r['score_sell'] if r['status']=='SELL' else max(r['score_buy'], r['score_sell'])
    sc_class  = score_color(score)
    trend_ico = '📈' if r['uptrend'] else '📉' if r['downtrend'] else '➡️'
    vol_str   = f"{r['vol_ratio']:.1f}x" if r['vol_ratio'] > 0 else 'N/A'

    # Livelli
    levels_html = ''
    if r['status'] == 'BUY':
        rr = round((r['tp_long'] - r['price']) / (r['price'] - r['sl_long']), 1) if r['price'] > r['sl_long'] else 0
        levels_html = (
            '<div style="margin-top:10px;padding:8px;background:#0a0f15;border-radius:4px;font-size:0.78em">'
            f'<span style="color:#00ff88">TP: {r["tp_long"]:.4f}</span>'
            ' &nbsp;|&nbsp; '
            f'<span style="color:#ff3355">SL: {r["sl_long"]:.4f}</span>'
            ' &nbsp;|&nbsp; '
            f'<span style="color:#ffcc00">R/R: {rr}:1</span>'
            '</div>'
        )
    elif r['status'] == 'SELL':
        rr = round((r['price'] - r['tp_short']) / (r['sl_short'] - r['price']), 1) if r['sl_short'] > r['price'] else 0
        levels_html = (
            '<div style="margin-top:10px;padding:8px;background:#0a0f15;border-radius:4px;font-size:0.78em">'
            f'<span style="color:#00ff88">TP: {r["tp_short"]:.4f}</span>'
            ' &nbsp;|&nbsp; '
            f'<span style="color:#ff3355">SL: {r["sl_short"]:.4f}</span>'
            ' &nbsp;|&nbsp; '
            f'<span style="color:#ffcc00">R/R: {rr}:1</span>'
            '</div>'
        )

    # Warning
    warn_html = ''
    if r['warnings']:
        warn_html = '<div style="margin-top:6px">' + ' '.join(
            [f'<span class="warn">{w}</span>' for w in r['warnings']]
        ) + '</div>'

    st.markdown(
        f'<div class="card {css}">'
        f'<div style="display:flex;justify-content:space-between;align-items:center">'
        f'<div>'
        f'<b style="color:white;font-size:1.1em">{r["name"]}</b>'
        f'<span style="color:#3a5070;font-size:0.75em;margin-left:8px">{r["ticker"]} · {r["cat"].upper()}</span>'
        f'</div>'
        f'<div style="text-align:right">'
        f'<b style="color:{sig_color};font-size:1em">{sig_icon}</b><br>'
        f'<span class="{sc_class}">{score}/10</span>'
        f'</div>'
        f'</div>'
        f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:10px">'
        f'<div><div class="label">Prezzo</div><div class="value">{r["price"]:.4f}</div></div>'
        f'<div><div class="label">RSI</div><div class="value">{r["rsi"]}</div></div>'
        f'<div><div class="label">ADX</div><div class="value">{r["adx"]}</div></div>'
        f'<div><div class="label">Trend</div><div class="value">{trend_ico} {vol_str}</div></div>'
        f'</div>'
        f'{levels_html}'
        f'{warn_html}'
        f'</div>',
        unsafe_allow_html=True
    )

# ═══════════════════════════════════════════════════════
# 🖥️ INTERFACCIA
# ═══════════════════════════════════════════════════════

st.markdown(
    '<h2 style="text-align:center;color:#00e5ff;letter-spacing:4px;margin-bottom:0">💎 SCANNER ULTRA V6</h2>'
    '<p style="text-align:center;color:#3a5070;font-size:0.75em;letter-spacing:3px">BREAKOUT · VOLUME · TREND · RSI · ADX · ATR</p>',
    unsafe_allow_html=True
)

now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
st.markdown(f'<p style="text-align:center;color:#ff8800;font-size:0.8em">🔄 {now}</p>',
            unsafe_allow_html=True)

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("### ⚙️ Filtri")
    cat_filter = st.multiselect(
        "Categorie", ['equity', 'commodity', 'forex'],
        default=['equity', 'commodity', 'forex']
    )
    min_score = st.slider("Score minimo", 0, 10, 7)
    show_wait = st.checkbox("Mostra ATTESA", value=False)

    st.markdown("---")
    st.markdown("### 📊 Filtri attivi V6")
    st.markdown("✅ RSI comprato: **40-65**")
    st.markdown("✅ RSI venduto: **35-60**")
    st.markdown("✅ Penalità RSI > 70")
    st.markdown("✅ Trend: MA50 > MA200")
    st.markdown("✅ ADX > 20 (trend forte)")
    st.markdown("✅ Volume > 1.2x media")
    st.markdown("✅ Score minimo: **7/10**")
    st.markdown("✅ Dati: **giornalieri** (più affidabili)")

    st.markdown("---")
    st.markdown("### ⏰ Sistemi attivi")
    st.markdown("🛢️ EIA Petrolio: **mer 16:30**")
    st.markdown("🔥 EIA Gas: **gio 16:30**")
    st.markdown("🥇 Gold IB: **ogni sera**")
    st.markdown("🌅 London Open: **10:00**")

# ── SCAN ──
with st.spinner('⏳ Scansione mercati...'):
    results = []
    for ticker, info in UNIVERSE.items():
        if info['cat'] not in cat_filter: continue
        r = get_signal(ticker, info)
        if r: results.append(r)

# ── SEPARA E ORDINA ──
buys  = sorted([r for r in results if r['status']=='BUY'  and r['score_buy']  >= min_score], key=lambda x: x['score_buy'],  reverse=True)
sells = sorted([r for r in results if r['status']=='SELL' and r['score_sell'] >= min_score], key=lambda x: x['score_sell'], reverse=True)
waits = [r for r in results if r['status']=='WAIT']

# ── METRICHE ──
c1, c2, c3 = st.columns(3)
c1.metric("🟢 BUY",    len(buys))
c2.metric("🔴 SELL",   len(sells))
c3.metric("⚪ ATTESA", len(waits))
st.markdown("---")

# ── RISULTATI ──
total = len(buys) + len(sells)

if total == 0:
    st.markdown(
        '<div class="card wait" style="text-align:center;padding:24px">'
        '<b style="color:#3a5070;letter-spacing:3px;font-size:0.9em">NESSUN SEGNALE DI QUALITÀ</b><br>'
        '<span style="color:#1a2535;font-size:0.75em">Tutti i filtri sono attivi — attendere setup migliori</span>'
        '</div>',
        unsafe_allow_html=True
    )
else:
    if buys:
        st.markdown('<p style="color:#00ff88;letter-spacing:3px;font-size:0.8em">▲ SEGNALI BUY</p>',
                    unsafe_allow_html=True)
        for r in buys: render_card(r)

    if sells:
        st.markdown('<p style="color:#ff3355;letter-spacing:3px;font-size:0.8em">▼ SEGNALI SELL</p>',
                    unsafe_allow_html=True)
        for r in sells: render_card(r)

if show_wait and waits:
    st.markdown('<p style="color:#3a5070;letter-spacing:3px;font-size:0.8em">⚪ IN ATTESA</p>',
                unsafe_allow_html=True)
    waits_sorted = sorted(waits, key=lambda x: max(x['score_buy'], x['score_sell']), reverse=True)
    for r in waits_sorted: render_card(r)

st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#1a2535;font-size:0.65em;letter-spacing:2px">'
    'SOLO USO INFORMATIVO — NON CONSULENZA FINANZIARIA'
    '</p>',
    unsafe_allow_html=True
)

# Auto-refresh 60 secondi
time.sleep(60)
st.rerun()


