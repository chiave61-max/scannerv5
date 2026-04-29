import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time

# ═══════════════════════════════════════════════════════
# ⚙️ CONFIGURAZIONE
# ═══════════════════════════════════════════════════════
st.set_page_config(page_title="Scanner Ultra V5", layout="centered")

st.markdown("""
<style>
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
</style>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# 📋 UNIVERSO — SEPARATO PER CATEGORIA
# ═══════════════════════════════════════════════════════
UNIVERSE = {
    # Equity ETF (volume affidabile)
    'SPY':  {'name': 'S&P 500',    'cat': 'equity', 'pip': False},
    'QQQ':  {'name': 'Nasdaq 100', 'cat': 'equity', 'pip': False},
    'IWM':  {'name': 'Small Cap',  'cat': 'equity', 'pip': False},
    'XLK':  {'name': 'Tech',       'cat': 'equity', 'pip': False},
    'XLE':  {'name': 'Energy',     'cat': 'equity', 'pip': False},
    'XLF':  {'name': 'Finance',    'cat': 'equity', 'pip': False},
    # Commodity (volume parziale)
    'GLD':  {'name': 'Gold ETF',   'cat': 'commodity', 'pip': False},
    'GC=F': {'name': 'Gold Fut',   'cat': 'commodity', 'pip': False},
    'BZ=F': {'name': 'Brent',      'cat': 'commodity', 'pip': False},
    'USO':  {'name': 'WTI Oil',    'cat': 'commodity', 'pip': False},
    # Forex (volume NON affidabile — escluso dai filtri volume)
    'EURUSD=X': {'name': 'EUR/USD', 'cat': 'forex', 'pip': True},
    'GBPUSD=X': {'name': 'GBP/USD', 'cat': 'forex', 'pip': True},
}

# ═══════════════════════════════════════════════════════
# 🔧 FUNZIONI
# ═══════════════════════════════════════════════════════

def calc_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).iloc[-1]

def calc_atr(df, period=14):
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift()).abs(),
        (df['Low']  - df['Close'].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean().iloc[-1]

def get_signal(ticker, info):
    try:
        df = yf.download(ticker, period='60d', interval='1h',
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 30: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        price   = float(df['Close'].iloc[-1])
        prev_h  = float(df['High'].iloc[-2])
        prev_l  = float(df['Low'].iloc[-2])
        ma20    = df['Close'].rolling(20).mean().iloc[-1]
        ma50    = df['Close'].rolling(50).mean().iloc[-1]
        rsi     = calc_rsi(df['Close'])
        atr     = calc_atr(df)

        # Volume (solo per non-forex)
        vol_ratio = 0
        if not info['pip'] and df['Volume'].iloc[-1] > 0:
            vol_avg   = df['Volume'].rolling(20).mean().iloc[-1]
            vol_ratio = float(df['Volume'].iloc[-1]) / float(vol_avg) if vol_avg > 0 else 0

        # ── SEGNALI ──
        # Breakout rialzista
        bull_break = float(df['Close'].iloc[-1]) > prev_h
        # Breakout ribassista
        bear_break = float(df['Close'].iloc[-1]) < prev_l
        # Trend rialzista (prezzo sopra MA50)
        uptrend    = price > float(ma50)
        # Trend ribassista
        downtrend  = price < float(ma50)
        # Volume confermato (solo equity/commodity)
        vol_ok     = info['pip'] or vol_ratio > 1.3
        # RSI non estremo
        rsi_buy_ok  = rsi < 70
        rsi_sell_ok = rsi > 30

        # ── PUNTEGGIO 0-10 ──
        score_buy = 0
        if bull_break:              score_buy += 3
        if uptrend:                 score_buy += 2
        if vol_ok and vol_ratio > 1.5: score_buy += 2
        if rsi_buy_ok:              score_buy += 1
        if price > float(ma20):    score_buy += 1
        if rsi < 60:               score_buy += 1

        score_sell = 0
        if bear_break:              score_sell += 3
        if downtrend:               score_sell += 2
        if vol_ok and vol_ratio > 1.5: score_sell += 2
        if rsi_sell_ok:             score_sell += 1
        if price < float(ma20):    score_sell += 1
        if rsi > 40:               score_sell += 1

        # ── SL/TP basati su ATR ──
        sl_long  = price - atr * 1.5
        tp_long  = price + atr * 3.0
        sl_short = price + atr * 1.5
        tp_short = price - atr * 3.0

        # ── DECISIONE ──
        if score_buy >= 6 and bull_break:
            status = 'BUY'
        elif score_sell >= 6 and bear_break:
            status = 'SELL'
        else:
            status = 'WAIT'

        return {
            'ticker':    ticker,
            'name':      info['name'],
            'cat':       info['cat'],
            'price':     price,
            'rsi':       round(rsi, 1),
            'vol_ratio': round(vol_ratio, 1),
            'ma50':      round(float(ma50), 4),
            'atr':       round(atr, 4),
            'uptrend':   uptrend,
            'score_buy':  score_buy,
            'score_sell': score_sell,
            'sl_long':   round(sl_long, 4),
            'tp_long':   round(tp_long, 4),
            'sl_short':  round(sl_short, 4),
            'tp_short':  round(tp_short, 4),
            'status':    status,
        }
    except Exception as e:
        return None

def score_color(s):
    if s >= 7: return 'score-high'
    if s >= 5: return 'score-mid'
    return 'score-low'

def render_card(r):
    css = 'buy' if r['status']=='BUY' else 'sell' if r['status']=='SELL' else 'wait'
    sig_color = '#00ff88' if r['status']=='BUY' else '#ff3355' if r['status']=='SELL' else '#3a5070'
    sig_icon  = '▲ BUY' if r['status']=='BUY' else '▼ SELL' if r['status']=='SELL' else '⚪ ATTESA'
    score     = r['score_buy'] if r['status']=='BUY' else r['score_sell'] if r['status']=='SELL' else max(r['score_buy'], r['score_sell'])
    sc_class  = score_color(score)
    trend_ico = '📈' if r['uptrend'] else '📉'
    vol_str   = f"{r['vol_ratio']:.1f}x" if r['vol_ratio'] > 0 else 'N/A'

    levels = ''
    if r['status'] == 'BUY':
        rr = (r['tp_long'] - r['price']) / (r['price'] - r['sl_long']) if r['price'] > r['sl_long'] else 0
        levels = f"""
        <div style="margin-top:10px;padding:8px;background:#0a0f15;border-radius:4px;font-size:0.78em">
            <span style="color:#00ff88">TP: {r['tp_long']:.4f}</span> &nbsp;|&nbsp;
            <span style="color:#ff3355">SL: {r['sl_long']:.4f}</span> &nbsp;|&nbsp;
            <span style="color:#ffcc00">R/R: {rr:.1f}:1</span>
        </div>"""
    elif r['status'] == 'SELL':
        rr = (r['price'] - r['tp_short']) / (r['sl_short'] - r['price']) if r['sl_short'] > r['price'] else 0
        levels = f"""
        <div style="margin-top:10px;padding:8px;background:#0a0f15;border-radius:4px;font-size:0.78em">
            <span style="color:#00ff88">TP: {r['tp_short']:.4f}</span> &nbsp;|&nbsp;
            <span style="color:#ff3355">SL: {r['sl_short']:.4f}</span> &nbsp;|&nbsp;
            <span style="color:#ffcc00">R/R: {rr:.1f}:1</span>
        </div>"""

    st.markdown(f"""
    <div class="card {css}">
        <div style="display:flex;justify-content:space-between;align-items:center">
            <div>
                <b style="color:white;font-size:1.1em">{r['name']}</b>
                <span style="color:#3a5070;font-size:0.75em;margin-left:8px">{r['ticker']} · {r['cat'].upper()}</span>
            </div>
            <div style="text-align:right">
                <b style="color:{sig_color};font-size:1em">{sig_icon}</b><br>
                <span class="{sc_class}">{score}/10</span>
            </div>
        </div>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:10px">
            <div><div class="label">Prezzo</div><div class="value">{r['price']:.4f}</div></div>
            <div><div class="label">RSI</div><div class="value">{r['rsi']}</div></div>
            <div><div class="label">Volume</div><div class="value">{vol_str}</div></div>
            <div><div class="label">Trend</div><div class="value">{trend_ico}</div></div>
        </div>
        {levels}
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# 🖥️ INTERFACCIA
# ═══════════════════════════════════════════════════════

st.markdown("""
<h2 style='text-align:center;color:#00e5ff;letter-spacing:4px;margin-bottom:0'>
💎 SCANNER ULTRA V5
</h2>
<p style='text-align:center;color:#3a5070;font-size:0.75em;letter-spacing:3px'>
BREAKOUT + VOLUME + TREND + ATR
</p>
""", unsafe_allow_html=True)

now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
st.markdown(f"<p style='text-align:center;color:#ff8800;font-size:0.8em'>🔄 {now}</p>",
            unsafe_allow_html=True)

# Filtri sidebar
with st.sidebar:
    st.markdown("### ⚙️ Filtri")
    cat_filter = st.multiselect(
        "Categorie", ['equity','commodity','forex'],
        default=['equity','commodity','forex']
    )
    min_score = st.slider("Score minimo segnale", 0, 10, 6)
    show_wait = st.checkbox("Mostra ATTESA", value=False)
    st.markdown("---")
    st.markdown("### 📌 Legenda Score")
    st.markdown("🟢 7-10 = segnale forte")
    st.markdown("🟡 5-6  = segnale medio")
    st.markdown("🔴 0-4  = segnale debole")
    st.markdown("---")
    st.markdown("### ⏰ Sistemi Attivi")
    st.markdown("🛢️ EIA Petrolio: **mer 16:30**")
    st.markdown("🔥 EIA Gas: **gio 16:30**")
    st.markdown("🥇 Gold IB: **ogni sera**")
    st.markdown("🌅 London Open: **10:00**")

# Carica dati
with st.spinner('⏳ Scansione mercati in corso...'):
    results = []
    for ticker, info in UNIVERSE.items():
        if info['cat'] not in cat_filter: continue
        r = get_signal(ticker, info)
        if r: results.append(r)

# Separa segnali
buys  = [r for r in results if r['status']=='BUY'  and max(r['score_buy'],r['score_sell']) >= min_score]
sells = [r for r in results if r['status']=='SELL' and max(r['score_buy'],r['score_sell']) >= min_score]
waits = [r for r in results if r['status']=='WAIT']

# Ordina per score
buys.sort(key=lambda x: x['score_buy'], reverse=True)
sells.sort(key=lambda x: x['score_sell'], reverse=True)

# Mostra risultati
total_signals = len(buys) + len(sells)
col1, col2, col3 = st.columns(3)
col1.metric("🟢 BUY",  len(buys))
col2.metric("🔴 SELL", len(sells))
col3.metric("⚪ ATTESA", len(waits))

st.markdown("---")

if total_signals == 0:
    st.markdown("""
    <div class="card wait" style="text-align:center;padding:20px">
        <b style="color:#3a5070;letter-spacing:3px">NESSUN SEGNALE ATTIVO</b><br>
        <span style="color:#1a2535;font-size:0.8em">Mercato in consolidamento</span>
    </div>
    """, unsafe_allow_html=True)
else:
    if buys:
        st.markdown("<p style='color:#00ff88;letter-spacing:3px;font-size:0.8em'>▲ SEGNALI BUY</p>",
                    unsafe_allow_html=True)
        for r in buys: render_card(r)

    if sells:
        st.markdown("<p style='color:#ff3355;letter-spacing:3px;font-size:0.8em'>▼ SEGNALI SELL</p>",
                    unsafe_allow_html=True)
        for r in sells: render_card(r)

if show_wait and waits:
    st.markdown("<p style='color:#3a5070;letter-spacing:3px;font-size:0.8em'>⚪ IN ATTESA</p>",
                unsafe_allow_html=True)
    for r in waits: render_card(r)

st.markdown("---")
st.markdown("""
<p style='text-align:center;color:#1a2535;font-size:0.65em;letter-spacing:2px'>
SOLO USO INFORMATIVO — NON CONSULENZA FINANZIARIA
</p>
""", unsafe_allow_html=True)

# Auto-refresh ogni 60 secondi
time.sleep(60)
st.rerun()


