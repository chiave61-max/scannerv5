import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

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
<link href='https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap' rel='stylesheet'>
""", unsafe_allow_html=True)

UNIVERSE = {
    'SPY':      {'name': 'S&P 500',    'cat': 'equity',    'pip': False},
    'QQQ':      {'name': 'Nasdaq 100', 'cat': 'equity',    'pip': False},
    'IWM':      {'name': 'Small Cap',  'cat': 'equity',    'pip': False},
    'XLK':      {'name': 'Tech',       'cat': 'equity',    'pip': False},
    'XLE':      {'name': 'Energy',     'cat': 'equity',    'pip': False},
    'XLF':      {'name': 'Finance',    'cat': 'equity',    'pip': False},
    'GLD':      {'name': 'Gold ETF',   'cat': 'commodity', 'pip': False},
    'GC=F':     {'name': 'Gold Fut',   'cat': 'commodity', 'pip': False},
    'BZ=F':     {'name': 'Brent',      'cat': 'commodity', 'pip': False},
    'USO':      {'name': 'WTI Oil',    'cat': 'commodity', 'pip': False},
    'EURUSD=X': {'name': 'EUR/USD',    'cat': 'forex',     'pip': True},
    'GBPUSD=X': {'name': 'GBP/USD',    'cat': 'forex',     'pip': True},
}

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

@st.cache_data(ttl=300)
def get_signal(ticker, pip):
    try:
        df = yf.download(ticker, period='60d', interval='1h',
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 30:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        price  = float(df['Close'].iloc[-1])
        prev_h = float(df['High'].iloc[-2])
        prev_l = float(df['Low'].iloc[-2])
        ma20   = df['Close'].rolling(20).mean().iloc[-1]
        ma50   = df['Close'].rolling(50).mean().iloc[-1]
        rsi    = calc_rsi(df['Close'])
        atr    = calc_atr(df)

        vol_ratio = 0
        if not pip and df['Volume'].iloc[-1] > 0:
            vol_avg   = df['Volume'].rolling(20).mean().iloc[-1]
            vol_ratio = float(df['Volume'].iloc[-1]) / float(vol_avg) if vol_avg > 0 else 0

        bull_break  = price > prev_h
        bear_break  = price < prev_l
        uptrend     = price > float(ma50)
        vol_ok      = pip or vol_ratio > 1.3

        score_buy = 0
        if bull_break:                 score_buy += 3
        if uptrend:                    score_buy += 2
        if vol_ok and vol_ratio > 1.5: score_buy += 2
        if rsi < 70:                   score_buy += 1
        if price > float(ma20):        score_buy += 1
        if rsi < 60:                   score_buy += 1

        score_sell = 0
        if bear_break:                  score_sell += 3
        if price < float(ma50):         score_sell += 2
        if vol_ok and vol_ratio > 1.5:  score_sell += 2
        if rsi > 30:                    score_sell += 1
        if price < float(ma20):         score_sell += 1
        if rsi > 40:                    score_sell += 1

        sl_long  = price - atr * 1.5
        tp_long  = price + atr * 3.0
        sl_short = price + atr * 1.5
        tp_short = price - atr * 3.0

        if score_buy >= 6 and bull_break:
            status = 'BUY'
        elif score_sell >= 6 and bear_break:
            status = 'SELL'
        else:
            status = 'WAIT'

        return {
            'ticker':     ticker,
            'price':      price,
            'rsi':        round(rsi, 1),
            'vol_ratio':  round(vol_ratio, 1),
            'uptrend':    uptrend,
            'score_buy':  score_buy,
            'score_sell': score_sell,
            'sl_long':    round(sl_long, 4),
            'tp_long':    round(tp_long, 4),
            'sl_short':   round(sl_short, 4),
            'tp_short':   round(tp_short, 4),
            'status':     status,
        }
    except Exception:
        return None

def score_color(s):
    if s >= 7: return 'score-high'
    if s >= 5: return 'score-mid'
    return 'score-low'

def render_card(r, info):
    css       = 'buy' if r['status'] == 'BUY' else 'sell' if r['status'] == 'SELL' else 'wait'
    sig_color = '#00ff88' if r['status'] == 'BUY' else '#ff3355' if r['status'] == 'SELL' else '#3a5070'
    sig_icon  = '&#9650; BUY' if r['status'] == 'BUY' else '&#9660; SELL' if r['status'] == 'SELL' else '&#9898; ATTESA'
    score     = r['score_buy'] if r['status'] == 'BUY' else r['score_sell'] if r['status'] == 'SELL' else max(r['score_buy'], r['score_sell'])
    sc_class  = score_color(score)
    trend_ico = '&#128200;' if r['uptrend'] else '&#128201;'
    vol_str   = f"{r['vol_ratio']:.1f}x" if r['vol_ratio'] > 0 else 'N/A'

    levels = ''
    if r['status'] == 'BUY':
        rr = (r['tp_long'] - r['price']) / (r['price'] - r['sl_long']) if r['price'] > r['sl_long'] else 0
        levels = (
            "<div style='margin-top:10px;padding:8px;background:#0a0f15;border-radius:4px;font-size:0.78em'>"
            f"<span style='color:#00ff88'>TP: {r['tp_long']:.4f}</span> &nbsp;|&nbsp;"
            f"<span style='color:#ff3355'>SL: {r['sl_long']:.4f}</span> &nbsp;|&nbsp;"
            f"<span style='color:#ffcc00'>R/R: {rr:.1f}:1</span>"
            "</div>"
        )
    elif r['status'] == 'SELL':
        rr = (r['price'] - r['tp_short']) / (r['sl_short'] - r['price']) if r['sl_short'] > r['price'] else 0
        levels = (
            "<div style='margin-top:10px;padding:8px;background:#0a0f15;border-radius:4px;font-size:0.78em'>"
            f"<span style='color:#00ff88'>TP: {r['tp_short']:.4f}</span> &nbsp;|&nbsp;"
            f"<span style='color:#ff3355'>SL: {r['sl_short']:.4f}</span> &nbsp;|&nbsp;"
            f"<span style='color:#ffcc00'>R/R: {rr:.1f}:1</span>"
            "</div>"
        )

    name = info['name']
    cat  = info['cat'].upper()
    tick = r['ticker']

    st.markdown(
        f"<div class='{css} card'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center'>"
        f"<div><b style='color:white;font-size:1.1em'>{name}</b>"
        f"<span style='color:#3a5070;font-size:0.75em;margin-left:8px'>{tick} &middot; {cat}</span></div>"
        f"<div style='text-align:right'><b style='color:{sig_color};font-size:1em'>{sig_icon}</b><br>"
        f"<span class='{sc_class}'>{score}/10</span></div>"
        f"</div>"
        f"<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:10px'>"
        f"<div><div class='label'>Prezzo</div><div class='value'>{r['price']:.4f}</div></div>"
        f"<div><div class='label'>RSI</div><div class='value'>{r['rsi']}</div></div>"
        f"<div><div class='label'>Volume</div><div class='value'>{vol_str}</div></div>"
        f"<div><div class='label'>Trend</div><div class='value'>{trend_ico}</div></div>"
        f"</div>"
        f"{levels}"
        f"</div>",
        unsafe_allow_html=True
    )

# ── HEADER ──
st.markdown(
    "<h2 style='text-align:center;color:#00e5ff;letter-spacing:4px;margin-bottom:0'>&#128142; SCANNER ULTRA V5</h2>"
    "<p style='text-align:center;color:#3a5070;font-size:0.75em;letter-spacing:3px'>BREAKOUT + VOLUME + TREND + ATR</p>",
    unsafe_allow_html=True
)

now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
st.markdown(f"<p style='text-align:center;color:#ff8800;font-size:0.8em'>&#128260; {now}</p>",
            unsafe_allow_html=True)

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("### &#9881;&#65039; Filtri")
    cat_filter = st.multiselect(
        "Categorie", ['equity', 'commodity', 'forex'],
        default=['equity', 'commodity', 'forex']
    )
    min_score = st.slider("Score minimo segnale", 0, 10, 6)
    show_wait = st.checkbox("Mostra ATTESA", value=False)
    st.markdown("---")
    st.markdown("### &#128205; Legenda Score")
    st.markdown("&#128994; 7-10 = segnale forte")
    st.markdown("&#128993; 5-6  = segnale medio")
    st.markdown("&#128308; 0-4  = segnale debole")
    st.markdown("---")
    st.markdown("### &#9200;&#65039; Sistemi Attivi")
    st.markdown("&#128738;&#65039; EIA Petrolio: **mer 16:30**")
    st.markdown("&#128293; EIA Gas: **gio 16:30**")
    st.markdown("&#129351; Gold IB: **ogni sera**")
    st.markdown("&#127749;&#65039; London Open: **10:00**")

# ── SCAN ──
with st.spinner('Scansione mercati in corso...'):
    results = []
    for ticker, info in UNIVERSE.items():
        if info['cat'] not in cat_filter:
            continue
        r = get_signal(ticker, info['pip'])
        if r:
            r['ticker'] = ticker
            results.append((r, info))

buys  = [(r, i) for r, i in results if r['status'] == 'BUY'  and max(r['score_buy'], r['score_sell']) >= min_score]
sells = [(r, i) for r, i in results if r['status'] == 'SELL' and max(r['score_buy'], r['score_sell']) >= min_score]
waits = [(r, i) for r, i in results if r['status'] == 'WAIT']

buys.sort(key=lambda x: x[0]['score_buy'],  reverse=True)
sells.sort(key=lambda x: x[0]['score_sell'], reverse=True)

# ── METRICHE ──
col1, col2, col3 = st.columns(3)
col1.metric("BUY",    len(buys))
col2.metric("SELL",   len(sells))
col3.metric("ATTESA", len(waits))

st.markdown("---")

# ── RISULTATI ──
if len(buys) + len(sells) == 0:
    st.markdown(
        "<div class='wait card' style='text-align:center;padding:20px'>"
        "<b style='color:#3a5070;letter-spacing:3px'>NESSUN SEGNALE ATTIVO</b><br>"
        "<span style='color:#1a2535;font-size:0.8em'>Mercato in consolidamento</span>"
        "</div>",
        unsafe_allow_html=True
    )
else:
    if buys:
        st.markdown("<p style='color:#00ff88;letter-spacing:3px;font-size:0.8em'>&#9650; SEGNALI BUY</p>",
                    unsafe_allow_html=True)
        for r, info in buys:
            render_card(r, info)
    if sells:
        st.markdown("<p style='color:#ff3355;letter-spacing:3px;font-size:0.8em'>&#9660; SEGNALI SELL</p>",
                    unsafe_allow_html=True)
        for r, info in sells:
            render_card(r, info)

if show_wait and waits:
    st.markdown("<p style='color:#3a5070;letter-spacing:3px;font-size:0.8em'>&#9898; IN ATTESA</p>",
                unsafe_allow_html=True)
    for r, info in waits:
        render_card(r, info)

st.markdown("---")

# ── REFRESH MANUALE ──
if st.button("&#128260; Aggiorna segnali"):
    st.cache_data.clear()
    st.rerun()

st.markdown(
    "<p style='text-align:center;color:#1a2535;font-size:0.65em;letter-spacing:2px'>"
    "SOLO USO INFORMATIVO — NON CONSULENZA FINANZIARIA</p>",
    unsafe_allow_html=True
        )


