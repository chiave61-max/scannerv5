import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

# ═══════════════════════════════════════════════════════
# ⚙️ CONFIGURAZIONE
# ═══════════════════════════════════════════════════════
st.set_page_config(page_title="News Trading System", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
.stApp { background-color: #050a10; }
.card {
    padding: 14px; border-radius: 8px;
    margin-bottom: 10px; border: 1px solid #1a2535;
    font-family: 'Share Tech Mono', monospace;
}
.pre  { background-color: #0a1020; border-left: 4px solid #00aaff; }
.post { background-color: #0a1f14; border-left: 4px solid #00ff88; }
.wait { background-color: #0d1520; border-left: 4px solid #1a2535; }
.danger { background-color: #1f0a0a; border-left: 4px solid #ff3355; }
.label { color: #3a5070; font-size: 0.72em; text-transform: uppercase; letter-spacing: 2px; }
.value { color: white; font-weight: bold; font-size: 1em; }
.highlight { color: #00e5ff; font-weight: bold; }
.warn { color: #ff8800; font-size: 0.75em; }
.green { color: #00ff88; }
.red { color: #ff3355; }
.blue { color: #00aaff; }
.yellow { color: #ffcc00; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# 📅 CALENDARIO EVENTI ECONOMICI
# Formato: (mese, giorno, ora_ET, nome, impatto, strumenti, consensus, precedente, unità)
# Aggiorna manualmente ogni settimana con i dati reali
# ═══════════════════════════════════════════════════════

EVENTS = [
    # ── MAGGIO 2026 ──
    {
        'name': 'NFP',
        'full_name': 'Non-Farm Payrolls',
        'date': '2026-05-01',
        'time_et': '08:30',
        'impact': 'HIGH',
        'instruments': ['SPY', 'QQQ', 'GLD', 'EURUSD=X'],
        'consensus': 130,
        'previous': 228,
        'unit': 'K jobs',
        'description': 'Occupazione USA — muove tutto il mercato'
    },
    {
        'name': 'CPI',
        'full_name': 'CPI Inflazione USA',
        'date': '2026-05-13',
        'time_et': '08:30',
        'impact': 'HIGH',
        'instruments': ['SPY', 'QQQ', 'GLD', 'TLT'],
        'consensus': 2.4,
        'previous': 2.4,
        'unit': '%',
        'description': 'Inflazione USA — impatto su Fed e tassi'
    },
    {
        'name': 'FED',
        'full_name': 'Fed Rate Decision',
        'date': '2026-05-07',
        'time_et': '14:00',
        'impact': 'HIGH',
        'instruments': ['SPY', 'QQQ', 'GLD', 'EURUSD=X'],
        'consensus': 4.25,
        'previous': 4.50,
        'unit': '%',
        'description': 'Decisione tassi Fed — impatto estremo'
    },
    {
        'name': 'EIA_OIL',
        'full_name': 'EIA Crude Oil Inventories',
        'date': '2026-05-07',
        'time_et': '10:30',
        'impact': 'MEDIUM',
        'instruments': ['USO', 'BZ=F', 'XLE'],
        'consensus': -1.2,
        'previous': -2.7,
        'unit': 'M barrels',
        'description': 'Scorte petrolio USA — impatto diretto su oil'
    },
    {
        'name': 'GDP',
        'full_name': 'GDP USA Q1 2026',
        'date': '2026-04-30',
        'time_et': '08:30',
        'impact': 'HIGH',
        'instruments': ['SPY', 'QQQ', 'IWM', 'EURUSD=X'],
        'consensus': 0.4,
        'previous': 2.4,
        'unit': '%',
        'description': 'PIL USA — indicatore macro fondamentale'
    },
    {
        'name': 'EIA_GAS',
        'full_name': 'EIA Natural Gas Storage',
        'date': '2026-05-08',
        'time_et': '10:30',
        'impact': 'MEDIUM',
        'instruments': ['UNG'],
        'consensus': 85,
        'previous': 88,
        'unit': 'Bcf',
        'description': 'Scorte gas naturale USA'
    },
]

# ═══════════════════════════════════════════════════════
# 🔧 FUNZIONI ANALISI
# ═══════════════════════════════════════════════════════

def get_market_data(ticker, period='60d'):
    """Scarica dati e calcola indicatori base."""
    try:
        df = yf.download(ticker, period=period, interval='1d',
                        progress=False, auto_adjust=True)
        if df.empty or len(df) < 20: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close = df['Close']
        price = float(close.iloc[-1])
        prev  = float(close.iloc[-2])
        change_pct = (price - prev) / prev * 100

        # ATR
        hl  = df['High'] - df['Low']
        hpc = (df['High'] - close.shift()).abs()
        lpc = (df['Low']  - close.shift()).abs()
        tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        atr_pct = atr / price * 100

        # Trend
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma50 = float(close.rolling(50).mean().iloc[-1])
        trend = 'UP' if price > ma20 > ma50 else 'DOWN' if price < ma20 < ma50 else 'NEUTRAL'

        # RSI semplice
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        rs    = gain / (loss + 1e-10)
        rsi   = float(100 - 100 / (1 + rs.iloc[-1]))

        # Volatilità storica 5gg
        returns = close.pct_change().dropna()
        vol_5d  = float(returns.tail(5).std() * 100)

        return {
            'ticker':     ticker,
            'price':      price,
            'change_pct': round(change_pct, 2),
            'atr':        round(atr, 4),
            'atr_pct':    round(atr_pct, 2),
            'trend':      trend,
            'rsi':        round(rsi, 1),
            'ma20':       round(ma20, 4),
            'ma50':       round(ma50, 4),
            'vol_5d':     round(vol_5d, 2),
        }
    except:
        return None

def analyze_surprise(consensus, previous, unit):
    """Analizza la sorpresa potenziale consensus vs precedente."""
    if consensus is None or previous is None:
        return 'NEUTRAL', 0
    diff = consensus - previous
    diff_pct = abs(diff / (previous + 1e-10)) * 100
    if diff > 0:
        direction = 'BULLISH'
    elif diff < 0:
        direction = 'BEARISH'
    else:
        direction = 'NEUTRAL'
    return direction, round(diff_pct, 1)

def get_pre_signal(event, market_data):
    """Genera segnale PRE-news basato su consensus + trend."""
    signals = []
    cons_dir, cons_diff = analyze_surprise(
        event['consensus'], event['previous'], event['unit']
    )

    for ticker in event['instruments']:
        if ticker not in market_data or market_data[ticker] is None:
            continue
        md = market_data[ticker]

        # Logica pre-news
        score = 0
        direction = 'WAIT'

        # Consensus favorevole
        if cons_dir == 'BULLISH':
            score += 2
            direction = 'LONG'
        elif cons_dir == 'BEARISH':
            score += 2
            direction = 'SHORT'

        # Trend allineato
        if direction == 'LONG' and md['trend'] == 'UP':
            score += 2
        elif direction == 'SHORT' and md['trend'] == 'DOWN':
            score += 2
        elif md['trend'] == 'NEUTRAL':
            score += 0
        else:
            score -= 1  # trend opposto

        # RSI non estremo
        if 40 < md['rsi'] < 65 and direction == 'LONG':
            score += 1
        elif 35 < md['rsi'] < 60 and direction == 'SHORT':
            score += 1
        elif md['rsi'] > 70 or md['rsi'] < 30:
            score -= 2

        # Volatilità attesa (ATR alto = opportunità)
        if md['atr_pct'] > 1.0:
            score += 1

        # Differenza consensus significativa
        if cons_diff > 10:
            score += 1

        score = max(0, min(10, score))

        # SL/TP basati su ATR
        atr = md['atr']
        if direction == 'LONG':
            entry  = md['price']
            sl     = round(entry - atr * 1.0, 4)
            tp     = round(entry + atr * 2.0, 4)
        elif direction == 'SHORT':
            entry  = md['price']
            sl     = round(entry + atr * 1.0, 4)
            tp     = round(entry - atr * 2.0, 4)
        else:
            entry = sl = tp = md['price']

        signals.append({
            'ticker':    ticker,
            'price':     md['price'],
            'direction': direction,
            'score':     score,
            'rsi':       md['rsi'],
            'trend':     md['trend'],
            'atr_pct':   md['atr_pct'],
            'vol_5d':    md['vol_5d'],
            'entry':     entry,
            'sl':        sl,
            'tp':        tp,
            'cons_dir':  cons_dir,
            'cons_diff': cons_diff,
        })

    return signals

def get_post_signal(event, market_data):
    """Genera segnale POST-news — conferma movimento reale."""
    signals = []
    for ticker in event['instruments']:
        if ticker not in market_data or market_data[ticker] is None:
            continue
        md = market_data[ticker]

        # Post-news: tradi la direzione confermata dal mercato
        change = md['change_pct']
        atr_pct = md['atr_pct']

        # Soglia minima di movimento (0.5x ATR%)
        threshold = atr_pct * 0.5

        if change > threshold:
            direction = 'LONG'
            score = min(10, int(abs(change) / atr_pct * 5) + 3)
        elif change < -threshold:
            direction = 'SHORT'
            score = min(10, int(abs(change) / atr_pct * 5) + 3)
        else:
            direction = 'WAIT'
            score = 2

        # Trend post-news allineato
        if direction == 'LONG' and md['trend'] == 'UP':
            score += 1
        elif direction == 'SHORT' and md['trend'] == 'DOWN':
            score += 1

        score = max(0, min(10, score))

        atr = md['atr']
        if direction == 'LONG':
            entry = md['price']
            sl    = round(entry - atr * 1.5, 4)
            tp    = round(entry + atr * 3.0, 4)
        elif direction == 'SHORT':
            entry = md['price']
            sl    = round(entry + atr * 1.5, 4)
            tp    = round(entry - atr * 3.0, 4)
        else:
            entry = sl = tp = md['price']

        signals.append({
            'ticker':    ticker,
            'price':     md['price'],
            'direction': direction,
            'score':     score,
            'change':    change,
            'threshold': round(threshold, 2),
            'atr_pct':   atr_pct,
            'trend':     md['trend'],
            'rsi':       md['rsi'],
            'entry':     entry,
            'sl':        sl,
            'tp':        tp,
        })

    return signals

def get_event_status(event):
    """Determina se l'evento è: UPCOMING, PRE (entro 2h), LIVE, POST (entro 4h), PAST."""
    et_tz = pytz.timezone('America/New_York')
    now_et = datetime.now(et_tz)

    event_dt_str = f"{event['date']} {event['time_et']}"
    event_dt = et_tz.localize(datetime.strptime(event_dt_str, '%Y-%m-%d %H:%M'))

    diff_minutes = (event_dt - now_et).total_seconds() / 60

    if diff_minutes > 120:
        return 'UPCOMING', diff_minutes
    elif 0 < diff_minutes <= 120:
        return 'PRE', diff_minutes
    elif -240 <= diff_minutes <= 0:
        return 'POST', abs(diff_minutes)
    else:
        return 'PAST', abs(diff_minutes)

# ═══════════════════════════════════════════════════════
# 🖥️ INTERFACCIA
# ═══════════════════════════════════════════════════════

st.markdown(
    '<h2 style="text-align:center;color:#00e5ff;letter-spacing:4px;margin-bottom:0">📅 NEWS TRADING SYSTEM</h2>'
    '<p style="text-align:center;color:#3a5070;font-size:0.75em;letter-spacing:3px">'
    'PRE-NEWS · POST-NEWS · NFP · CPI · FED · EIA · GDP'
    '</p>',
    unsafe_allow_html=True
)

now_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
st.markdown(f'<p style="text-align:center;color:#ff8800;font-size:0.8em">🔄 {now_str}</p>',
            unsafe_allow_html=True)

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("### ⚙️ Impostazioni")
    show_upcoming = st.checkbox("Mostra eventi futuri", value=True)
    show_past     = st.checkbox("Mostra eventi passati", value=False)
    min_score     = st.slider("Score minimo segnali", 0, 10, 5)

    st.markdown("---")
    st.markdown("### 📖 Come funziona")
    st.markdown("🔵 **PRE-NEWS** (entro 2h)")
    st.markdown("→ Entri prima del dato")
    st.markdown("→ Basato su consensus vs precedente")
    st.markdown("→ SL stretto: 1x ATR")
    st.markdown("")
    st.markdown("🟢 **POST-NEWS** (entro 4h)")
    st.markdown("→ Entri dopo conferma")
    st.markdown("→ Basato su movimento reale")
    st.markdown("→ SL normale: 1.5x ATR")
    st.markdown("")
    st.markdown("⚠️ **REGOLA D'ORO**")
    st.markdown("Esci SEMPRE prima del dato")
    st.markdown("se sei in PRE-NEWS!")

    st.markdown("---")
    st.markdown("### 🗓️ Prossimi eventi")
    et_tz = pytz.timezone('America/New_York')
    now_et = datetime.now(et_tz)
    for ev in sorted(EVENTS, key=lambda x: x['date']+x['time_et']):
        status, mins = get_event_status(ev)
        if status != 'PAST':
            color = '#00aaff' if status == 'PRE' else '#00ff88' if status == 'POST' else '#3a5070'
            st.markdown(f'<span style="color:{color}">● {ev["name"]}</span> {ev["date"]} {ev["time_et"]} ET',
                       unsafe_allow_html=True)

# ── CARICA DATI ──
all_tickers = list(set(t for ev in EVENTS for t in ev['instruments']))
with st.spinner('⏳ Caricamento dati mercato...'):
    market_data = {}
    for ticker in all_tickers:
        market_data[ticker] = get_market_data(ticker)

# ── PROCESSA EVENTI ──
pre_events  = []
post_events = []
upcoming    = []
past_events = []

for event in EVENTS:
    status, mins = get_event_status(event)
    if status == 'PRE':
        pre_events.append((event, mins))
    elif status == 'POST':
        post_events.append((event, mins))
    elif status == 'UPCOMING':
        upcoming.append((event, mins))
    else:
        past_events.append((event, mins))

# ── METRICHE ──
c1, c2, c3, c4 = st.columns(4)
c1.metric("🔵 PRE",     len(pre_events))
c2.metric("🟢 POST",    len(post_events))
c3.metric("⏳ PROSSIMI", len(upcoming))
c4.metric("📋 TOTALE",  len(EVENTS))
st.markdown("---")

# ══════════════════════════════
# 🔵 SEGNALI PRE-NEWS
# ══════════════════════════════
if pre_events:
    st.markdown('<p style="color:#00aaff;letter-spacing:3px;font-size:0.9em">🔵 ZONA PRE-NEWS — ENTRA ORA</p>',
                unsafe_allow_html=True)
    for event, mins_to_event in pre_events:
        cons_dir, cons_diff = analyze_surprise(event['consensus'], event['previous'], event['unit'])
        dir_color = '#00ff88' if cons_dir == 'BULLISH' else '#ff3355' if cons_dir == 'BEARISH' else '#ffcc00'
        dir_icon  = '▲' if cons_dir == 'BULLISH' else '▼' if cons_dir == 'BEARISH' else '→'

        st.markdown(
            f'<div class="card pre">'
            f'<div style="display:flex;justify-content:space-between">'
            f'<div><b style="color:#00aaff;font-size:1.1em">📰 {event["full_name"]}</b><br>'
            f'<span style="color:#3a5070;font-size:0.75em">{event["date"]} {event["time_et"]} ET · {event["impact"]}</span></div>'
            f'<div style="text-align:right"><span style="color:#ff8800;font-size:0.85em">⏱ {int(mins_to_event)}min al dato</span></div>'
            f'</div>'
            f'<div style="margin-top:10px;display:grid;grid-template-columns:repeat(3,1fr);gap:8px">'
            f'<div><div class="label">Consensus</div><div class="value">{event["consensus"]} {event["unit"]}</div></div>'
            f'<div><div class="label">Precedente</div><div class="value">{event["previous"]} {event["unit"]}</div></div>'
            f'<div><div class="label">Bias</div><div class="value" style="color:{dir_color}">{dir_icon} {cons_dir} ({cons_diff}%)</div></div>'
            f'</div>'
            f'<div style="margin-top:8px;color:#3a5070;font-size:0.75em">{event["description"]}</div>'
            f'<div style="margin-top:8px;padding:6px;background:#050a10;border-radius:4px;color:#ff8800;font-size:0.75em">'
            f'⚠️ ESCI prima del dato — non tenere posizioni aperte al rilascio!'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Segnali per strumento
        signals = get_pre_signal(event, market_data)
        signals = [s for s in signals if s['score'] >= min_score]
        if signals:
            for sig in sorted(signals, key=lambda x: x['score'], reverse=True):
                sig_color = '#00ff88' if sig['direction'] == 'LONG' else '#ff3355' if sig['direction'] == 'SHORT' else '#3a5070'
                sig_icon  = '▲ LONG' if sig['direction'] == 'LONG' else '▼ SHORT' if sig['direction'] == 'SHORT' else '⚪ WAIT'
                trend_ico = '📈' if sig['trend'] == 'UP' else '📉' if sig['trend'] == 'DOWN' else '➡️'
                rr = 2.0

                st.markdown(
                    f'<div class="card pre" style="margin-left:16px;border-left:2px solid {sig_color}">'
                    f'<div style="display:flex;justify-content:space-between">'
                    f'<b style="color:white">{sig["ticker"]}</b>'
                    f'<span style="color:{sig_color}">{sig_icon} &nbsp; <b>{sig["score"]}/10</b></span>'
                    f'</div>'
                    f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-top:8px">'
                    f'<div><div class="label">Prezzo</div><div class="value">{sig["price"]:.4f}</div></div>'
                    f'<div><div class="label">RSI</div><div class="value">{sig["rsi"]}</div></div>'
                    f'<div><div class="label">ATR%</div><div class="value">{sig["atr_pct"]}%</div></div>'
                    f'<div><div class="label">Trend</div><div class="value">{trend_ico}</div></div>'
                    f'</div>'
                    f'<div style="margin-top:8px;padding:6px;background:#0a0f15;border-radius:4px;font-size:0.78em">'
                    f'<span style="color:#00ff88">TP: {sig["tp"]:.4f}</span>'
                    f' &nbsp;|&nbsp; <span style="color:#ff3355">SL: {sig["sl"]:.4f}</span>'
                    f' &nbsp;|&nbsp; <span style="color:#ffcc00">R/R: {rr}:1</span>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

# ══════════════════════════════
# 🟢 SEGNALI POST-NEWS
# ══════════════════════════════
if post_events:
    st.markdown('<p style="color:#00ff88;letter-spacing:3px;font-size:0.9em">🟢 ZONA POST-NEWS — CONFERMA</p>',
                unsafe_allow_html=True)
    for event, mins_since in post_events:
        st.markdown(
            f'<div class="card post">'
            f'<div style="display:flex;justify-content:space-between">'
            f'<div><b style="color:#00ff88;font-size:1.1em">✅ {event["full_name"]}</b><br>'
            f'<span style="color:#3a5070;font-size:0.75em">{event["date"]} {event["time_et"]} ET</span></div>'
            f'<div style="text-align:right"><span style="color:#ff8800;font-size:0.85em">⏱ {int(mins_since)}min fa</span></div>'
            f'</div>'
            f'<div style="margin-top:8px;color:#3a5070;font-size:0.75em">{event["description"]}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        signals = get_post_signal(event, market_data)
        signals = [s for s in signals if s['score'] >= min_score]
        if signals:
            for sig in sorted(signals, key=lambda x: x['score'], reverse=True):
                sig_color = '#00ff88' if sig['direction'] == 'LONG' else '#ff3355' if sig['direction'] == 'SHORT' else '#3a5070'
                sig_icon  = '▲ LONG' if sig['direction'] == 'LONG' else '▼ SHORT' if sig['direction'] == 'SHORT' else '⚪ WAIT'
                trend_ico = '📈' if sig['trend'] == 'UP' else '📉' if sig['trend'] == 'DOWN' else '➡️'
                chg_color = '#00ff88' if sig['change'] > 0 else '#ff3355'

                st.markdown(
                    f'<div class="card post" style="margin-left:16px;border-left:2px solid {sig_color}">'
                    f'<div style="display:flex;justify-content:space-between">'
                    f'<b style="color:white">{sig["ticker"]}</b>'
                    f'<span style="color:{sig_color}">{sig_icon} &nbsp; <b>{sig["score"]}/10</b></span>'
                    f'</div>'
                    f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-top:8px">'
                    f'<div><div class="label">Prezzo</div><div class="value">{sig["price"]:.4f}</div></div>'
                    f'<div><div class="label">Movimento</div><div class="value" style="color:{chg_color}">{sig["change"]:+.2f}%</div></div>'
                    f'<div><div class="label">Soglia</div><div class="value">{sig["threshold"]}%</div></div>'
                    f'<div><div class="label">Trend</div><div class="value">{trend_ico}</div></div>'
                    f'</div>'
                    f'<div style="margin-top:8px;padding:6px;background:#0a0f15;border-radius:4px;font-size:0.78em">'
                    f'<span style="color:#00ff88">TP: {sig["tp"]:.4f}</span>'
                    f' &nbsp;|&nbsp; <span style="color:#ff3355">SL: {sig["sl"]:.4f}</span>'
                    f' &nbsp;|&nbsp; <span style="color:#ffcc00">R/R: 2:1</span>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

# ══════════════════════════════
# ⏳ PROSSIMI EVENTI
# ══════════════════════════════
if show_upcoming and upcoming:
    st.markdown('<p style="color:#3a5070;letter-spacing:3px;font-size:0.9em">⏳ PROSSIMI EVENTI</p>',
                unsafe_allow_html=True)
    for event, mins_to in sorted(upcoming, key=lambda x: x[1]):
        cons_dir, cons_diff = analyze_surprise(event['consensus'], event['previous'], event['unit'])
        dir_color = '#00ff88' if cons_dir == 'BULLISH' else '#ff3355' if cons_dir == 'BEARISH' else '#ffcc00'
        impact_color = '#ff3355' if event['impact'] == 'HIGH' else '#ffcc00'

        hours = int(mins_to // 60)
        mins  = int(mins_to % 60)
        time_str = f"{hours}h {mins}m" if hours > 0 else f"{mins}m"

        st.markdown(
            f'<div class="card wait">'
            f'<div style="display:flex;justify-content:space-between;align-items:center">'
            f'<div>'
            f'<b style="color:white">{event["full_name"]}</b>'
            f'<span style="color:{impact_color};font-size:0.75em;margin-left:8px">● {event["impact"]}</span><br>'
            f'<span style="color:#3a5070;font-size:0.75em">{event["date"]} · {event["time_et"]} ET</span>'
            f'</div>'
            f'<div style="text-align:right">'
            f'<span style="color:#ff8800">⏱ {time_str}</span><br>'
            f'<span style="color:{dir_color};font-size:0.8em">{"▲" if cons_dir=="BULLISH" else "▼" if cons_dir=="BEARISH" else "→"} {cons_dir}</span>'
            f'</div>'
            f'</div>'
            f'<div style="margin-top:8px;display:grid;grid-template-columns:repeat(3,1fr);gap:6px">'
            f'<div><div class="label">Consensus</div><div class="value">{event["consensus"]} {event["unit"]}</div></div>'
            f'<div><div class="label">Precedente</div><div class="value">{event["previous"]} {event["unit"]}</div></div>'
            f'<div><div class="label">Strumenti</div><div class="value" style="font-size:0.8em">{", ".join(event["instruments"][:3])}</div></div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#1a2535;font-size:0.65em;letter-spacing:2px">'
    'SOLO USO INFORMATIVO — NON CONSULENZA FINANZIARIA — AGGIORNA CALENDARIO SETTIMANALMENTE'
    '</p>',
    unsafe_allow_html=True
)
