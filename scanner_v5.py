import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import time
import warnings
warnings.filterwarnings('ignore')

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except:
    HMM_AVAILABLE = False

# ═══════════════════════════════════════════════════════
# ⚙️ CONFIGURAZIONE
# ═══════════════════════════════════════════════════════
st.set_page_config(page_title="Scanner Ultra V8", layout="centered")

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
.regime-bull { color: #00ff88; font-size: 0.8em; }
.regime-bear { color: #ff3355; font-size: 0.8em; }
.regime-lat  { color: #ffcc00; font-size: 0.8em; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# 🔑 TELEGRAM
# ═══════════════════════════════════════════════════════
TELEGRAM_TOKEN   = "8661470519:AAFJV3D2kzaXIXx_0EmvEtXkXnL_4hm-HC8"
TELEGRAM_CHAT_ID = "5675996555"

def send_telegram(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={
            'chat_id': TELEGRAM_CHAT_ID,
            'text':    message,
            'parse_mode': 'HTML'
        }, timeout=5)
    except:
        pass

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
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta.clip(upper=0))
    avg_gain = gain.iloc[1:period+1].mean()
    avg_loss = loss.iloc[1:period+1].mean()
    gains, losses = [avg_gain], [avg_loss]
    for i in range(period+1, len(gain)):
        gains.append((gains[-1]*(period-1) + gain.iloc[i]) / period)
        losses.append((losses[-1]*(period-1) + loss.iloc[i]) / period)
    if not gains or losses[-1] == 0:
        return 50
    rs = gains[-1] / losses[-1]
    return round(100 - 100 / (1 + rs), 2)

def calc_atr(df, period=14):
    hl  = df['High'] - df['Low']
    hpc = (df['High'] - df['Close'].shift()).abs()
    lpc = (df['Low']  - df['Close'].shift()).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    return tr.rolling(period).mean().iloc[-1]

def calc_adx(df, period=14):
    try:
        high, low, close = df['High'], df['Low'], df['Close']
        pdm = high.diff().clip(lower=0)
        mdm = (-low.diff()).clip(lower=0)
        mask = pdm > mdm
        pdm  = pdm.where(mask, 0)
        mdm  = mdm.where(~mask, 0)
        tr   = pd.concat([high-low,(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1)
        atr  = tr.ewm(alpha=1/period, adjust=False).mean()
        dip  = 100*pdm.ewm(alpha=1/period, adjust=False).mean()/(atr+1e-10)
        dim  = 100*mdm.ewm(alpha=1/period, adjust=False).mean()/(atr+1e-10)
        dx   = 100*(dip-dim).abs()/(dip+dim+1e-10)
        return round(float(dx.ewm(alpha=1/period, adjust=False).mean().iloc[-1]), 1)
    except:
        return 0

def calc_macd(series, fast=12, slow=26, signal=9):
    ef = series.ewm(span=fast, adjust=False).mean()
    es = series.ewm(span=slow, adjust=False).mean()
    m  = ef - es
    s  = m.ewm(span=signal, adjust=False).mean()
    return float(m.iloc[-1]), float(s.iloc[-1]), float((m-s).iloc[-1])

def calc_bollinger(series, period=20, std_dev=2):
    ma  = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = ma + std_dev * std
    lower = ma - std_dev * std
    bw  = float(upper.iloc[-1] - lower.iloc[-1])
    pctb = float((series.iloc[-1] - lower.iloc[-1]) / (bw + 1e-10))
    return round(pctb, 3), round(bw, 4)

def calc_stoch_rsi(series, rsi_period=14, stoch_period=14, smooth_k=3, smooth_d=3):
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/rsi_period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/rsi_period, adjust=False).mean()
    rsi_s = 100 - 100/(1 + gain/(loss+1e-10))
    min_r = rsi_s.rolling(stoch_period).min()
    max_r = rsi_s.rolling(stoch_period).max()
    k = (100*(rsi_s - min_r)/(max_r - min_r + 1e-10)).rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    return round(float(k.iloc[-1]), 1), round(float(d.iloc[-1]), 1)

# ═══════════════════════════════════════════════════════
# 🧠 HMM REGIME DETECTION
# ═══════════════════════════════════════════════════════

_hmm_cache = {}

def get_hmm_regime(close_series, ticker, n_states=3):
    """Detecta regime HMM con cache per performance."""
    cache_key = f"{ticker}_{datetime.now().strftime('%Y%m%d%H')}"
    if cache_key in _hmm_cache:
        return _hmm_cache[cache_key]

    if not HMM_AVAILABLE or len(close_series) < 100:
        return 'UNKNOWN', 0.0

    try:
        ret = close_series.pct_change().dropna()
        vol = ret.rolling(5).std().dropna()
        mom = close_series.pct_change(10).dropna()
        df_f = pd.DataFrame({'r': ret, 'v': vol, 'm': mom}).dropna()

        if len(df_f) < 60:
            return 'UNKNOWN', 0.0

        X = df_f.values
        X = (X - X.mean(0)) / (X.std(0) + 1e-10)

        model = hmm.GaussianHMM(n_components=n_states, covariance_type='full',
                                n_iter=100, random_state=42)
        model.fit(X)
        states = model.predict(X)

        sr = {s: df_f['r'][states == s].mean() for s in range(n_states)}
        ss = sorted(sr.items(), key=lambda x: x[1])
        cur = states[-1]

        if cur == ss[-1][0]:   regime = 'BULL'
        elif cur == ss[0][0]:  regime = 'BEAR'
        else:                  regime = 'LATERAL'

        _, posteriors = model.score_samples(X)
        prob = round(float(posteriors[-1][cur]) * 100, 1)

        _hmm_cache[cache_key] = (regime, prob)
        return regime, prob

    except:
        return 'UNKNOWN', 0.0

# ═══════════════════════════════════════════════════════
# 📡 SEGNALE PRINCIPALE V8
# ═══════════════════════════════════════════════════════

def get_signal(ticker, info):
    try:
        df = yf.download(ticker, period='300d', interval='1d',
                        progress=False, auto_adjust=True)
        if df.empty or len(df) < 210: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close  = df['Close']
        price  = float(close.iloc[-1])
        prev_h = float(df['High'].iloc[-2])
        prev_l = float(df['Low'].iloc[-2])

        ma20  = float(close.rolling(20).mean().iloc[-1])
        ma50  = float(close.rolling(50).mean().iloc[-1])
        ma200 = float(close.rolling(200).mean().iloc[-1])

        rsi              = calc_rsi(close)
        atr              = calc_atr(df)
        adx              = calc_adx(df)
        macd, macd_sig, macd_hist = calc_macd(close)
        pct_b, bband_w   = calc_bollinger(close)
        stoch_k, stoch_d = calc_stoch_rsi(close)

        vol_ratio = 0.0
        if not info['forex']:
            vol       = float(df['Volume'].iloc[-1])
            vol_avg   = float(df['Volume'].rolling(20).mean().iloc[-1])
            vol_ratio = vol / vol_avg if vol_avg > 0 else 0

        uptrend   = price > ma50 > ma200
        downtrend = price < ma50 < ma200
        bull_break = price > prev_h
        bear_break = price < prev_l
        vol_ok     = info['forex'] or vol_ratio > 1.2
        strong_t   = adx > 22
        macd_bull  = macd > macd_sig and macd_hist > 0
        macd_bear  = macd < macd_sig and macd_hist < 0
        rsi_buy_ok  = 42 < rsi < 63
        rsi_sell_ok = 37 < rsi < 58

        # ── HMM REGIME ──
        regime, regime_prob = get_hmm_regime(close, ticker)

        # ── SCORE V7 BASE ──
        score_buy = 0
        if bull_break:      score_buy += 2
        if uptrend:         score_buy += 2
        if macd_bull:       score_buy += 1
        if vol_ratio > 1.5: score_buy += 2
        elif vol_ok:        score_buy += 1
        if rsi_buy_ok:      score_buy += 1
        if strong_t:        score_buy += 1
        if price > ma20:    score_buy += 1
        if rsi > 70:        score_buy -= 3
        if rsi > 63:        score_buy -= 1
        if pct_b > 1.0:     score_buy -= 2
        if not macd_bull:   score_buy -= 1
        if vol_ratio < 0.8: score_buy -= 1
        if not uptrend:     score_buy -= 1

        score_sell = 0
        if bear_break:      score_sell += 2
        if downtrend:       score_sell += 2
        if macd_bear:       score_sell += 1
        if vol_ratio > 1.5: score_sell += 2
        elif vol_ok:        score_sell += 1
        if rsi_sell_ok:     score_sell += 1
        if strong_t:        score_sell += 1
        if price < ma20:    score_sell += 1
        if rsi < 30:        score_sell -= 3
        if rsi < 37:        score_sell -= 1
        if pct_b < 0.0:     score_sell -= 2
        if not macd_bear:   score_sell -= 1
        if vol_ratio < 0.8: score_sell -= 1
        if not downtrend:   score_sell -= 1

        # ── HMM SOFT FILTER ──
        tp_mult_buy  = 3.0
        tp_mult_sell = 3.0

        if regime != 'UNKNOWN':
            # BUY adjustment
            if regime == 'BULL':
                score_buy += 2
                tp_mult_buy = 3.6   # TP +20%
            elif regime == 'LATERAL':
                score_buy -= 3
            elif regime == 'BEAR':
                score_buy -= 2

            # SELL adjustment
            if regime == 'BEAR':
                score_sell += 2
                tp_mult_sell = 3.6
            elif regime == 'LATERAL':
                score_sell -= 3
            elif regime == 'BULL':
                score_sell -= 2

        score_buy  = max(0, min(10, score_buy))
        score_sell = max(0, min(10, score_sell))

        # SL/TP
        sl_long  = price - atr * 1.5
        tp_long  = price + atr * tp_mult_buy
        sl_short = price + atr * 1.5
        tp_short = price - atr * tp_mult_sell

        # Decisione
        if (score_buy >= 7 and bull_break and rsi_buy_ok
                and vol_ok and macd_bull and pct_b < 0.95):
            status = 'BUY'
        elif (score_sell >= 7 and bear_break and rsi_sell_ok
                and vol_ok and macd_bear and pct_b > 0.05):
            status = 'SELL'
        else:
            status = 'WAIT'

        warnings = []
        if rsi > 70:              warnings.append('RSI IPERCOMPRATO')
        if rsi < 30:              warnings.append('RSI IPERVENDUTO')
        if vol_ratio > 0 and vol_ratio < 0.8: warnings.append('VOLUME BASSO')
        if adx < 15:              warnings.append('TREND DEBOLE')
        if pct_b > 1.0:           warnings.append('SOPRA BOLLINGER')
        if pct_b < 0.0:           warnings.append('SOTTO BOLLINGER')
        if regime == 'LATERAL':   warnings.append('HMM: MERCATO LATERALE')

        return {
            'ticker':    ticker,
            'name':      info['name'],
            'cat':       info['cat'],
            'forex':     info['forex'],
            'price':     price,
            'rsi':       round(rsi, 1),
            'adx':       round(adx, 1),
            'macd_hist': round(macd_hist, 4),
            'pct_b':     round(pct_b, 2),
            'stoch_k':   stoch_k,
            'stoch_d':   stoch_d,
            'vol_ratio': round(vol_ratio, 1),
            'ma20':      round(ma20, 4),
            'ma50':      round(ma50, 4),
            'ma200':     round(ma200, 4),
            'atr':       round(atr, 4),
            'uptrend':   uptrend,
            'downtrend': downtrend,
            'regime':    regime,
            'regime_prob': regime_prob,
            'score_buy':  score_buy,
            'score_sell': score_sell,
            'sl_long':   round(sl_long, 4),
            'tp_long':   round(tp_long, 4),
            'sl_short':  round(sl_short, 4),
            'tp_short':  round(tp_short, 4),
            'status':    status,
            'warnings':  warnings,
        }
    except:
        return None

# ═══════════════════════════════════════════════════════
# 📱 NOTIFICA TELEGRAM
# ═══════════════════════════════════════════════════════

_notified = set()

def notify_signal(r):
    """Manda notifica Telegram solo per nuovi segnali."""
    key = f"{r['ticker']}_{r['status']}_{datetime.now().strftime('%Y%m%d')}"
    if key in _notified:
        return
    _notified.add(key)

    icon  = '🟢' if r['status'] == 'BUY' else '🔴'
    score = r['score_buy'] if r['status'] == 'BUY' else r['score_sell']
    sl    = r['sl_long']   if r['status'] == 'BUY' else r['sl_short']
    tp    = r['tp_long']   if r['status'] == 'BUY' else r['tp_short']
    rr    = round(abs(tp - r['price']) / abs(r['price'] - sl + 1e-10), 1)

    regime_icon = '📈' if r['regime']=='BULL' else '📉' if r['regime']=='BEAR' else '➡️'

    msg = (
        f"{icon} <b>SEGNALE V8 — {r['status']}</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"<b>{r['name']}</b> ({r['ticker']})\n"
        f"Score: <b>{score}/10</b>\n"
        f"Prezzo: <b>{r['price']:.4f}</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"TP: <b>{tp:.4f}</b>\n"
        f"SL: <b>{sl:.4f}</b>\n"
        f"R/R: <b>{rr}:1</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"RSI: {r['rsi']} | ADX: {r['adx']}\n"
        f"HMM: {regime_icon} {r['regime']} ({r['regime_prob']}%)\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"Solo uso informativo"
    )
    send_telegram(msg)

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
    sig_icon  = '▲ BUY'  if r['status']=='BUY' else '▼ SELL' if r['status']=='SELL' else '⚪ WAIT'
    score     = r['score_buy'] if r['status']=='BUY' else r['score_sell'] if r['status']=='SELL' else max(r['score_buy'], r['score_sell'])
    sc_class  = score_color(score)
    trend_ico = '📈' if r['uptrend'] else '📉' if r['downtrend'] else '➡️'
    vol_str   = f"{r['vol_ratio']:.1f}x" if r['vol_ratio'] > 0 else 'N/A'

    macd_color = '#00ff88' if r['macd_hist'] > 0 else '#ff3355'
    macd_str   = f"{'▲' if r['macd_hist']>0 else '▼'} {abs(r['macd_hist']):.3f}"
    stoch_color = '#ff3355' if r['stoch_k']>80 else '#00ff88' if r['stoch_k']<20 else 'white'

    # Regime HMM badge
    reg_color = '#00ff88' if r['regime']=='BULL' else '#ff3355' if r['regime']=='BEAR' else '#ffcc00' if r['regime']=='LATERAL' else '#3a5070'
    reg_icon  = '▲' if r['regime']=='BULL' else '▼' if r['regime']=='BEAR' else '→'
    regime_html = (
        f'<div style="margin-top:8px;padding:6px;background:#0a0f15;border-radius:4px;'
        f'display:flex;justify-content:space-between;align-items:center">'
        f'<span style="color:#3a5070;font-size:0.72em;letter-spacing:2px">HMM REGIME</span>'
        f'<span style="color:{reg_color};font-weight:bold">{reg_icon} {r["regime"]} '
        f'<span style="color:#3a5070;font-size:0.8em">({r["regime_prob"]}%)</span></span>'
        f'</div>'
    )

    levels_html = ''
    if r['status'] == 'BUY':
        rr = round((r['tp_long']-r['price'])/(r['price']-r['sl_long']+1e-10),1)
        levels_html = (
            '<div style="margin-top:10px;padding:8px;background:#0a0f15;border-radius:4px;font-size:0.78em">'
            f'<span style="color:#00ff88">TP: {r["tp_long"]:.4f}</span>'
            f' &nbsp;|&nbsp; <span style="color:#ff3355">SL: {r["sl_long"]:.4f}</span>'
            f' &nbsp;|&nbsp; <span style="color:#ffcc00">R/R: {rr}:1</span>'
            '</div>'
        )
    elif r['status'] == 'SELL':
        rr = round((r['price']-r['tp_short'])/(r['sl_short']-r['price']+1e-10),1)
        levels_html = (
            '<div style="margin-top:10px;padding:8px;background:#0a0f15;border-radius:4px;font-size:0.78em">'
            f'<span style="color:#00ff88">TP: {r["tp_short"]:.4f}</span>'
            f' &nbsp;|&nbsp; <span style="color:#ff3355">SL: {r["sl_short"]:.4f}</span>'
            f' &nbsp;|&nbsp; <span style="color:#ffcc00">R/R: {rr}:1</span>'
            '</div>'
        )

    ma_html = (
        f'<div style="margin-top:8px;font-size:0.72em;color:#3a5070">'
        f'MA20: <span style="color:white">{r["ma20"]:.2f}</span> &nbsp;'
        f'MA50: <span style="color:white">{r["ma50"]:.2f}</span> &nbsp;'
        f'MA200: <span style="color:white">{r["ma200"]:.2f}</span>'
        f'</div>'
    )

    warn_html = ''
    if r['warnings']:
        warn_html = '<div style="margin-top:6px">' + ' '.join(
            [f'<span class="warn">⚠️ {w}</span>' for w in r['warnings']]
        ) + '</div>'

    st.markdown(
        f'<div class="card {css}">'
        f'<div style="display:flex;justify-content:space-between;align-items:center">'
        f'<div><b style="color:white;font-size:1.1em">{r["name"]}</b>'
        f'<span style="color:#3a5070;font-size:0.75em;margin-left:8px">{r["ticker"]} · {r["cat"].upper()}</span></div>'
        f'<div style="text-align:right">'
        f'<b style="color:{sig_color};font-size:1em">{sig_icon}</b><br>'
        f'<span class="{sc_class}">{score}/10</span>'
        f'</div></div>'
        f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:10px">'
        f'<div><div class="label">Prezzo</div><div class="value">{r["price"]:.4f}</div></div>'
        f'<div><div class="label">RSI</div><div class="value">{r["rsi"]}</div></div>'
        f'<div><div class="label">ADX</div><div class="value">{r["adx"]}</div></div>'
        f'<div><div class="label">Trend</div><div class="value">{trend_ico} {vol_str}</div></div>'
        f'</div>'
        f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:8px">'
        f'<div><div class="label">MACD</div><div class="value" style="color:{macd_color}">{macd_str}</div></div>'
        f'<div><div class="label">StochRSI</div><div class="value" style="color:{stoch_color}">{r["stoch_k"]}</div></div>'
        f'<div><div class="label">%B Boll</div><div class="value">{r["pct_b"]:.2f}</div></div>'
        f'<div><div class="label">ATR</div><div class="value">{r["atr"]:.2f}</div></div>'
        f'</div>'
        f'{regime_html}'
        f'{ma_html}'
        f'{levels_html}'
        f'{warn_html}'
        f'</div>',
        unsafe_allow_html=True
    )

# ═══════════════════════════════════════════════════════
# 🖥️ INTERFACCIA
# ═══════════════════════════════════════════════════════

st.markdown(
    '<h2 style="text-align:center;color:#00e5ff;letter-spacing:4px;margin-bottom:0">💎 SCANNER ULTRA V8</h2>'
    '<p style="text-align:center;color:#3a5070;font-size:0.75em;letter-spacing:3px">'
    'V7 + HMM SOFT FILTER + TELEGRAM ALERTS'
    '</p>',
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
    telegram_on = st.checkbox("Notifiche Telegram", value=True)

    st.markdown("---")
    st.markdown("### 📊 Filtri V8")
    st.markdown("✅ RSI Wilder 42-63 / 37-58")
    st.markdown("✅ MA200 reale (300gg)")
    st.markdown("✅ ADX Wilder > 22")
    st.markdown("✅ MACD conferma")
    st.markdown("✅ Bollinger %B")
    st.markdown("✅ Stochastic RSI")
    st.markdown("✅ **HMM Soft Filter**")
    st.markdown("  BULL → score +2, TP +20%")
    st.markdown("  LATERAL → score -3")
    st.markdown("  BEAR → score -2")
    st.markdown("✅ Score minimo: 7/10")

    st.markdown("---")
    st.markdown("### 📱 Telegram")
    if telegram_on:
        st.markdown("🟢 Notifiche ATTIVE")
        st.markdown("@scannerv8_massimo_bot")
    else:
        st.markdown("🔴 Notifiche DISATTIVE")

    st.markdown("---")
    st.markdown("### ⏰ Sistemi attivi")
    st.markdown("🛢️ EIA Petrolio: mer 16:30")
    st.markdown("🔥 EIA Gas: gio 16:30")
    st.markdown("🥇 Gold IB: ogni sera")
    st.markdown("📅 Fed: 07/05 20:00")

# ── SCAN ──
with st.spinner('⏳ Scansione V8 in corso...'):
    results = []
    for ticker, info in UNIVERSE.items():
        if info['cat'] not in cat_filter: continue
        r = get_signal(ticker, info)
        if r: results.append(r)

# ── SEPARA ──
buys  = sorted([r for r in results if r['status']=='BUY'  and r['score_buy']  >= min_score], key=lambda x: x['score_buy'],  reverse=True)
sells = sorted([r for r in results if r['status']=='SELL' and r['score_sell'] >= min_score], key=lambda x: x['score_sell'], reverse=True)
waits = [r for r in results if r['status']=='WAIT']

# ── NOTIFICHE ──
if telegram_on:
    for r in buys + sells:
        notify_signal(r)

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
        '<b style="color:#3a5070;letter-spacing:3px;font-size:0.9em">NESSUN SEGNALE DI QUALITA</b><br>'
        '<span style="color:#1a2535;font-size:0.75em">Filtri V8 + HMM attivi</span>'
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
    for r in sorted(waits, key=lambda x: max(x['score_buy'],x['score_sell']), reverse=True):
        render_card(r)

st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#1a2535;font-size:0.65em;letter-spacing:2px">'
    'SOLO USO INFORMATIVO — NON CONSULENZA FINANZIARIA'
    '</p>',
    unsafe_allow_html=True
)

time.sleep(60)
st.rerun()



