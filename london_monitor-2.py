#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════
# 🏙️ LONDON BREAKOUT MONITOR — GBP/USD
# Per PythonAnywhere — task schedulato ogni ora
# Parametri ottimizzati dal backtest 10 anni:
# - Solo Mer e Gio
# - Range asiatico 20-40 pips
# - TP 1.0x range
# - SL 5 pips oltre range
# ═══════════════════════════════════════════════════════

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import pytz
import json
import os

# ═══════════════════════════════════════════════════════
# ⚙️ CONFIGURAZIONE
# ═══════════════════════════════════════════════════════

TELEGRAM_TOKEN   = "8661470519:AAFJV3D2kzaXIXx_0EmvEtXkXnL_4hm-HC8"
TELEGRAM_CHAT_ID = "5675996555"

CONFIG = {
    'pip':            0.0001,
    'allowed_days':   [2, 3],      # Mer=2, Gio=3
    'asian_start':    0,
    'asian_end':      7,
    'london_start':   7,
    'london_trap':    10,
    'london_close':   13,
    'min_range_pips': 20,
    'max_range_pips': 40,
    'trap_break_pips': 8,
    'trap_return_bars': 3,
    'sl_buffer_pips': 5,
    'tp_range_mult':  1.0,
    'min_adx':        15,

    # File di stato — traccia cosa è già stato notificato oggi
    'state_file': '/home/otageva/london_state.json',

    # News days da saltare — aggiorna ogni settimana!
    'news_skip_days': [
        # Fed 2025
        '2025-01-29', '2025-03-19', '2025-05-07', '2025-06-18',
        '2025-07-30', '2025-09-17', '2025-11-05', '2025-12-17',
        # NFP 2025
        '2025-01-10', '2025-02-07', '2025-03-07', '2025-04-04',
        '2025-05-02', '2025-06-06', '2025-07-03', '2025-08-01',
        '2025-09-05', '2025-10-03', '2025-11-07', '2025-12-05',
        # BOE 2025
        '2025-02-06', '2025-03-20', '2025-05-08', '2025-06-19',
        '2025-08-07', '2025-09-18', '2025-11-06', '2025-12-18',
        # Fed 2026
        '2026-01-28', '2026-03-18', '2026-05-06', '2026-06-17',
        '2026-07-29', '2026-09-16', '2026-11-04', '2026-12-16',
        # NFP 2026
        '2026-01-09', '2026-02-06', '2026-03-06', '2026-04-03',
        '2026-05-01', '2026-06-05', '2026-07-02', '2026-08-07',
        '2026-09-04', '2026-10-02', '2026-11-06', '2026-12-04',
        # BOE 2026
        '2026-02-05', '2026-03-19', '2026-05-07', '2026-06-18',
        '2026-08-06', '2026-09-17', '2026-11-05', '2026-12-17',
    ],
}

# ═══════════════════════════════════════════════════════
# 📱 TELEGRAM
# ═══════════════════════════════════════════════════════

def send_telegram(message):
    try:
        url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        resp = requests.post(url, data={
            'chat_id':    TELEGRAM_CHAT_ID,
            'text':       message,
            'parse_mode': 'HTML'
        }, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        log(f"Telegram error: {e}")
        return False

# ═══════════════════════════════════════════════════════
# 📋 STATO GIORNALIERO
# ═══════════════════════════════════════════════════════

def load_state():
    """Carica stato del giorno corrente."""
    try:
        if os.path.exists(CONFIG['state_file']):
            with open(CONFIG['state_file'], 'r') as f:
                state = json.load(f)
            # Reset se è un nuovo giorno
            today = datetime.now(pytz.UTC).strftime('%Y-%m-%d')
            if state.get('date') != today:
                return fresh_state(today)
            return state
    except:
        pass
    today = datetime.now(pytz.UTC).strftime('%Y-%m-%d')
    return fresh_state(today)

def fresh_state(today):
    return {
        'date':              today,
        'range_notified':    False,
        'trap_notified':     False,
        'close_notified':    False,
        'trade_active':      False,
        'trade_direction':   None,
        'trade_entry':       None,
        'trade_sl':          None,
        'trade_tp':          None,
        'asian_high':        None,
        'asian_low':         None,
        'asian_range_pips':  None,
    }

def save_state(state):
    try:
        with open(CONFIG['state_file'], 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        log(f"State save error: {e}")

# ═══════════════════════════════════════════════════════
# 📊 DATI E INDICATORI
# ═══════════════════════════════════════════════════════

def log(msg):
    now = datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M UTC')
    print(f"[{now}] {msg}")

def get_gbpusd_data():
    """Scarica dati GBP/USD H1 ultimi 5 giorni."""
    try:
        df = yf.download('GBPUSD=X', period='5d', interval='1h',
                        progress=False, auto_adjust=True)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')
        df['date'] = df.index.date
        log(f"Dati scaricati: {len(df)} barre, ultima: {df.index[-1]}")
        return df
    except Exception as e:
        log(f"Errore download: {e}")
        return None

def calc_daily_adx(df, period=14):
    """ADX su daily dai dati H1."""
    try:
        daily = df.groupby(df.index.date).agg(
            High=('High','max'), Low=('Low','min'), Close=('Close','last'))
        h,l,c = daily['High'],daily['Low'],daily['Close']
        pdm = h.diff().clip(lower=0); mdm = (-l.diff()).clip(lower=0)
        mask = pdm>mdm; pdm=pdm.where(mask,0); mdm=mdm.where(~mask,0)
        tr  = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period,adjust=False).mean()
        dip = 100*pdm.ewm(alpha=1/period,adjust=False).mean()/(atr+1e-10)
        dim = 100*mdm.ewm(alpha=1/period,adjust=False).mean()/(atr+1e-10)
        dx  = 100*(dip-dim).abs()/(dip+dim+1e-10)
        adx = dx.ewm(alpha=1/period,adjust=False).mean()
        today = str(datetime.now(pytz.UTC).date())
        return round(float(adx.get(today, adx.iloc[-1])), 1)
    except:
        return 20  # default se errore

# ═══════════════════════════════════════════════════════
# 🏙️ LOGICA LONDON BREAKOUT
# ═══════════════════════════════════════════════════════

def get_asian_range(df, today, cfg):
    """Calcola range sessione asiatica del giorno."""
    pip = cfg['pip']
    mask = (df['date'] == today) & \
           (df.index.hour >= cfg['asian_start']) & \
           (df.index.hour < cfg['asian_end'])
    asian = df[mask]
    if len(asian) < 2: return None
    high = float(asian['High'].max())
    low  = float(asian['Low'].min())
    rng  = round((high - low) / pip, 1)
    return {'high': high, 'low': low, 'range_pips': rng}

def detect_trap(df, today, asian_high, asian_low, cfg):
    """Cerca trappola nella sessione London."""
    pip        = cfg['pip']
    trap_break = cfg['trap_break_pips'] * pip
    max_ret    = cfg['trap_return_bars']

    mask = (df['date'] == today) & \
           (df.index.hour >= cfg['london_start']) & \
           (df.index.hour < cfg['london_trap'])
    london_bars = df[mask]
    if len(london_bars) < 2: return None

    for i in range(len(london_bars)):
        bar   = london_bars.iloc[i]
        high  = float(bar['High'])
        low   = float(bar['Low'])
        close = float(bar['Close'])

        # Bull Trap → SHORT
        if high > asian_high + trap_break:
            if close < asian_high:
                return {'type':'BULL_TRAP', 'direction':'SHORT', 'bar_idx': i}
            for j in range(1, max_ret+1):
                if i+j >= len(london_bars): break
                if float(london_bars.iloc[i+j]['Close']) < asian_high:
                    return {'type':'BULL_TRAP', 'direction':'SHORT', 'bar_idx': i+j}

        # Bear Trap → LONG
        if low < asian_low - trap_break:
            if close > asian_low:
                return {'type':'BEAR_TRAP', 'direction':'LONG', 'bar_idx': i}
            for j in range(1, max_ret+1):
                if i+j >= len(london_bars): break
                if float(london_bars.iloc[i+j]['Close']) > asian_low:
                    return {'type':'BEAR_TRAP', 'direction':'LONG', 'bar_idx': i+j}
    return None

def calc_levels(direction, entry_price, asian_high, asian_low, rng, cfg):
    """Calcola SL e TP."""
    pip    = cfg['pip']
    sl_buf = cfg['sl_buffer_pips'] * pip
    if direction == 'LONG':
        sl = asian_low  - sl_buf
        tp = entry_price + rng * pip * cfg['tp_range_mult']
    else:
        sl = asian_high + sl_buf
        tp = entry_price - rng * pip * cfg['tp_range_mult']
    sl_pips = round(abs(entry_price - sl) / pip, 1)
    tp_pips = round(abs(tp - entry_price) / pip, 1)
    return sl, tp, sl_pips, tp_pips

# ═══════════════════════════════════════════════════════
# 🔔 NOTIFICHE
# ═══════════════════════════════════════════════════════

def notify_range(asian, adx, day_name, is_news):
    if is_news:
        msg = (
            f"📅 <b>LONDON MONITOR — {day_name}</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"⚠️ <b>GIORNO NEWS — SKIP</b>\n"
            f"Nessun trade oggi per dati macro in uscita."
        )
    elif asian is None:
        msg = (
            f"📅 <b>LONDON MONITOR — {day_name}</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"❌ Range asiatico non disponibile"
        )
    elif not (CONFIG['min_range_pips'] <= asian['range_pips'] <= CONFIG['max_range_pips']):
        msg = (
            f"📅 <b>LONDON MONITOR — {day_name}</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"⛔ Range asiatico: <b>{asian['range_pips']} pips</b>\n"
            f"Range fuori zona ({CONFIG['min_range_pips']}-{CONFIG['max_range_pips']} pips)\n"
            f"Nessun trade oggi."
        )
    else:
        msg = (
            f"📅 <b>LONDON MONITOR — {day_name}</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"✅ Range asiatico: <b>{asian['range_pips']} pips</b> — VALIDO\n"
            f"High: <b>{asian['high']:.5f}</b>\n"
            f"Low:  <b>{asian['low']:.5f}</b>\n"
            f"ADX:  <b>{adx}</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"🔍 Monitoraggio trappola attivo (07:00-10:00 GMT)\n"
            f"Attendi segnale..."
        )
    send_telegram(msg)

def notify_trap(trap, entry_price, sl, tp, sl_pips, tp_pips, asian_range):
    icon = '🔴' if trap['direction'] == 'SHORT' else '🟢'
    rr   = round(tp_pips / (sl_pips + 1e-10), 1)
    msg  = (
        f"{icon} <b>TRAPPOLA RILEVATA — GBP/USD</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"Tipo: <b>{trap['type']}</b>\n"
        f"Direzione: <b>{trap['direction']}</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"Entrata: <b>{entry_price:.5f}</b>\n"
        f"TP: <b>{tp:.5f}</b> (+{tp_pips} pips)\n"
        f"SL: <b>{sl:.5f}</b> (-{sl_pips} pips)\n"
        f"R/R: <b>{rr}:1</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"Range asiatico: {asian_range} pips\n"
        f"⏰ Chiusura forzata: 13:00 GMT\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"⚠️ Solo uso informativo"
    )
    send_telegram(msg)

def notify_close_reminder():
    msg = (
        f"⏰ <b>REMINDER CHIUSURA — GBP/USD</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"Sono le 12:45 GMT.\n"
        f"Se hai una posizione aperta sul London Breakout\n"
        f"<b>chiudila entro le 13:00 GMT!</b>"
    )
    send_telegram(msg)

def notify_no_trade():
    msg = (
        f"📊 <b>LONDON MONITOR — Fine sessione</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"Nessuna trappola rilevata oggi.\n"
        f"Prossima sessione: domani."
    )
    send_telegram(msg)

# ═══════════════════════════════════════════════════════
# 🚀 MAIN — ESEGUITO OGNI ORA DA PYTHONANYWHERE
# ═══════════════════════════════════════════════════════

def main():
    now_utc  = datetime.now(pytz.UTC)
    hour_utc = now_utc.hour
    today    = now_utc.date()
    today_str = str(today)
    dow      = today.weekday()  # 0=Lun, 2=Mer, 3=Gio
    day_names = ['Lun','Mar','Mer','Gio','Ven','Sab','Dom']
    day_name  = day_names[dow]

    log(f"Esecuzione: {now_utc.strftime('%Y-%m-%d %H:%M UTC')} — {day_name}")

    # Carica stato
    state = load_state()

    # ── FUORI GIORNI OPERATIVI ──
    if dow not in CONFIG['allowed_days']:
        log(f"Giorno non operativo ({day_name}) — skip")
        return

    # ── CHECK NEWS DAY ──
    is_news = today_str in CONFIG['news_skip_days']

    # ── 07:00 GMT — NOTIFICA RANGE ASIATICO ──
    if hour_utc == 7 and not state['range_notified']:
        log("Ora 07:00 — calcolo range asiatico")
        df = get_gbpusd_data()
        if df is not None:
            adx   = calc_daily_adx(df)
            asian = get_asian_range(df, today, CONFIG)
            notify_range(asian, adx, day_name, is_news)

            if asian and not is_news and \
               CONFIG['min_range_pips'] <= asian['range_pips'] <= CONFIG['max_range_pips'] and \
               adx >= CONFIG['min_adx']:
                state['asian_high']       = asian['high']
                state['asian_low']        = asian['low']
                state['asian_range_pips'] = asian['range_pips']

        state['range_notified'] = True
        save_state(state)
        return

    # ── 07:00-10:00 GMT — MONITORAGGIO TRAPPOLA ──
    if 7 <= hour_utc < 10 and not state['trap_notified'] and \
       state['asian_high'] is not None and not is_news:
        log(f"Ora {hour_utc}:00 — ricerca trappola")
        df = get_gbpusd_data()
        if df is not None:
            trap = detect_trap(
                df, today,
                state['asian_high'], state['asian_low'],
                CONFIG
            )
            if trap:
                log(f"Trappola rilevata: {trap['type']} → {trap['direction']}")
                # Prezzo di entrata = ultimo prezzo disponibile
                mask = (df['date'] == today) & \
                       (df.index.hour >= CONFIG['london_start']) & \
                       (df.index.hour < CONFIG['london_trap'])
                london = df[mask]
                entry_price = float(london.iloc[-1]['Close'])

                sl, tp, sl_pips, tp_pips = calc_levels(
                    trap['direction'], entry_price,
                    state['asian_high'], state['asian_low'],
                    state['asian_range_pips'], CONFIG
                )

                notify_trap(trap, entry_price, sl, tp, sl_pips, tp_pips,
                           state['asian_range_pips'])

                state['trap_notified']   = True
                state['trade_active']    = True
                state['trade_direction'] = trap['direction']
                state['trade_entry']     = entry_price
                state['trade_sl']        = sl
                state['trade_tp']        = tp
                save_state(state)
            else:
                log("Nessuna trappola trovata ancora")
        return

    # ── 10:00 GMT — FINE FINESTRA, NESSUNA TRAPPOLA ──
    if hour_utc == 10 and not state['trap_notified'] and \
       state['asian_high'] is not None and not is_news:
        log("Ora 10:00 — fine finestra, nessuna trappola")
        notify_no_trade()
        state['trap_notified'] = True
        save_state(state)
        return

    # ── 12:45 GMT — REMINDER CHIUSURA ──
    if hour_utc == 12 and not state['close_notified'] and state['trade_active']:
        log("Ora 12:xx — reminder chiusura")
        notify_close_reminder()
        state['close_notified'] = True
        save_state(state)
        return

    log(f"Nessuna azione richiesta a {hour_utc}:00 UTC")

if __name__ == '__main__':
    main()
