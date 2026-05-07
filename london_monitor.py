#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════
# 🏙️ LONDON BREAKOUT MONITOR — GBP/USD
# Per GitHub Actions — task schedulato ogni ora
# Nessun file di stato — ogni run è indipendente
# ═══════════════════════════════════════════════════════

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import pytz

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

    'news_skip_days': [
        # Fed 2025
        '2025-01-29','2025-03-19','2025-05-07','2025-06-18',
        '2025-07-30','2025-09-17','2025-11-05','2025-12-17',
        # NFP 2025
        '2025-01-10','2025-02-07','2025-03-07','2025-04-04',
        '2025-05-02','2025-06-06','2025-07-03','2025-08-01',
        '2025-09-05','2025-10-03','2025-11-07','2025-12-05',
        # BOE 2025
        '2025-02-06','2025-03-20','2025-05-08','2025-06-19',
        '2025-08-07','2025-09-18','2025-11-06','2025-12-18',
        # Fed 2026
        '2026-01-28','2026-03-18','2026-05-06','2026-05-07',
        '2026-06-17','2026-07-29','2026-09-16','2026-11-04','2026-12-16',
        # NFP 2026
        '2026-01-09','2026-02-06','2026-03-06','2026-04-03',
        '2026-05-01','2026-06-05','2026-07-02','2026-08-07',
        '2026-09-04','2026-10-02','2026-11-06','2026-12-04',
        # BOE 2026
        '2026-02-05','2026-03-19','2026-05-07','2026-06-18',
        '2026-08-06','2026-09-17','2026-11-05','2026-12-17',
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
        log(f"Telegram: {resp.status_code}")
        return resp.status_code == 200
    except Exception as e:
        log(f"Telegram error: {e}")
        return False

def log(msg):
    now = datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M UTC')
    print(f"[{now}] {msg}")

# ═══════════════════════════════════════════════════════
# 📊 DATI
# ═══════════════════════════════════════════════════════

def get_gbpusd_data():
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
        log(f"Dati: {len(df)} barre, ultima: {df.index[-1]}")
        return df
    except Exception as e:
        log(f"Errore download: {e}")
        return None

def calc_daily_adx(df, period=14):
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
        val = adx.get(today, adx.iloc[-1])
        return round(float(val), 1)
    except:
        return 20

# ═══════════════════════════════════════════════════════
# 🏙️ LOGICA
# ═══════════════════════════════════════════════════════

def get_asian_range(df, today, cfg):
    pip  = cfg['pip']
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
                return {'type':'BULL_TRAP','direction':'SHORT','bar_idx':i}
            for j in range(1, max_ret+1):
                if i+j >= len(london_bars): break
                if float(london_bars.iloc[i+j]['Close']) < asian_high:
                    return {'type':'BULL_TRAP','direction':'SHORT','bar_idx':i+j}

        # Bear Trap → LONG
        if low < asian_low - trap_break:
            if close > asian_low:
                return {'type':'BEAR_TRAP','direction':'LONG','bar_idx':i}
            for j in range(1, max_ret+1):
                if i+j >= len(london_bars): break
                if float(london_bars.iloc[i+j]['Close']) > asian_low:
                    return {'type':'BEAR_TRAP','direction':'LONG','bar_idx':i+j}
    return None

# ═══════════════════════════════════════════════════════
# 🚀 MAIN
# ═══════════════════════════════════════════════════════

def main():
    now_utc   = datetime.now(pytz.UTC)
    hour_utc  = now_utc.hour
    today     = now_utc.date()
    today_str = str(today)
    dow       = today.weekday()
    day_names = ['Lun','Mar','Mer','Gio','Ven','Sab','Dom']
    day_name  = day_names[dow]

    log(f"Esecuzione: {now_utc.strftime('%Y-%m-%d %H:%M UTC')} — {day_name}")

    # Fuori giorni operativi
    if dow not in CONFIG['allowed_days']:
        log(f"Giorno non operativo ({day_name}) — skip")
        return

    # Check news day
    is_news = today_str in CONFIG['news_skip_days']
    if is_news:
        log(f"Giorno news — skip")
        if hour_utc == 7:
            send_telegram(
                f"📅 <b>LONDON MONITOR — {day_name}</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"⚠️ <b>GIORNO NEWS — NESSUN TRADE</b>\n"
                f"Dati macro importanti oggi."
            )
        return

    # Scarica dati
    df = get_gbpusd_data()
    if df is None:
        log("Dati non disponibili")
        return

    # Range asiatico
    asian = get_asian_range(df, today, CONFIG)
    adx   = calc_daily_adx(df)

    log(f"Range asiatico: {asian['range_pips'] if asian else 'N/A'} pips | ADX: {adx}")

    # ── ORA 07:00 — NOTIFICA RANGE ──
    if hour_utc == 7:
        if asian is None:
            send_telegram(
                f"📅 <b>LONDON MONITOR — {day_name}</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"❌ Dati asiatici non disponibili"
            )
        elif not (CONFIG['min_range_pips'] <= asian['range_pips'] <= CONFIG['max_range_pips']):
            send_telegram(
                f"📅 <b>LONDON MONITOR — {day_name}</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"⛔ Range asiatico: <b>{asian['range_pips']} pips</b>\n"
                f"Fuori zona ({CONFIG['min_range_pips']}-{CONFIG['max_range_pips']} pips)\n"
                f"Nessun trade oggi."
            )
        else:
            send_telegram(
                f"📅 <b>LONDON MONITOR — {day_name}</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"✅ Range asiatico: <b>{asian['range_pips']} pips</b> — VALIDO\n"
                f"High: <b>{asian['high']:.5f}</b>\n"
                f"Low:  <b>{asian['low']:.5f}</b>\n"
                f"ADX:  <b>{adx}</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"🔍 Monitoraggio trappola attivo\n"
                f"Prossimo check: 08:00, 09:00, 10:00 GMT"
            )
        return

    # ── ORE 08:00-10:00 — RICERCA TRAPPOLA ──
    if 8 <= hour_utc <= 10:
        if asian is None or \
           not (CONFIG['min_range_pips'] <= asian['range_pips'] <= CONFIG['max_range_pips']) or \
           adx < CONFIG['min_adx']:
            log("Condizioni non valide per trappola")
            return

        trap = detect_trap(df, today, asian['high'], asian['low'], CONFIG)

        if trap:
            log(f"Trappola: {trap['type']} → {trap['direction']}")
            pip = CONFIG['pip']

            # Prezzo entrata
            mask = (df['date'] == today) & \
                   (df.index.hour >= CONFIG['london_start']) & \
                   (df.index.hour < CONFIG['london_trap'])
            london = df[mask]
            entry_price = float(london.iloc[-1]['Close'])

            # SL / TP
            sl_buf = CONFIG['sl_buffer_pips'] * pip
            rng    = asian['range_pips']
            if trap['direction'] == 'LONG':
                sl = asian['low']  - sl_buf
                tp = entry_price + rng * pip * CONFIG['tp_range_mult']
            else:
                sl = asian['high'] + sl_buf
                tp = entry_price - rng * pip * CONFIG['tp_range_mult']

            sl_pips = round(abs(entry_price - sl) / pip, 1)
            tp_pips = round(abs(tp - entry_price) / pip, 1)
            rr      = round(tp_pips / (sl_pips + 1e-10), 1)
            icon    = '🔴' if trap['direction'] == 'SHORT' else '🟢'

            send_telegram(
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
                f"Range asiatico: {rng} pips\n"
                f"⏰ Chiudi entro 13:00 GMT\n"
                f"⚠️ Solo uso informativo"
            )
        else:
            log(f"Nessuna trappola trovata a {hour_utc}:00")
            if hour_utc == 10:
                send_telegram(
                    f"📊 <b>LONDON MONITOR — Fine sessione</b>\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"Nessuna trappola rilevata oggi ({day_name}).\n"
                    f"Range asiatico: {asian['range_pips']} pips"
                )
        return

    # ── ORA 12:00 — REMINDER CHIUSURA ──
    if hour_utc == 12:
        send_telegram(
            f"⏰ <b>REMINDER CHIUSURA — GBP/USD</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"Sono le 12:00 GMT.\n"
            f"Se hai una posizione aperta\n"
            f"<b>chiudila entro le 13:00 GMT!</b>"
        )
        return

    log(f"Nessuna azione a {hour_utc}:00 UTC")

if __name__ == '__main__':
    main()

