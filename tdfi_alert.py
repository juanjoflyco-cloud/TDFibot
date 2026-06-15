"""
TDFI Screener - BTC/USDT 1h
----------------------------
Calcula el indicador TDFI (Trend Direction & Force Index) sobre velas de 1h
de BTC/USDT (Binance) y envía una alerta a Telegram cuando el TDFI, EN VELA
CERRADA, pasa de zona NEUTRAL (-0.05 a +0.05) a zona VERDE (>+0.05) o
zona ROJA (<-0.05).

Diseñado para ejecutarse periódicamente (cada hora) vía GitHub Actions,
manteniendo el estado entre ejecuciones en state.json.
"""

import os
import json
import time
import requests
import numpy as np
import pandas as pd

# ── Configuración ───────────────────────────────────────────────────────────
SYMBOL   = "BTCUSDT"
INTERVAL = "1h"

# Parámetros TDFI (estándar, equivalentes a la versión TradingView/ThinkScript)
LOOKBACK = 13
MMA_LEN  = 13
SMMA_LEN = 13
N_POWER  = 3

ZONE_THRESHOLD = 0.05   # +-0.05 define la zona neutral
KLINES_LIMIT   = 200    # suficientes velas para el warm-up del TDFI (~39 + EMAs)

STATE_FILE = "state.json"

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

BINANCE_URL = "https://api.binance.com/api/v3/klines"


# ── Datos ────────────────────────────────────────────────────────────────────
def fetch_klines(symbol, interval, limit):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(BINANCE_URL, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "tbb", "tbq", "ignore"
    ])
    df["close"] = df["close"].astype(float)
    df["open_time"] = df["open_time"].astype(np.int64)
    df["close_time"] = df["close_time"].astype(np.int64)
    return df


# ── Indicador TDFI ───────────────────────────────────────────────────────────
def calc_tdfi(closes, lookback=LOOKBACK, mma_len=MMA_LEN, smma_len=SMMA_LEN, n=N_POWER):
    """
    mma        = EMA(close, mma_len)
    smma       = EMA(mma, smma_len)
    impetmma   = mma - mma[-1]
    impetsmma  = smma - smma[-1]
    divma      = |mma - smma|
    averimpet  = (impetmma + impetsmma) / 2
    tdf        = divma * averimpet ** n
    tdfi       = tdf / max(|tdf|, lookback * n)   -> rango aprox [-1, 1]
    """
    s = pd.Series(closes)
    mma = s.ewm(span=mma_len, adjust=False).mean()
    smma = mma.ewm(span=smma_len, adjust=False).mean()
    impetmma = mma.diff()
    impetsmma = smma.diff()
    divma = (mma - smma).abs()
    averimpet = (impetmma + impetsmma) / 2
    tdf = divma * (averimpet ** n)
    roll_max = tdf.abs().rolling(lookback * n).max()
    tdfi = tdf / roll_max.replace(0, np.nan)
    return tdfi


def get_zone(value):
    if value > ZONE_THRESHOLD:
        return "green"
    if value < -ZONE_THRESHOLD:
        return "red"
    return "neutral"


# ── Estado persistente ──────────────────────────────────────────────────────
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"last_close_time": None, "last_zone": None}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


# ── Telegram ─────────────────────────────────────────────────────────────────
def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️  TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID no configurados, no se envía alerta.")
        print(message)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    resp = requests.post(
        url,
        data={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"},
        timeout=15,
    )
    if not resp.ok:
        print("Error enviando a Telegram:", resp.status_code, resp.text)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    df = fetch_klines(SYMBOL, INTERVAL, KLINES_LIMIT)
    now_ms = int(time.time() * 1000)

    # Solo velas ya cerradas (close_time < ahora)
    closed = df[df["close_time"] < now_ms].reset_index(drop=True)
    if len(closed) < (MMA_LEN + SMMA_LEN + LOOKBACK * N_POWER + 5):
        print("No hay suficientes velas cerradas para calcular el TDFI todavía.")
        return

    tdfi = calc_tdfi(closed["close"].values)

    last_idx = len(closed) - 1
    last_value = tdfi.iloc[last_idx]
    last_close_time = int(closed["close_time"].iloc[last_idx])

    if pd.isna(last_value):
        print("TDFI todavía en periodo de warm-up.")
        return

    current_zone = get_zone(last_value)
    state = load_state()

    # Evitar procesar dos veces la misma vela
    if state.get("last_close_time") == last_close_time:
        print(f"Vela ya procesada (close_time={last_close_time}). "
              f"TDFI={last_value:.4f} zona={current_zone}")
        return

    previous_zone = state.get("last_zone")
    price = closed["close"].iloc[last_idx]

    print(f"{SYMBOL} {INTERVAL} | TDFI={last_value:.4f} | "
          f"zona anterior={previous_zone} -> zona actual={current_zone} | precio={price}")

    if previous_zone == "neutral" and current_zone == "green":
        send_telegram(
            f"🟢 <b>{SYMBOL}</b> ({INTERVAL})\n"
            f"TDFI ha pasado de zona <b>NEUTRAL</b> a zona <b>VERDE</b> (alcista)\n"
            f"TDFI: {last_value:.4f}\n"
            f"Precio: {price:,.2f}"
        )
    elif previous_zone == "neutral" and current_zone == "red":
        send_telegram(
            f"🔴 <b>{SYMBOL}</b> ({INTERVAL})\n"
            f"TDFI ha pasado de zona <b>NEUTRAL</b> a zona <b>ROJA</b> (bajista)\n"
            f"TDFI: {last_value:.4f}\n"
            f"Precio: {price:,.2f}"
        )

    state["last_close_time"] = last_close_time
    state["last_zone"] = current_zone
    save_state(state)


if __name__ == "__main__":
    main()
