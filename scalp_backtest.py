#!/usr/bin/env python3
"""
Short-term mean-reversion scalp backtest for big, liquid names.

Goal: deploy a fixed $ amount in one name, sell shortly after for a small gain
(~$100-200 on $25k = ~0.5%). This is the OPPOSITE archetype to the EMA-stack
trend strategy — here we want a high hit rate on small moves, so a small fixed
profit target is the right tool (it's what kills the trend system, but helps here).

Edge: buy short-term OVERSOLD dips inside an uptrend (Connors-style RSI-2), then
take a small profit fast. Tested with several exit rules and reported in DOLLARS
on a fixed position size, with the tail risk made explicit.
"""

from __future__ import annotations

import statistics as stats

import numpy as np
import pandas as pd

from ema_scanner import download

CAPITAL = 25_000          # $ deployed per trade
SLIP = 0.0004             # round-trip slippage/fees (~4 bps; mega-caps are liquid)
PERIOD = "5y"

# Names liquid enough to "dump 20-30k" into without moving the tape.
BIG_NAMES = [
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "GOOG", "GOOGL", "META",
    "AVGO", "TSLA", "AMD", "NFLX", "ADBE", "CRM", "ORCL", "QCOM", "INTC",
    "CSCO", "TXN", "MU", "AMAT", "JPM", "BAC", "V", "MA", "WMT", "COST",
    "HD", "DIS", "NKE", "PG", "KO", "PEP", "XOM", "CVX", "UNH", "LLY",
    "JNJ", "BA", "CAT", "GE", "BRK-B", "PLTR", "UBER",
]


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def simulate(ticker, df, *, rsi_buy=10.0, target=None, stop=0.02,
             max_hold=4, exit_above_sma5=False) -> list[dict]:
    """Buy next open after an oversold close in an uptrend; exit on the first of:
    profit target, stop, close>SMA5 (optional), or max_hold days."""
    if df is None or df.empty:
        return []
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if len(df) < 260:
        return []

    close = df["Close"]
    o = df["Open"].to_numpy(); h = df["High"].to_numpy()
    lo = df["Low"].to_numpy(); c = close.to_numpy()
    sma200 = close.rolling(200).mean().to_numpy()
    sma5 = close.rolling(5).mean().to_numpy()
    r2 = rsi(close, 2).to_numpy()

    trades = []; n = len(c); i = 201
    while i < n - 1:
        uptrend = c[i] > sma200[i]
        oversold = r2[i] < rsi_buy
        if uptrend and oversold:
            ei = i + 1
            entry = o[ei] * (1 + SLIP / 2)
            if not np.isfinite(entry) or entry <= 0:
                i += 1; continue
            shares = int(CAPITAL // entry)
            if shares == 0:
                i += 1; continue
            tgt = entry * (1 + target) if target else None
            stp = entry * (1 - stop)
            exit_idx = None; exit_px = None
            j = ei
            while j < n:
                # Stop checked first (conservative on a down day).
                if lo[j] <= stp:
                    exit_idx = j; exit_px = stp; break
                if tgt is not None and h[j] >= tgt:
                    exit_idx = j; exit_px = o[j] if o[j] >= tgt else tgt; break
                if exit_above_sma5 and c[j] > sma5[j]:
                    exit_idx = j; exit_px = c[j]; break
                if j - ei >= max_hold:
                    exit_idx = j; exit_px = c[j]; break
                j += 1
            if exit_idx is None:
                exit_idx = n - 1; exit_px = c[-1]
            exit_px *= (1 - SLIP / 2)
            pnl = shares * (exit_px - entry)
            trades.append(dict(ticker=ticker, ret=exit_px / entry - 1,
                               pnl=pnl, hold=exit_idx - ei))
            i = exit_idx + 1
        else:
            i += 1
    return trades


def summarize(label, trades):
    if not trades:
        print(f"{label:<34}  (no trades)"); return
    pnls = [t["pnl"] for t in trades]
    rets = [t["ret"] for t in trades]
    holds = [t["hold"] for t in trades]
    wins = [p for p in pnls if p > 0]
    worst = sorted(pnls)[:3]
    win_rate = len(wins) / len(pnls) * 100
    avg = stats.mean(pnls); med = stats.median(pnls); tot = sum(pnls)
    avg_win = stats.mean(wins) if wins else 0
    losses = [p for p in pnls if p <= 0]
    avg_loss = stats.mean(losses) if losses else 0
    print(f"{label:<34}{len(pnls):>7,}{win_rate:>7.1f}{avg:>+9.0f}{med:>+8.0f}"
          f"{avg_win:>+8.0f}{avg_loss:>+8.0f}{stats.mean(holds):>6.1f}{tot:>+12,.0f}"
          f"   worst: {worst[0]:+,.0f}/{worst[1]:+,.0f}")


def main():
    print(f"Mean-reversion scalp backtest — ${CAPITAL:,}/trade on {len(BIG_NAMES)} big names")
    print(f"Downloading {PERIOD} daily ...")
    data = download(BIG_NAMES, PERIOD, "1d")

    configs = [
        ("RSI2<10, +0.5% tgt, -2% stop, 4d", dict(rsi_buy=10, target=0.005, stop=0.02, max_hold=4)),
        ("RSI2<10, +0.75% tgt, -2% stop, 4d", dict(rsi_buy=10, target=0.0075, stop=0.02, max_hold=4)),
        ("RSI2<10, +1% tgt, -3% stop, 5d", dict(rsi_buy=10, target=0.01, stop=0.03, max_hold=5)),
        ("RSI2<5, +1% tgt, -3% stop, 5d", dict(rsi_buy=5, target=0.01, stop=0.03, max_hold=5)),
        ("RSI2<10, exit>SMA5, -3% stop, 5d", dict(rsi_buy=10, target=None, stop=0.03, max_hold=5, exit_above_sma5=True)),
    ]
    hdr = f"{'Strategy':<34}{'Trades':>7}{'Win%':>7}{'Avg$':>9}{'Med$':>8}{'AvgW$':>8}{'AvgL$':>8}{'Hold':>6}{'TotalP/L':>12}"
    print("\n" + hdr); print("-" * (len(hdr) + 24))
    for label, kw in configs:
        trades = []
        for t in BIG_NAMES:
            trades.extend(simulate(t, data.get(t), **kw))
        summarize(label, trades)
    print(f"\n${CAPITAL:,}/trade · {SLIP*100:.2f}% round-trip slippage · "
          f"entry next open after an oversold close above the 200-day.")


if __name__ == "__main__":
    raise SystemExit(main())
