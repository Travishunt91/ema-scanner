#!/usr/bin/env python3
"""
Win-rate lab: which levers raise the hit rate of the EMA9-break / stacked-trend
strategy, and what they cost in expectancy.

Base trade: enter the day after a daily EMA9 break while the daily stack is bull;
exit on a daily close below the EMA50 (the best exit from backtest.py).

Levers tested
-------------
  PT x%      : profit target — bank the gain if price reaches entry*(1+x) intraday.
  SPY filter : only enter when SPY closed above its own 200-EMA on the signal day.
  Tight entry: only enter when the open is within N% of the EMA21 (not extended).

Each lever is reported with win rate AND net expectancy so the trade-off is explicit.
"""

from __future__ import annotations

import statistics as stats

import numpy as np
import pandas as pd

from ema_scanner import EMA_PERIODS, download, ema, get_universe
from backtest import weekly_bull_daily_aligned  # reuse

DAILY_PERIOD = "5y"
COST = 0.002  # round-trip cost for net expectancy


def spy_uptrend() -> pd.Series:
    """Boolean daily series: SPY close > its 200-EMA (broad-market uptrend)."""
    d = download(["SPY"], DAILY_PERIOD, "1d")["SPY"]
    if isinstance(d.columns, pd.MultiIndex):
        d = d.droplevel(0, axis=1)  # ('SPY','Close') -> 'Close'
    c = d["Close"].dropna()
    return c > ema(c, 200)


def simulate(ticker, df, *, profit_target=None, regime=None,
             require_weekly=False, max_ext21=None) -> list[dict]:
    """Stacked-trend EMA9-break entry; exit on close<EMA50 or an optional PT."""
    if df is None or df.empty:
        return []
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if len(df) < 260:
        return []

    close = df["Close"]
    open_ = df["Open"].to_numpy(); low = df["Low"].to_numpy()
    high = df["High"].to_numpy(); c = close.to_numpy()
    dates = close.index
    e = {p: ema(close, p).to_numpy() for p in EMA_PERIODS}

    stacked = (e[9] > e[21]) & (e[21] > e[33]) & (e[33] > e[50]) & (e[50] > e[200])
    valid = ~np.isnan(e[200])
    wbull = (weekly_bull_daily_aligned(close).to_numpy() if require_weekly
             else np.ones(len(c), dtype=bool))
    reg = (regime.reindex(dates, method="ffill").fillna(False).to_numpy()
           if regime is not None else np.ones(len(c), dtype=bool))

    trades = []; n = len(c); i = 1
    while i < n - 1:
        if (c[i] > e[9][i] and c[i - 1] <= e[9][i - 1]
                and stacked[i] and valid[i] and wbull[i] and reg[i]):
            ei = i + 1
            entry = open_[ei]
            if not np.isfinite(entry) or entry <= 0:
                i += 1; continue
            if max_ext21 is not None and (entry - e[21][ei]) / e[21][ei] > max_ext21:
                i += 1; continue  # too extended above EMA21
            target = entry * (1 + profit_target) if profit_target else None

            exit_idx = None; exit_price = None; mfe = entry
            j = ei
            while j < n:
                if high[j] > mfe:
                    mfe = high[j]
                if target is not None and high[j] >= target:
                    exit_idx = j
                    exit_price = open_[j] if open_[j] >= target else target
                    break
                if c[j] < e[50][j]:  # stop: close below EMA50
                    exit_idx = j; exit_price = c[j]
                    break
                j += 1
            if exit_idx is None:
                exit_idx = n - 1; exit_price = c[-1]; is_open = True
            else:
                is_open = False
            ret = exit_price / entry - 1
            trades.append(dict(ticker=ticker, ret=ret, hold=exit_idx - ei,
                               mfe=mfe / entry - 1, open=is_open))
            i = exit_idx + 1
        else:
            i += 1
    return trades


def simulate_scaleout(ticker, df, *, profit_target, scale_frac, regime=None):
    """Bank `scale_frac` of the position at the profit target, trail the rest to
    a close-below-EMA50 stop. Realized return blends the two legs."""
    if df is None or df.empty:
        return []
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if len(df) < 260:
        return []
    close = df["Close"]
    open_ = df["Open"].to_numpy(); high = df["High"].to_numpy(); c = close.to_numpy()
    dates = close.index
    e = {p: ema(close, p).to_numpy() for p in EMA_PERIODS}
    stacked = (e[9] > e[21]) & (e[21] > e[33]) & (e[33] > e[50]) & (e[50] > e[200])
    valid = ~np.isnan(e[200])
    reg = (regime.reindex(dates, method="ffill").fillna(False).to_numpy()
           if regime is not None else np.ones(len(c), dtype=bool))

    trades = []; n = len(c); i = 1
    while i < n - 1:
        if (c[i] > e[9][i] and c[i - 1] <= e[9][i - 1]
                and stacked[i] and valid[i] and reg[i]):
            ei = i + 1; entry = open_[ei]
            if not np.isfinite(entry) or entry <= 0:
                i += 1; continue
            target = entry * (1 + profit_target)
            banked = 0.0; frac_left = 1.0; mfe = entry; j = ei; exit_idx = None
            is_open = False
            while j < n:
                if high[j] > mfe:
                    mfe = high[j]
                # bank the scale-out leg the first time the target is reached
                if frac_left == 1.0 and high[j] >= target:
                    fill = open_[j] if open_[j] >= target else target
                    banked += scale_frac * (fill / entry - 1)
                    frac_left = 1.0 - scale_frac
                # stop the remainder on a close below EMA50
                if c[j] < e[50][j]:
                    banked += frac_left * (c[j] / entry - 1)
                    exit_idx = j; break
                j += 1
            if exit_idx is None:
                banked += frac_left * (c[-1] / entry - 1)
                exit_idx = n - 1; is_open = True
            trades.append(dict(ticker=ticker, ret=banked, hold=exit_idx - ei,
                               mfe=mfe / entry - 1, open=is_open))
            i = exit_idx + 1
        else:
            i += 1
    return trades


def stat_row(label, trades):
    closed = [t for t in trades if not t["open"]]
    if not closed:
        return {"label": label, "n": 0}
    rets = [t["ret"] for t in closed]
    wins = [r for r in rets if r > 0]; losses = [r for r in rets if r <= 0]
    srt = sorted(rets, reverse=True); tot = sum(rets)
    top1 = srt[: max(1, len(srt) // 100)]
    return {
        "label": label, "n": len(closed),
        "win": len(wins) / len(rets) * 100,
        "exp": stats.mean(rets) * 100,
        "exp_net": (stats.mean(rets) - COST) * 100,
        "med": stats.median(rets) * 100,
        "avg_win": (stats.mean(wins) if wins else 0) * 100,
        "avg_loss": (stats.mean(losses) if losses else 0) * 100,
        "hold": stats.mean([t["hold"] for t in closed]),
        "top1": (sum(top1) / tot * 100) if tot else float("nan"),
    }


def print_table(rows):
    hdr = f"{'Variant':<32}{'Trades':>8}{'Win%':>7}{'Exp':>7}{'Exp-net':>9}{'Med':>7}{'AvgW':>7}{'AvgL':>7}{'Hold':>6}{'Top1%':>7}"
    print("\n" + hdr); print("-" * len(hdr))
    for r in rows:
        if not r.get("n"):
            print(f"{r['label']:<32}  (no trades)"); continue
        print(f"{r['label']:<32}{r['n']:>8,}{r['win']:>7.1f}{r['exp']:>+7.2f}"
              f"{r['exp_net']:>+9.2f}{r['med']:>+7.2f}{r['avg_win']:>+7.2f}"
              f"{r['avg_loss']:>+7.2f}{r['hold']:>6.1f}{r['top1']:>6.0f}%")
    print(f"\nBase = EMA9-break / daily stack / exit on close<EMA50. "
          f"Exp-net after {COST*100:.1f}% cost. Top1% = profit share of best 1% of trades.")


def main():
    print("Win-rate lab — building universe ...")
    tickers = get_universe()
    print(f"  {len(tickers)} tickers. Downloading {DAILY_PERIOD} daily ...")
    data = download(tickers, DAILY_PERIOD, "1d")
    print("  Downloading SPY for the regime filter ...")
    spy = spy_uptrend()

    experiments = [
        ("Base (close<EMA50)",        dict()),
        ("+ SPY uptrend filter",      dict(regime=spy)),
        ("+ tight entry (<2% of E21)", dict(max_ext21=0.02)),
        ("+ PT 5%",                   dict(profit_target=0.05)),
        ("+ PT 8%",                   dict(profit_target=0.08)),
        ("+ PT 5% + SPY filter",      dict(profit_target=0.05, regime=spy)),
        ("+ PT 5% + SPY + tight",     dict(profit_target=0.05, regime=spy, max_ext21=0.02)),
        ("+ SPY + tight (no PT)",     dict(regime=spy, max_ext21=0.02)),
    ]
    rows = []
    for label, kw in experiments:
        trades = []
        for t in tickers:
            trades.extend(simulate(t, data.get(t), **kw))
        rows.append(stat_row(label, trades))
    print_table(rows)


if __name__ == "__main__":
    raise SystemExit(main())
