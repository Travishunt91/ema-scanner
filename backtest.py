#!/usr/bin/env python3
"""
Backtest: EMA9-break entry in a stacked-up trend, exit on the 21 EMA touch.

Rules
-----
Regime  : daily EMAs stacked bull on the signal day (9>21>33>50>200).
Signal  : close crosses UP through the daily EMA9 (close[t-1] <= EMA9, close[t] > EMA9).
Entry   : next day's OPEN (you act the morning after the break).
Exit    : first day price TOUCHES the 21 EMA (intraday low <= EMA21), filled at EMA21.
Position: one trade at a time per ticker (no overlap).

Two variants are reported:
  (A) Daily stack only
  (B) Daily stack AND weekly stack bull (the full strategy)

Outputs aggregate profit stats and writes every trade to backtest_trades.csv.
"""

from __future__ import annotations

import csv
import statistics as stats

import numpy as np
import pandas as pd

from ema_scanner import EMA_PERIODS, download, ema, get_universe

DAILY_PERIOD = "5y"   # long history for a meaningful sample
EMA_TREND = EMA_PERIODS  # [9, 21, 33, 50, 200]


def weekly_bull_daily_aligned(close: pd.Series) -> pd.Series:
    """Boolean series (indexed like `close`) — is the weekly stack full bull?"""
    wk = close.resample("W-FRI").last().dropna()
    if len(wk) < 5:
        return pd.Series(False, index=close.index)
    emas = {p: ema(wk, p) for p in EMA_TREND}
    df = pd.DataFrame(emas)
    bull = (
        (df[9] > df[21]) & (df[21] > df[33]) & (df[33] > df[50]) & (df[50] > df[200])
    )
    # Forward-fill the weekly verdict onto daily dates (knowledge as of prior close)
    return bull.reindex(close.index, method="ffill").fillna(False)


def _exit_hit(mode: str, j: int, c, low, e21, e50) -> bool:
    """Has the exit condition triggered on bar j?"""
    if mode == "touch21":
        return low[j] <= e21[j]            # first intraday touch of EMA21
    if mode == "close21":
        return c[j] < e21[j]               # first daily close below EMA21
    if mode == "close50":
        return c[j] < e50[j]               # trail EMA21, stop on close below EMA50
    raise ValueError(mode)


def backtest_ticker(ticker: str, df: pd.DataFrame, require_weekly: bool,
                    exit_mode: str = "touch21") -> list[dict]:
    if df is None or df.empty:
        return []
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if len(df) < 260:
        return []

    close = df["Close"]
    open_ = df["Open"].to_numpy()
    low = df["Low"].to_numpy()
    high = df["High"].to_numpy()

    e = {p: ema(close, p).to_numpy() for p in EMA_TREND}
    c = close.to_numpy()
    dates = close.index

    stacked = (e[9] > e[21]) & (e[21] > e[33]) & (e[33] > e[50]) & (e[50] > e[200])
    valid = ~np.isnan(e[200])

    if require_weekly:
        wbull = weekly_bull_daily_aligned(close).to_numpy()
    else:
        wbull = np.ones(len(c), dtype=bool)

    trades = []
    n = len(c)
    i = 1
    while i < n - 1:
        broke = c[i] > e[9][i] and c[i - 1] <= e[9][i - 1]
        if broke and stacked[i] and valid[i] and wbull[i]:
            entry_idx = i + 1
            entry_price = open_[entry_idx]
            if not np.isfinite(entry_price) or entry_price <= 0:
                i += 1
                continue
            # Walk forward to the exit condition.
            exit_idx = None
            mfe = entry_price  # max favorable (highest high) before exit
            j = entry_idx
            while j < n:
                if high[j] > mfe:
                    mfe = high[j]
                if _exit_hit(exit_mode, j, c, low, e[21], e[50]):
                    exit_idx = j
                    break
                j += 1
            if exit_idx is None:
                # still open at end of data — record but mark open
                exit_price = c[-1]
                ret = exit_price / entry_price - 1
                trades.append(
                    dict(ticker=ticker, entry_date=dates[entry_idx].date(),
                         exit_date=dates[-1].date(), entry=entry_price,
                         exit=exit_price, ret=ret, hold=(n - 1 - entry_idx),
                         mfe=mfe / entry_price - 1, open=True)
                )
                break
            # touch21 fills at the EMA level; close-based exits fill at the close.
            exit_price = e[21][exit_idx] if exit_mode == "touch21" else c[exit_idx]
            ret = exit_price / entry_price - 1
            trades.append(
                dict(ticker=ticker, entry_date=dates[entry_idx].date(),
                     exit_date=dates[exit_idx].date(), entry=entry_price,
                     exit=exit_price, ret=ret, hold=(exit_idx - entry_idx),
                     mfe=mfe / entry_price - 1, open=False)
            )
            i = exit_idx + 1  # no overlapping trades
        else:
            i += 1
    return trades


COST = 0.002  # assumed round-trip cost (commission + slippage) for net expectancy


def stat_row(label: str, trades: list[dict]) -> dict:
    closed = [t for t in trades if not t["open"]]
    if not closed:
        return {"label": label, "n": 0}
    rets = [t["ret"] for t in closed]
    holds = [t["hold"] for t in closed]
    mfes = [t["mfe"] for t in closed]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    # Tail dependence: share of total return-sum from the top 1% of trades
    srt = sorted(rets, reverse=True)
    top1 = srt[: max(1, len(srt) // 100)]
    tot = sum(rets)
    return {
        "label": label,
        "n": len(closed),
        "win": len(wins) / len(rets) * 100,
        "exp": stats.mean(rets) * 100,
        "exp_net": (stats.mean(rets) - COST) * 100,
        "med": stats.median(rets) * 100,
        "avg_win": (stats.mean(wins) if wins else 0) * 100,
        "avg_loss": (stats.mean(losses) if losses else 0) * 100,
        "hold": stats.mean(holds),
        "mfe": stats.mean(mfes) * 100,
        "best": max(rets) * 100,
        "top1": (sum(top1) / tot * 100) if tot else float("nan"),
    }


def print_comparison(rows: list[dict]) -> None:
    cols = [
        ("Variant", "label", "{:<34}"),
        ("Trades", "n", "{:>7,}"),
        ("Win%", "win", "{:>5.1f}"),
        ("Exp", "exp", "{:>+6.2f}"),
        ("Exp-net", "exp_net", "{:>+7.2f}"),
        ("Med", "med", "{:>+6.2f}"),
        ("AvgW", "avg_win", "{:>+6.2f}"),
        ("AvgL", "avg_loss", "{:>+6.2f}"),
        ("Hold", "hold", "{:>5.1f}"),
        ("MFE", "mfe", "{:>+5.2f}"),
        ("Top1%share", "top1", "{:>6.0f}%"),
    ]
    head = " ".join(f"{c[0]:>{max(len(c[0]),6)}}" if c[1] != 'label' else f"{c[0]:<34}" for c in cols)
    print("\n" + head)
    print("-" * len(head))
    for r in rows:
        if not r.get("n"):
            print(f"{r['label']:<34}  (no trades)")
            continue
        cells = []
        for _, key, fmt in cols:
            cells.append(fmt.format(r[key]))
        print(" ".join(cells))
    print("\nExp = avg % return/trade (gross) · Exp-net = after "
          f"{COST*100:.1f}% round-trip cost · MFE = avg peak gain before exit")
    print("Top1%share = share of total return-sum produced by the best 1% of trades "
          "(>100% means the rest net negative)")


def main() -> int:
    print("Backtest: EMA9-break entry in a stacked-up trend — exit comparison")
    print("Building universe ...")
    tickers = get_universe()
    print(f"  {len(tickers)} tickers. Downloading {DAILY_PERIOD} daily history ...")
    data = download(tickers, DAILY_PERIOD, "1d")

    # Run every (regime, exit) combo off the single download.
    combos = [
        ("A touch21  (daily stack)",          False, "touch21"),
        ("A close21  (daily stack)",          False, "close21"),
        ("A close50  (daily stack, trail)",   False, "close50"),
        ("B close21  (daily+weekly stack)",   True,  "close21"),
        ("B close50  (daily+weekly, trail)",  True,  "close50"),
    ]
    results = {}
    rows = []
    for label, req_wk, mode in combos:
        trades = []
        for t in tickers:
            trades.extend(backtest_ticker(t, data.get(t), req_wk, mode))
        results[label] = trades
        rows.append(stat_row(label, trades))

    print_comparison(rows)

    # Write the best (close50 daily-stack) trade log for inspection.
    best_label = "A close50  (daily stack, trail)"
    best = results[best_label]
    with open("backtest_trades.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["ticker", "entry_date", "exit_date", "entry", "exit",
                           "ret", "hold", "mfe", "open"]
        )
        w.writeheader()
        for tr in best:
            row = dict(tr)
            row["entry"] = f"{tr['entry']:.2f}"
            row["exit"] = f"{tr['exit']:.2f}"
            row["ret"] = f"{tr['ret']:.4f}"
            row["mfe"] = f"{tr['mfe']:.4f}"
            w.writerow(row)
    print(f"\nTrade log ({best_label.strip()}) written to backtest_trades.csv "
          f"({len(best):,} trades).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
