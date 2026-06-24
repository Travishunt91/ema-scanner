#!/usr/bin/env python3
"""
Validate the MARKET-TIMING edge: do oversold/stretched days precede bounces?

Selection (which name) showed ~0 edge at 1-4 weeks. Timing (when to buy) is the
effect that actually exists. This aggregates the cached factor panel to daily
market-state metrics (% oversold, breadth, SPY stretch) and checks whether those
metrics predict the next 1-4 weeks of EQUAL-WEIGHT universe / SPY return.

Uses score_panel.pkl (built by swing_score_lab.py) + a small SPY pull.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ema_scanner import download

PANEL = "score_panel.pkl"


def spearman(a, b) -> float:
    m = a.notna() & b.notna()
    return a[m].rank().corr(b[m].rank())


def buckets(daily, metric, fwd, n=5):
    q = pd.qcut(daily[metric], n, labels=False, duplicates="drop")
    return daily.groupby(q)[fwd].mean() * 100


def main():
    print("Market-timing lab — do oversold days precede bounces?")
    p = pd.read_pickle(PANEL)
    p["below9"] = p["neg_pct9"] > 0          # price below its EMA9
    p["os30"] = p["rsi14"] < 30              # oversold (RSI14<30)
    p["os20"] = p["rsi14"] < 20              # deeply oversold
    p["below200"] = p["dist200"] < 0         # below 200-day

    daily = p.groupby("date").agg(
        pct_below9=("below9", "mean"),
        pct_os30=("os30", "mean"),
        pct_os20=("os20", "mean"),
        pct_below200=("below200", "mean"),
        mean_rsi=("rsi14", "mean"),
        ew_fwd10=("fwd10", "mean"),
        ew_fwd20=("fwd20", "mean"),
    )

    # SPY stretch + regime + forward returns
    spy = download(["SPY"], "5y", "1d")["SPY"]
    if isinstance(spy.columns, pd.MultiIndex):
        spy = spy.droplevel(0, axis=1)
    sc = spy["Close"].dropna()
    sd = pd.DataFrame(index=sc.index)
    sd["spy_dist10"] = (sc / sc.rolling(10).mean() - 1) * 100
    sd["spy_dist20"] = (sc / sc.rolling(20).mean() - 1) * 100
    sd["spy_above200"] = (sc > sc.rolling(200).mean()).astype(int)
    sd["spy_fwd10"] = (sc.shift(-10) / sc - 1)
    sd["spy_fwd20"] = (sc.shift(-20) / sc - 1)
    daily = daily.join(sd, how="inner").dropna(subset=["ew_fwd10", "spy_fwd10"])
    print(f"  {len(daily):,} trading days\n")

    print("=" * 60)
    print("Time-series IC — metric vs next-2-week return")
    print("=" * 60)
    for m, tgt in [("pct_os30", "ew_fwd10"), ("pct_os20", "ew_fwd10"),
                   ("pct_below9", "ew_fwd10"), ("spy_dist10", "spy_fwd10"),
                   ("spy_dist20", "spy_fwd10")]:
        sign = "+ oversold" if "os" in m or "below" in m else "- stretch"
        print(f"  {m:>12} vs {tgt:<9}: IC = {spearman(daily[m], daily[tgt]):+.3f}")

    print("\n" + "=" * 60)
    print("Equal-weight fwd return by % OVERSOLD (RSI14<30) quintile")
    print("=" * 60)
    b10 = buckets(daily, "pct_os30", "ew_fwd10")
    b20 = buckets(daily, "pct_os30", "ew_fwd20")
    print(f"  {'quintile':>10} {'fwd10%':>9} {'fwd20%':>9}")
    for q in b10.index:
        lo = "least" if q == 0 else "most" if q == b10.index.max() else ""
        print(f"  {f'Q{q+1} {lo}':>10} {b10[q]:>+8.2f} {b20[q]:>+8.2f}")

    print("\n" + "=" * 60)
    print("SPY fwd return by SPY stretch-below-10-day quintile")
    print("=" * 60)
    s10 = buckets(daily, "spy_dist10", "spy_fwd10")
    s20 = buckets(daily, "spy_dist10", "spy_fwd20")
    print(f"  {'quintile':>14} {'fwd10%':>9} {'fwd20%':>9}")
    for q in s10.index:
        lab = "most stretched DOWN" if q == 0 else "most extended UP" if q == s10.index.max() else ""
        print(f"  {f'Q{q+1} {lab}':>14} {s10[q]:>+8.2f} {s20[q]:>+8.2f}")

    # The headline combo: uptrend regime + broad oversold
    print("\n" + "=" * 60)
    print("Conditional: SPY above 200-day, by oversold breadth")
    print("=" * 60)
    up = daily[daily["spy_above200"] == 1]
    hi = up[up["pct_os30"] >= up["pct_os30"].quantile(0.8)]
    lo = up[up["pct_os30"] <= up["pct_os30"].quantile(0.2)]
    base = daily
    print(f"  baseline (all days)              : fwd10 {base['ew_fwd10'].mean()*100:+.2f}%  fwd20 {base['ew_fwd20'].mean()*100:+.2f}%")
    print(f"  uptrend + LOW oversold breadth   : fwd10 {lo['ew_fwd10'].mean()*100:+.2f}%  fwd20 {lo['ew_fwd20'].mean()*100:+.2f}%")
    print(f"  uptrend + HIGH oversold breadth  : fwd10 {hi['ew_fwd10'].mean()*100:+.2f}%  fwd20 {hi['ew_fwd20'].mean()*100:+.2f}%   <- bounce setup")
    print(f"    (high-breadth days: {len(hi)} of {len(daily)})")
    print("\nNote: overlapping forward windows inflate IC significance; survivorship")
    print("inflates absolute levels. The SPREAD across buckets is the signal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
