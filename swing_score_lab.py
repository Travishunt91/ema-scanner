#!/usr/bin/env python3
"""
Engineer an OPTIMAL 1-4 week score (mean-reversion-within-trend) and validate it.

score_lab.py showed the trend score is anti-predictive at 1-4 weeks (reversal
dominates). Here we build candidate composites from the factors that were
actually positive — oversold, pullback depth, distance below recent high — using
within-day cross-sectional ranks, optionally gated to an uptrend, and measure
each candidate's IC and decile spread so we can pick the best for the dashboard.

Panel is cached to score_panel.pkl so candidate weights iterate without re-downloading.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from score_lab import HORIZONS, process
from ema_scanner import download, get_universe

CACHE = "score_panel.pkl"


def panel() -> pd.DataFrame:
    if os.path.exists(CACHE):
        print(f"  loading cached panel {CACHE}")
        return pd.read_pickle(CACHE)
    tickers = get_universe()
    print(f"  {len(tickers)} tickers. Downloading 5y daily ...")
    data = download(tickers, "5y", "1d")
    frames = [r for t in tickers if (r := process(t, data.get(t))) is not None]
    df = pd.concat(frames, ignore_index=True)
    df.to_pickle(CACHE)
    print(f"  built + cached {len(df):,} rows")
    return df


def spearman(a, b) -> float:
    m = a.notna() & b.notna()
    return a[m].rank().corr(b[m].rank())


def wr(df, series, asc=True):
    """Within-day cross-sectional percentile rank (0-1), oriented bullish."""
    s = series if asc else -series
    return s.groupby(df["date"]).rank(pct=True)


def decile_spread(comp, fwd):
    d = pd.qcut(comp, 10, labels=False, duplicates="drop")
    sub = pd.DataFrame({"d": d, "f": fwd}).dropna()
    top = sub[sub["d"] == sub["d"].max()]["f"].mean()
    bot = sub[sub["d"] == sub["d"].min()]["f"].mean()
    return (top - bot) * 100, top * 100, bot * 100


def main():
    print("Swing score lab — engineering an optimal 1-4 week score")
    df = panel()
    df = df.dropna(subset=["fwd10", "fwd20", "rsi14", "neg_pct9", "dist52",
                           "vol20", "mom3", "dist200"]).copy()
    print(f"  {len(df):,} usable rows\n")

    # Oriented within-day rank factors (higher rank = predicts higher fwd return)
    df["osc"] = wr(df, df["rsi14"], asc=False)      # more oversold
    df["pb"] = wr(df, df["neg_pct9"], asc=True)     # deeper below EMA9 (pullback)
    df["dd"] = wr(df, df["dist52"], asc=False)      # further below 52-wk high
    df["volr"] = wr(df, df["vol20"], asc=True)      # higher realized vol
    df["mrev"] = wr(df, df["mom3"], asc=False)      # 3-month reversal (recent loser)
    uptrend = df["dist200"] > 0                     # regime gate

    candidates = {
        "osc only":                 df["osc"],
        "mrev only":                df["mrev"],
        "0.5 osc + 0.5 pb":         0.5 * df["osc"] + 0.5 * df["pb"],
        "osc+pb+dd (1/3 each)":     (df["osc"] + df["pb"] + df["dd"]) / 3,
        "osc+pb+dd+volr":           0.3 * df["osc"] + 0.25 * df["pb"] + 0.25 * df["dd"] + 0.2 * df["volr"],
        "osc+pb+dd+mrev":           0.3 * df["osc"] + 0.25 * df["pb"] + 0.25 * df["dd"] + 0.2 * df["mrev"],
    }

    hdr = f"  {'candidate':<24}{'IC.fwd10':>10}{'IC.fwd20':>10}{'spread10':>10}{'spread20':>10}"
    print("=" * 64)
    print("A) FULL universe (ungated)")
    print("=" * 64)
    print(hdr)
    for name, comp in candidates.items():
        ic10, ic20 = spearman(comp, df["fwd10"]), spearman(comp, df["fwd20"])
        sp10 = decile_spread(comp, df["fwd10"])[0]
        sp20 = decile_spread(comp, df["fwd20"])[0]
        print(f"  {name:<24}{ic10:>+10.4f}{ic20:>+10.4f}{sp10:>+9.2f}%{sp20:>+9.2f}%")

    # Gated to uptrend: re-rank within the uptrend universe each day
    print("\n" + "=" * 64)
    print("B) GATED to uptrend (price > 200-day SMA), re-ranked within gate")
    print("=" * 64)
    g = df[uptrend].copy()
    g["osc"] = wr(g, g["rsi14"], asc=False)
    g["pb"] = wr(g, g["neg_pct9"], asc=True)
    g["dd"] = wr(g, g["dist52"], asc=False)
    g["volr"] = wr(g, g["vol20"], asc=True)
    gated = {
        "osc only":             g["osc"],
        "0.5 osc + 0.5 pb":     0.5 * g["osc"] + 0.5 * g["pb"],
        "osc+pb+dd (1/3)":      (g["osc"] + g["pb"] + g["dd"]) / 3,
        "osc+pb+dd+volr":       0.3 * g["osc"] + 0.25 * g["pb"] + 0.25 * g["dd"] + 0.2 * g["volr"],
    }
    print(hdr)
    for name, comp in gated.items():
        ic10, ic20 = spearman(comp, g["fwd10"]), spearman(comp, g["fwd20"])
        sp10 = decile_spread(comp, g["fwd10"])[0]
        sp20 = decile_spread(comp, g["fwd20"])[0]
        print(f"  {name:<24}{ic10:>+10.4f}{ic20:>+10.4f}{sp10:>+9.2f}%{sp20:>+9.2f}%")
    print(f"\n  (gated universe: {len(g):,} rows of {len(df):,})")
    print("\nIC = rank corr with forward return · spread = top-decile minus bottom-decile fwd return")
    print("Reminder: survivorship bias flatters the reversal/low-price side; treat as upper bound.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
