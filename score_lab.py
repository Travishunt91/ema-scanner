#!/usr/bin/env python3
"""
Score validation lab — does the EMA-scanner score predict 1-4 week returns?

Reconstructs every score component at every historical bar across the universe,
attaches forward returns at 1/2/3/4-week horizons, and measures:
  1. Decile spread — do high-score names actually outperform low-score names?
  2. Information Coefficient (Spearman) of the composite score, per horizon.
  3. IC of each CURRENT component — which carry signal, which are dead weight.
  4. IC of CANDIDATE factors we don't use yet — would they help?
  5. In-sample vs out-of-sample, to guard against overfitting.

This is research input for re-weighting compute_score(); it changes nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ema_scanner import download, ema, get_universe

PERIOD = "5y"
HORIZONS = [5, 10, 15, 20]   # trading days ≈ 1, 2, 3, 4 weeks
PROX_MAX_EXT = 8.0


def spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman rank correlation without scipy (Pearson on ranks)."""
    m = a.notna() & b.notna()
    return a[m].rank().corr(b[m].rank())


def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(50)


def bars_since_cross(close, e9):
    cross = (close > e9) & (close.shift(1) <= e9.shift(1))
    pos = np.where(cross.to_numpy(), np.arange(len(cross)), np.nan)
    last = pd.Series(pos, index=close.index).ffill()
    return pd.Series(np.arange(len(close)), index=close.index) - last


def process(ticker, df):
    c = df["Close"].dropna()
    if len(c) < 300:
        return None
    e9, e21, e33, e50, e200 = (ema(c, n) for n in (9, 21, 33, 50, 200))
    sma200 = c.rolling(200).mean()

    # ---- current score components ----
    d_pairs = ((e9 > e21).astype(int) + (e21 > e33).astype(int)
               + (e33 > e50).astype(int) + (e50 > e200).astype(int))
    pct9 = (c - e9) / e9 * 100.0
    above9 = c > e9
    prox = np.where(above9, np.clip(3 * (1 - pct9 / PROX_MAX_EXT), 0, 3), 0.0)
    bsc = bars_since_cross(c, e9)
    fresh = np.where(above9 & (bsc <= 1), 1.0, np.where(above9 & (bsc <= 5), 0.5, 0.0))

    # weekly stack + weekly>EMA9, aligned to daily
    wk = c.resample("W-FRI").last().dropna()
    if len(wk) < 40:
        return None
    w9, w21, w33, w50, w200 = (ema(wk, n) for n in (9, 21, 33, 50, 200))
    w_pairs_wk = ((w9 > w21).astype(int) + (w21 > w33).astype(int)
                  + (w33 > w50).astype(int) + (w50 > w200).astype(int))
    w_above_wk = (wk > w9).astype(int)
    w_pairs = w_pairs_wk.reindex(c.index, method="ffill")
    w_above = w_above_wk.reindex(c.index, method="ffill")

    score = (w_pairs / 4 * 3 + d_pairs / 4 * 2 + pd.Series(prox, index=c.index)
             + pd.Series(fresh, index=c.index) + w_above * 0.75).clip(upper=10)

    # ---- candidate factors (not currently scored) ----
    mom3 = c / c.shift(63) - 1
    mom6 = c / c.shift(126) - 1
    dist200 = (c - sma200) / sma200
    rsi14 = rsi(c, 14)
    dist52 = c / c.rolling(252).max() - 1          # closeness to 52-wk high (≤0)
    vol20 = c.pct_change().rolling(20).std()        # realized vol

    out = pd.DataFrame({
        "ticker": ticker, "date": c.index, "score": score,
        "w_pairs": w_pairs, "d_pairs": d_pairs, "prox": prox, "neg_pct9": -pct9,
        "fresh": fresh, "w_above": w_above,
        "mom3": mom3, "mom6": mom6, "dist200": dist200, "rsi14": rsi14,
        "dist52": dist52, "vol20": vol20,
    })
    for h in HORIZONS:
        out[f"fwd{h}"] = (c.shift(-h) / c - 1).to_numpy()
    return out


def main():
    print("Score lab — validating the EMA score vs forward returns")
    tickers = get_universe()
    print(f"  {len(tickers)} tickers. Downloading {PERIOD} daily ...")
    data = download(tickers, PERIOD, "1d")

    frames = [r for t in tickers if (r := process(t, data.get(t))) is not None]
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["score", f"fwd{HORIZONS[-1]}"])
    print(f"  {len(df):,} ticker-day observations\n")

    comps = ["w_pairs", "d_pairs", "prox", "neg_pct9", "fresh", "w_above"]
    cands = ["mom3", "mom6", "dist200", "rsi14", "dist52", "vol20"]

    # 1) Composite score decile spread (forward return by score bucket)
    print("=" * 64)
    print("1) Forward return by SCORE bucket (does higher = better?)")
    print("=" * 64)
    df["bucket"] = pd.cut(df["score"], bins=[0, 2, 4, 5, 6, 7, 8, 9, 10],
                          include_lowest=True)
    g = df.groupby("bucket", observed=True)
    print(f"  {'score':>9} {'n':>8} {'fwd5%':>8} {'fwd10%':>8} {'fwd15%':>8} {'fwd20%':>8}")
    for b, sub in g:
        print(f"  {str(b):>9} {len(sub):>8,} "
              + " ".join(f"{sub[f'fwd{h}'].mean()*100:>7.2f}" for h in HORIZONS))

    # 2) IC of composite score per horizon
    print("\n" + "=" * 64)
    print("2) Information Coefficient (Spearman) — composite score")
    print("=" * 64)
    for h in HORIZONS:
        ic = spearman(df["score"], df[f"fwd{h}"])
        print(f"  fwd{h:>2}d: IC = {ic:+.4f}")

    # 3) IC of each current component (vs 2-week and 4-week)
    print("\n" + "=" * 64)
    print("3) IC of each CURRENT component (Spearman)")
    print("=" * 64)
    print(f"  {'component':>10} {'IC.fwd10':>10} {'IC.fwd20':>10}")
    for col in comps:
        ic10 = spearman(df[col], df["fwd10"])
        ic20 = spearman(df[col], df["fwd20"])
        print(f"  {col:>10} {ic10:>+10.4f} {ic20:>+10.4f}")

    # 4) IC of candidate factors
    print("\n" + "=" * 64)
    print("4) IC of CANDIDATE factors not currently scored")
    print("=" * 64)
    print(f"  {'candidate':>10} {'IC.fwd10':>10} {'IC.fwd20':>10}")
    for col in cands:
        ic10 = spearman(df[col], df["fwd10"])
        ic20 = spearman(df[col], df["fwd20"])
        print(f"  {col:>10} {ic10:>+10.4f} {ic20:>+10.4f}")

    # 5) In-sample vs out-of-sample IC (composite, fwd10)
    print("\n" + "=" * 64)
    print("5) Robustness — composite IC, in-sample vs out-of-sample (fwd10)")
    print("=" * 64)
    cut = df["date"].quantile(0.6)
    ins, oos = df[df["date"] <= cut], df[df["date"] > cut]
    print(f"  in-sample  (≤{cut.date()}): IC = {spearman(ins['score'], ins['fwd10']):+.4f}  ({len(ins):,} obs)")
    print(f"  out-sample (>{cut.date()}): IC = {spearman(oos['score'], oos['fwd10']):+.4f}  ({len(oos):,} obs)")

    print("\nNote: current index members (survivorship bias inflates absolute")
    print("forward returns; relative ranking by score is far less affected).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
