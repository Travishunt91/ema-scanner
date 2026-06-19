#!/usr/bin/env python3
"""
Quarter-over-quarter trend check for "next NVIDIA" finalists.

Annual numbers hide lumpiness. This pulls the last ~5 quarters and shows:
  - Revenue per quarter
  - Sequential QoQ revenue growth (is each quarter > the last?)
  - YoY revenue growth per quarter (same quarter vs 4 ago, if available)
  - Gross & net margin per quarter (is leverage expanding sequentially?)

A real inflection = QoQ growth holding/rising AND net margin stair-stepping up,
quarter after quarter. One big jump = noise; a staircase = a trend.
"""
from __future__ import annotations
import sys, warnings
import numpy as np
import yfinance as yf

warnings.filterwarnings("ignore")

FINALISTS = sys.argv[1:] or ["PLTR", "MRVL", "PODD", "ADI", "MPWR"]


def series(af, names):
    for n in ([names] if isinstance(names, str) else names):
        if n in af.index:
            return af.loc[n]
    return None


def show(tk: str):
    t = yf.Ticker(tk)
    qf = t.quarterly_financials
    if qf is None or qf.empty:
        print(f"\n{tk}: no quarterly data")
        return
    cols = list(qf.columns)[::-1]          # oldest -> newest
    rev = series(qf, ["Total Revenue", "Operating Revenue"])
    gp = series(qf, "Gross Profit")
    ni = series(qf, "Net Income")
    if rev is None or ni is None:
        print(f"\n{tk}: missing rows")
        return

    print(f"\n=== {tk} ===")
    print(f'{"Qtr end":<12}{"Rev$M":>9}{"QoQ%":>8}{"GM%":>8}{"NM%":>8}')
    prev = None
    qoqs = []
    nms = []
    for c in cols:
        r = rev[c]
        if r is None or (isinstance(r, float) and np.isnan(r)):
            continue
        gm = (gp[c] / r * 100) if gp is not None and not np.isnan(gp[c]) else np.nan
        nm = (ni[c] / r * 100) if not np.isnan(ni[c]) else np.nan
        qoq = ((r / prev - 1) * 100) if prev else np.nan
        if not np.isnan(qoq):
            qoqs.append(qoq)
        if not np.isnan(nm):
            nms.append(nm)
        qoq_s = f"{qoq:+.1f}" if not np.isnan(qoq) else "   -"
        gm_s = f"{gm:.1f}" if not np.isnan(gm) else "  -"
        print(f'{str(c.date()):<12}{r/1e6:>9.0f}{qoq_s:>8}{gm_s:>8}{nm:>8.1f}')
        prev = r

    # verdict
    if len(qoqs) >= 3:
        rising = qoqs[-1] >= qoqs[0] and all(q > 0 for q in qoqs[-3:])
        accel = qoqs[-1] > np.mean(qoqs[:-1])
        nm_steps_up = len(nms) >= 3 and nms[-1] > nms[0]
        tag = []
        tag.append("QoQ all-positive & holding" if rising else "QoQ choppy/negative")
        tag.append("re-accelerating" if accel else "steady/cooling")
        tag.append("margin stair-stepping UP" if nm_steps_up else "margin flat/down")
        print("  -> " + " | ".join(tag))


for tk in FINALISTS:
    show(tk)
