#!/usr/bin/env python3
"""
Concentrated momentum-rotation portfolio — a different bet from the breakout sim.

Each month, rank the universe by trailing momentum, hold the TOP_N strongest
names, and size each position by its signal strength (momentum-weighted). Only
hold names in their own uptrend; switch to CASH when the broad market (SPY) is
below its 200-day. Benchmarked vs buy-and-hold SPY.

The thesis: the edge lives in a few big winners, so concentrate and overweight
them instead of equal-weighting 30 slots (which diluted the winners away).

Outputs a terminal comparison + momentum_sim.html (equity vs SPY).
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from ema_scanner import download, get_universe

START = 100_000.0
PERIOD = "5y"
COST = 0.0005          # per-side slippage/fees
SKIP = 21              # skip the most recent ~month (classic momentum)
REBAL = 21             # rebalance roughly monthly
MAX_WEIGHT = 0.40      # cap any single position


def spy_series():
    d = download(["SPY"], PERIOD, "1d")["SPY"]
    if isinstance(d.columns, pd.MultiIndex):
        d = d.droplevel(0, axis=1)
    return d["Close"]


def panels(tickers, data, lookback):
    closes = {}
    for t in tickers:
        df = data.get(t)
        if df is None or df.empty:
            continue
        c = df["Close"].dropna()
        if len(c) < lookback + SKIP + 220:
            continue
        closes[t] = c
    C = pd.DataFrame(closes).sort_index()
    mom = C.shift(SKIP) / C.shift(SKIP + lookback) - 1.0
    sma200 = C.rolling(200).mean()
    up = C > sma200
    return C, mom, up


def stats(equity, idx):
    end = equity[-1]
    yrs = (idx[-1] - idx[0]).days / 365.25
    cagr = (end / START) ** (1 / yrs) - 1 if yrs > 0 else 0
    peak = np.maximum.accumulate(equity)
    maxdd = float((equity / peak - 1).min())
    return end, end / START - 1, cagr, maxdd, yrs


def simulate(C, mom, up, spy_up, *, top_n, lookback, sizing="mom",
             market_regime=True, uptrend=True):
    cols = list(C.columns)
    idx = C.index
    Cv, Mv, Uv = C.values, mom.values, up.values
    n = len(idx)
    warmup = max(lookback + SKIP, 200) + 1

    cash = START
    pos: dict[int, int] = {}     # col -> shares
    equity = np.empty(n)
    invested = 0
    n_trades = 0
    rebal_eq = []                # equity at each rebalance, for period stats

    for i in range(n):
        is_rebal = i >= warmup and (i - warmup) % REBAL == 0
        if is_rebal:
            eq = cash + sum(s * Cv[i, c] for c, s in pos.items() if np.isfinite(Cv[i, c]))
            rebal_eq.append(eq)

            # Decide target holdings
            target: dict[int, int] = {}
            risk_on = (not market_regime) or bool(spy_up[i])
            if risk_on and eq > 0:
                m = Mv[i].copy()
                valid = np.isfinite(m) & (m > 0)
                if uptrend:
                    valid &= np.where(np.isfinite(Uv[i]), Uv[i].astype(bool), False)
                cand = np.where(valid)[0]
                if len(cand):
                    cand = cand[np.argsort(m[cand])[::-1]][:top_n]   # top_n by momentum
                    if sizing == "equal":
                        w = np.ones(len(cand))
                    elif sizing == "rank":
                        w = np.arange(len(cand), 0, -1).astype(float)
                    else:                                            # momentum-weighted
                        w = m[cand].copy()
                    w = w / w.sum()
                    w = np.minimum(w, MAX_WEIGHT)
                    w = w / w.sum()
                    for col, wi in zip(cand, w):
                        px = Cv[i, col]
                        if np.isfinite(px) and px > 0:
                            target[int(col)] = int((eq * wi) / px)

            # Rebalance: sells first (free cash), then buys
            for col in list(pos):
                tgt = target.get(col, 0)
                if tgt < pos[col]:
                    px = Cv[i, col]
                    if not np.isfinite(px):
                        continue
                    d = pos[col] - tgt
                    cash += d * px * (1 - COST); n_trades += 1
                    pos[col] = tgt
                    if pos[col] == 0:
                        del pos[col]
            for col, tgt in target.items():
                cur = pos.get(col, 0)
                if tgt > cur:
                    px = Cv[i, col]
                    if not np.isfinite(px) or px <= 0:
                        continue
                    buy = tgt - cur
                    cost = buy * px * (1 + COST)
                    if cost > cash:
                        buy = int(cash / (px * (1 + COST)))
                        cost = buy * px * (1 + COST)
                    if buy > 0:
                        cash -= cost; pos[col] = cur + buy; n_trades += 1

        equity[i] = cash + sum(s * Cv[i, c] for c, s in pos.items() if np.isfinite(Cv[i, c]))
        if pos:
            invested += 1

    return equity, n_trades, invested / n * 100, np.array(rebal_eq)


def run(tickers, data, spy_up_full, *, top_n, lookback, **kw):
    C, mom, up = panels(tickers, data, lookback)
    spy_up = spy_up_full.reindex(C.index).ffill().fillna(False).values
    equity, n_trades, exposure, rebal_eq = simulate(
        C, mom, up, spy_up, top_n=top_n, lookback=lookback, **kw)
    end, tot, cagr, maxdd, yrs = stats(equity, C.index)
    per = np.diff(rebal_eq) / rebal_eq[:-1] if len(rebal_eq) > 1 else np.array([0.0])
    return dict(index=C.index, equity=equity, end=end, tot=tot, cagr=cagr,
                maxdd=maxdd, yrs=yrs, trades=n_trades, exposure=exposure,
                win=(per > 0).mean() * 100, avg=per.mean() * 100)


def main():
    print("Momentum-rotation sim — concentrated top-N, signal-weighted")
    tickers = get_universe()
    print(f"  {len(tickers)} tickers. Downloading {PERIOD} daily ...")
    data = download(tickers, PERIOD, "1d")
    spy_c = spy_series()
    spy_up_full = spy_c > spy_c.rolling(200).mean()

    configs = [
        ("Top5  3mo mom, regime on",  dict(top_n=5,  lookback=63)),
        ("Top5  6mo mom, regime on",  dict(top_n=5,  lookback=126)),
        ("Top5  12mo mom, regime on", dict(top_n=5,  lookback=252)),
        ("Top5  6mo mom, regime OFF", dict(top_n=5,  lookback=126, market_regime=False)),
        ("Top5  6mo, EQUAL weight",   dict(top_n=5,  lookback=126, sizing="equal")),
        ("Top10 6mo mom, regime on",  dict(top_n=10, lookback=126)),
    ]
    # SPY benchmark over the longest panel index
    ref_idx = panels(tickers, data, 126)[0].index
    spy_eq = (spy_c.reindex(ref_idx).ffill() /
              spy_c.reindex(ref_idx).ffill().dropna().iloc[0] * START).values
    _, spy_tot, spy_cagr, spy_dd, _ = stats(spy_eq, ref_idx)

    hdr = f"{'Config':<28}{'Trades':>7}{'Win%':>6}{'Avg%':>7}{'MaxDD':>8}{'CAGR':>7}{'TotRet':>9}{'vsSPY':>8}"
    print("\n" + hdr); print("-" * len(hdr))
    results = {}
    for label, kw in configs:
        r = run(tickers, data, spy_up_full, **kw)
        results[label] = r
        print(f"{label:<28}{r['trades']:>7,}{r['win']:>6.0f}{r['avg']:>+7.2f}"
              f"{r['maxdd']*100:>7.1f}%{r['cagr']*100:>+6.1f}%{r['tot']*100:>+8.1f}%"
              f"{(r['tot']-spy_tot)*100:>+8.1f}")
    print("-" * len(hdr))
    print(f"{'SPY buy & hold':<28}{'—':>7}{'—':>6}{'—':>7}{spy_dd*100:>7.1f}%"
          f"{spy_cagr*100:>+6.1f}%{spy_tot*100:>+8.1f}%{0:>+8.1f}")

    best = max(results, key=lambda k: results[k]["tot"])
    print(f"\nHeadline (best total return): {best}")
    write_html(results[best], spy_eq, ref_idx, best, spy_tot, spy_cagr, spy_dd)
    print("✅ momentum_sim.html written.")
    return 0


def _path(vals, w, h, lo, hi):
    n = len(vals)
    pts = [f"{i/(n-1)*w:.1f},{h-(v-lo)/(hi-lo)*h:.1f}" for i, v in enumerate(vals)]
    return "M" + " L".join(pts)


def write_html(b, spy_eq, idx, title, spy_tot, spy_cagr, spy_dd):
    W, H = 1000, 360
    eq = b["equity"]
    spy = np.nan_to_num(spy_eq, nan=spy_eq[np.isfinite(spy_eq)][0])
    lo, hi = min(eq.min(), spy.min()) * 0.98, max(eq.max(), spy.max()) * 1.02
    beat = b["tot"] - spy_tot

    def stat(l, v, good=None):
        col = "var(--muted)" if good is None else ("var(--bull)" if good else "var(--bear)")
        return f'<div class="stat"><div class="sl">{l}</div><div class="sv" style="color:{col}">{v}</div></div>'

    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Momentum Rotation Sim</title>
<style>
  :root {{ --bg:#0d1117; --panel:#161b22; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --bull:#2ea043; --bear:#f85149; --blue:#388bfd; }}
  body {{ margin:0; background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif; }}
  header {{ padding:28px 32px 8px; }} h1 {{ margin:0; font-size:22px; }} .sub {{ color:var(--muted); font-size:13px; margin-top:4px; }}
  .wrap {{ padding:20px 32px 40px; }}
  .stats {{ display:flex; flex-wrap:wrap; gap:14px; margin-bottom:20px; }}
  .stat {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:12px 16px; min-width:130px; }}
  .sl {{ color:var(--muted); font-size:12px; }} .sv {{ font-size:20px; font-weight:700; margin-top:4px; font-variant-numeric:tabular-nums; }}
  .chart {{ background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:18px; }}
  .legend {{ font-size:12px; color:var(--muted); margin-bottom:8px; }} .legend b.s {{ color:var(--blue); }}
  footer {{ color:var(--muted); font-size:12px; padding:0 32px 30px; }}
</style></head><body>
<header><h1>🚀 Momentum Rotation — concentrated, signal-weighted</h1>
<div class="sub">{title} · {idx[0].date()} → {idx[-1].date()} · {b['yrs']:.1f}y · ${START:,.0f} start · monthly rebalance</div></header>
<div class="wrap">
  <div class="stats">
    {stat("Final equity", f"${b['end']:,.0f}")}
    {stat("Total return", f"{b['tot']*100:+.1f}%", b['tot']>0)}
    {stat("vs SPY", f"{beat*100:+.1f} pts", beat>0)}
    {stat("CAGR", f"{b['cagr']*100:+.1f}%", b['cagr']>spy_cagr)}
    {stat("Max drawdown", f"{b['maxdd']*100:.1f}%", b['maxdd']>spy_dd)}
    {stat("% up months", f"{b['win']:.0f}%")}
    {stat("Avg/month", f"{b['avg']:+.2f}%", b['avg']>0)}
    {stat("Time invested", f"{b['exposure']:.0f}%")}
    {stat("Rebalance trades", f"{b['trades']:,}")}
  </div>
  <div class="chart">
    <div class="legend"><b class="s">━ Momentum strategy</b> &nbsp; <b>━ SPY buy &amp; hold</b> &nbsp;(both start ${START:,.0f})</div>
    <svg viewBox="0 0 {W} {H}" width="100%" preserveAspectRatio="none">
      <path d="{_path(spy, W, H, lo, hi)}" fill="none" stroke="#8b949e" stroke-width="1.5" opacity="0.8"/>
      <path d="{_path(eq, W, H, lo, hi)}" fill="none" stroke="#388bfd" stroke-width="2"/>
    </svg>
  </div>
</div>
<footer>Current index members (survivorship-biased) · {COST*100*2:.2f}% round-trip cost · price-only · not investment advice.</footer>
</body></html>"""
    with open("momentum_sim.html", "w") as f:
        f.write(html)


if __name__ == "__main__":
    raise SystemExit(main())
