#!/usr/bin/env python3
"""
Portfolio simulation — EMA9 fresh-breakout entry, EMA21 close exit.

Rules
-----
Entry signal (at a day's close): price crosses UP through the EMA9 AND is within
  ENTRY_BAND% above it (fresh, not extended). -> buy at the NEXT day's open.
Exit: the day price CLOSES below the EMA21 -> sell at that close.

Shared capital across up to MAX_POSITIONS names (equal-weight target). When more
signals fire than open slots/cash allow, the freshest (closest to EMA9) win.
Benchmarked against buy-and-hold SPY over the same window.

Outputs a terminal report, portfolio_trades.csv, and portfolio_sim.html
(equity curve vs SPY + stats).
"""

from __future__ import annotations

import csv
from datetime import datetime

import numpy as np
import pandas as pd

from ema_scanner import download, ema, get_universe

START_CAPITAL = 100_000.0
MAX_POSITIONS = 10
ENTRY_BAND = 1.0      # within this % above EMA9 on the signal day
COST = 0.0005         # slippage/fees per side (5 bps)
PERIOD = "5y"


def panels_from_data(tickers, data, entry_band=ENTRY_BAND, exit_ema=21,
                     regime=True):
    """Aligned date×ticker frames for open, close, exit-EMA, signal, signal-day %."""
    opens, closes, exits, sigs, pcts = {}, {}, {}, {}, {}
    for t in tickers:
        df = data.get(t)
        if df is None or df.empty:
            continue
        df = df.dropna(subset=["Open", "Close"])
        if len(df) < 60:
            continue
        c = df["Close"]
        e9 = ema(c, 9)
        e21, e33, e50, e200 = ema(c, 21), ema(c, 33), ema(c, 50), ema(c, 200)
        pct9 = (c - e9) / e9 * 100.0
        cross = (c > e9) & (c.shift(1) <= e9.shift(1))
        sig = cross & (pct9 >= 0) & (pct9 <= entry_band)
        if regime:   # only take the breakout when the daily stack is full bull
            sig = sig & (e9 > e21) & (e21 > e33) & (e33 > e50) & (e50 > e200)
        opens[t] = df["Open"]; closes[t] = c; exits[t] = ema(c, exit_ema)
        sigs[t] = sig; pcts[t] = pct9
    O = pd.DataFrame(opens).sort_index()
    C = pd.DataFrame(closes).reindex(O.index)
    E = pd.DataFrame(exits).reindex(O.index)
    S = pd.DataFrame(sigs).reindex(O.index).fillna(False).astype(bool)
    P = pd.DataFrame(pcts).reindex(O.index)
    return O, C, E, S, P


def spy_curve(index):
    d = download(["SPY"], PERIOD, "1d")["SPY"]
    if isinstance(d.columns, pd.MultiIndex):
        d = d.droplevel(0, axis=1)
    c = d["Close"].reindex(index).ffill()
    shares = START_CAPITAL / c.dropna().iloc[0]
    return (c * shares).values


def simulate(O, C, E, S, P, max_positions=MAX_POSITIONS):
    cols = list(O.columns)
    idx = O.index
    Ov, Cv, Ev, Sv, Pv = O.values, C.values, E.values, S.values, P.values
    n = len(idx)

    cash = START_CAPITAL
    positions = {}                  # col -> dict(shares, entry, entry_i)
    pending: list[tuple[int, float]] = []
    equity = np.empty(n)
    invested_days = 0
    trades = []

    for i in range(n):
        # 1) Execute yesterday's signals at today's open (freshest first)
        eq_now = cash + sum(p["shares"] * (Ov[i, c] if np.isfinite(Ov[i, c]) else p["entry"])
                            for c, p in positions.items())
        for col, _ in sorted(pending, key=lambda x: x[1]):
            if len(positions) >= max_positions or col in positions:
                continue
            px = Ov[i, col]
            if not np.isfinite(px) or px <= 0:
                continue
            target = eq_now / max_positions
            alloc = min(target, cash)
            shares = int(alloc / (px * (1 + COST)))
            if shares <= 0:
                continue
            cost = shares * px * (1 + COST)
            cash -= cost
            positions[col] = {"shares": shares, "entry": px, "entry_i": i}
        pending = []

        # 2) Exit at close any position that closed below its EMA21
        for col in list(positions):
            cl, e21 = Cv[i, col], Ev[i, col]
            if np.isfinite(cl) and np.isfinite(e21) and cl < e21:
                p = positions.pop(col)
                proceeds = p["shares"] * cl * (1 - COST)
                cash += proceeds
                ret = (cl * (1 - COST)) / (p["entry"] * (1 + COST)) - 1
                trades.append(dict(ticker=cols[col], entry_date=idx[p["entry_i"]].date(),
                                   exit_date=idx[i].date(), entry=p["entry"], exit=cl,
                                   ret=ret, pnl=proceeds - p["shares"] * p["entry"],
                                   hold=i - p["entry_i"]))

        # 3) Mark equity at close
        equity[i] = cash + sum(p["shares"] * Cv[i, c] for c, p in positions.items()
                               if np.isfinite(Cv[i, c]))
        if positions:
            invested_days += 1

        # 4) Today's fresh signals -> queue for tomorrow's open
        for col in np.where(Sv[i])[0]:
            if col not in positions:
                pending.append((int(col), Pv[i, col]))

    # close out anything still open at the last close (mark-to-market)
    for col, p in positions.items():
        cl = Cv[n - 1, col]
        if np.isfinite(cl):
            trades.append(dict(ticker=cols[col], entry_date=idx[p["entry_i"]].date(),
                               exit_date=idx[n - 1].date(), entry=p["entry"], exit=cl,
                               ret=cl / p["entry"] - 1, pnl=p["shares"] * (cl - p["entry"]),
                               hold=n - 1 - p["entry_i"], open=True))
    return equity, trades, invested_days / n * 100


def stats(equity, idx):
    start, end = equity[0] if equity[0] else START_CAPITAL, equity[-1]
    yrs = (idx[-1] - idx[0]).days / 365.25
    cagr = (end / START_CAPITAL) ** (1 / yrs) - 1 if yrs > 0 else 0
    peak = np.maximum.accumulate(equity)
    maxdd = float((equity / peak - 1).min())
    return end, (end / START_CAPITAL - 1), cagr, maxdd, yrs


def run_one(tickers, data, spy, *, entry_band, exit_ema, regime=True,
            max_positions=MAX_POSITIONS):
    O, C, E, S, P = panels_from_data(tickers, data, entry_band=entry_band,
                                     exit_ema=exit_ema, regime=regime)
    equity, trades, exposure = simulate(O, C, E, S, P, max_positions=max_positions)
    end, tot, cagr, maxdd, yrs = stats(equity, O.index)
    closed = [t for t in trades if not t.get("open")]
    rets = [t["ret"] for t in closed]
    wins = [r for r in rets if r > 0]
    holds = [t["hold"] for t in closed]
    return dict(
        index=O.index, equity=equity, trades=trades, exposure=exposure,
        end=end, tot=tot, cagr=cagr, maxdd=maxdd, yrs=yrs, n=len(closed),
        win=(len(wins) / len(rets) * 100 if rets else 0),
        avg=(np.mean(rets) * 100 if rets else 0),
        med=(np.median(rets) * 100 if rets else 0),
        hold=(np.mean(holds) if holds else 0),
    )


def main():
    print("Portfolio sim — EMA9 fresh-breakout entry, regime filter + exit variants")
    tickers = get_universe()
    print(f"  {len(tickers)} tickers. Downloading {PERIOD} daily ...")
    data = download(tickers, PERIOD, "1d")
    spy_index = panels_from_data(tickers, data, exit_ema=21)[0].index
    spy = spy_curve(spy_index)
    _, spy_tot, spy_cagr, spy_dd, _ = stats(spy, spy_index)

    # Capacity sweep: does giving the EMA50 exit more slots let it hold winners
    # AND still take new breakouts? EMA21 at 10 slots shown as the reference.
    configs = [
        ("EMA21 exit, 1% band, 10 slots", dict(entry_band=1.0, exit_ema=21, max_positions=10)),
        ("EMA50 exit, 1% band, 10 slots", dict(entry_band=1.0, exit_ema=50, max_positions=10)),
        ("EMA50 exit, 1% band, 20 slots", dict(entry_band=1.0, exit_ema=50, max_positions=20)),
        ("EMA50 exit, 1% band, 30 slots", dict(entry_band=1.0, exit_ema=50, max_positions=30)),
        ("EMA50 exit, 1% band, 50 slots", dict(entry_band=1.0, exit_ema=50, max_positions=50)),
        ("EMA50 exit, 3% band, 30 slots", dict(entry_band=3.0, exit_ema=50, max_positions=30)),
    ]
    hdr = f"{'Config':<32}{'Trades':>7}{'Win%':>6}{'Avg%':>7}{'Med%':>7}{'Hold':>6}{'MaxDD':>8}{'CAGR':>7}{'TotRet':>8}{'vsSPY':>8}"
    print("\n" + hdr); print("-" * len(hdr))
    results = {}
    for label, kw in configs:
        r = run_one(tickers, data, spy, **kw)
        results[label] = r
        print(f"{label:<32}{r['n']:>7,}{r['win']:>6.1f}{r['avg']:>+7.2f}{r['med']:>+7.2f}"
              f"{r['hold']:>6.1f}{r['maxdd']*100:>7.1f}%{r['cagr']*100:>+6.1f}%"
              f"{r['tot']*100:>+7.1f}%{(r['tot']-spy_tot)*100:>+7.1f}")
    print("-" * len(hdr))
    print(f"{'SPY buy & hold':<32}{'—':>7}{'—':>6}{'—':>7}{'—':>7}{'—':>6}"
          f"{spy_dd*100:>7.1f}%{spy_cagr*100:>+6.1f}%{spy_tot*100:>+7.1f}%{0:>+8.1f}")

    # Headline = the configuration with the best total return.
    best_label = max(results, key=lambda k: results[k]["tot"])
    b = results[best_label]
    print(f"\nHeadline (best total return): {best_label}")

    with open("portfolio_trades.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ticker", "entry_date", "exit_date", "entry",
                                          "exit", "ret", "pnl", "hold", "open"])
        w.writeheader()
        for t in b["trades"]:
            row = dict(t); row.setdefault("open", False)
            row["entry"] = f"{t['entry']:.2f}"; row["exit"] = f"{t['exit']:.2f}"
            row["ret"] = f"{t['ret']:.4f}"; row["pnl"] = f"{t['pnl']:.2f}"
            w.writerow(row)

    write_html(b["index"], b["equity"], spy, dict(
        end=b["end"], tot=b["tot"], cagr=b["cagr"], maxdd=b["maxdd"],
        spy_tot=spy_tot, spy_cagr=spy_cagr, spy_dd=spy_dd, yrs=b["yrs"],
        n=b["n"], win=b["win"], avg=b["avg"], hold=b["hold"], exposure=b["exposure"],
        title=best_label))
    print(f"✅ portfolio_sim.html + portfolio_trades.csv written (headline: {b['n']:,} trades).")
    return 0


def _path(vals, w, h, lo, hi):
    n = len(vals)
    pts = []
    for i, v in enumerate(vals):
        x = i / (n - 1) * w
        y = h - (v - lo) / (hi - lo) * h
        pts.append(f"{x:.1f},{y:.1f}")
    return "M" + " L".join(pts)


def write_html(idx, equity, spy, s):
    W, H = 1000, 360
    lo = min(equity.min(), np.nanmin(spy)) * 0.98
    hi = max(equity.max(), np.nanmax(spy)) * 1.02
    eq_path = _path(equity, W, H, lo, hi)
    spy_path = _path(np.nan_to_num(spy, nan=spy[np.isfinite(spy)][0]), W, H, lo, hi)
    beat = s["tot"] - s["spy_tot"]

    def stat(label, val, good=None):
        col = "var(--muted)" if good is None else ("var(--bull)" if good else "var(--bear)")
        return f'<div class="stat"><div class="sl">{label}</div><div class="sv" style="color:{col}">{val}</div></div>'

    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Portfolio Simulation</title>
<style>
  :root {{ --bg:#0d1117; --panel:#161b22; --border:#30363d; --text:#e6edf3; --muted:#8b949e;
          --bull:#2ea043; --bear:#f85149; --blue:#388bfd; }}
  body {{ margin:0; background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }}
  header {{ padding:28px 32px 8px; }} h1 {{ margin:0; font-size:22px; }}
  .sub {{ color:var(--muted); font-size:13px; margin-top:4px; }}
  .wrap {{ padding:20px 32px 40px; }}
  .stats {{ display:flex; flex-wrap:wrap; gap:14px; margin-bottom:20px; }}
  .stat {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:12px 16px; min-width:130px; }}
  .sl {{ color:var(--muted); font-size:12px; }} .sv {{ font-size:20px; font-weight:700; margin-top:4px; font-variant-numeric:tabular-nums; }}
  .chart {{ background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:18px; }}
  .legend {{ font-size:12px; color:var(--muted); margin-bottom:8px; }}
  .legend b.s {{ color:var(--blue); }} .legend b.k {{ color:var(--muted); }}
  footer {{ color:var(--muted); font-size:12px; padding:0 32px 30px; }}
</style></head><body>
<header><h1>📊 Portfolio Simulation — EMA9 breakout strategy</h1>
<div class="sub">{s.get('title', '')} · {idx[0].date()} → {idx[-1].date()} · {s['yrs']:.1f} years · ${START_CAPITAL:,.0f} start · max {MAX_POSITIONS} positions</div></header>
<div class="wrap">
  <div class="stats">
    {stat("Final equity", f"${s['end']:,.0f}")}
    {stat("Total return", f"{s['tot']*100:+.1f}%", s['tot']>0)}
    {stat("vs SPY", f"{beat*100:+.1f} pts", beat>0)}
    {stat("CAGR", f"{s['cagr']*100:+.1f}%", s['cagr']>0)}
    {stat("Max drawdown", f"{s['maxdd']*100:.1f}%", s['maxdd']>s['spy_dd'])}
    {stat("Win rate", f"{s['win']:.0f}%")}
    {stat("Avg/trade", f"{s['avg']:+.2f}%", s['avg']>0)}
    {stat("Avg hold", f"{s['hold']:.0f}d")}
    {stat("Time invested", f"{s['exposure']:.0f}%")}
    {stat("Trades", f"{s['n']:,}")}
  </div>
  <div class="chart">
    <div class="legend"><b class="s">━ Strategy</b> &nbsp; <b class="k">━ SPY buy &amp; hold</b> &nbsp;(both start ${START_CAPITAL:,.0f})</div>
    <svg viewBox="0 0 {W} {H}" width="100%" preserveAspectRatio="none">
      <path d="{spy_path}" fill="none" stroke="#8b949e" stroke-width="1.5" opacity="0.8"/>
      <path d="{eq_path}" fill="none" stroke="#388bfd" stroke-width="2"/>
    </svg>
  </div>
</div>
<footer>Backtest on current index members (survivorship-biased) · {COST*100*2:.2f}% round-trip cost · price-only (no dividends) · not investment advice.</footer>
</body></html>"""
    with open("portfolio_sim.html", "w") as f:
        f.write(html)


if __name__ == "__main__":
    raise SystemExit(main())
