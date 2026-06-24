#!/usr/bin/env python3
"""
Daily oversold scanner (mean-reversion dip-buy) — S&P 500 + Nasdaq 100.

Lists which names closed OVERSOLD inside an uptrend today — the setup the scalp
backtest found works: buy the next open, exit when price closes back above its
5-day SMA (let the bounce revert to the mean — do NOT cap at a tiny fixed target),
hard -3% stop, and never hold a single name through earnings.

Writes oversold_dashboard.html, sorted most-oversold first.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

import earnings_calendar as ec
from ema_scanner import TICKER_NAMES, download, get_universe
from scalp_backtest import rsi

EXIT_SMA = 5       # exit when close climbs back above this SMA (the "mean")
TREND_SMA = 200    # only buy dips above this (uptrend filter)
STOP_PCT = 3.0     # hard stop below entry
# Per-name oversold tiers on RSI(14) — same metric as the bounce gauge, and a
# better per-trade edge than RSI(2) in testing (+0.31% vs +0.15% net/trade).
RSI_DEEP = 20.0    # deeply oversold
RSI_OVERSOLD = 30.0
RSI_WATCH = 40.0
EARNINGS_WINDOW = 90   # look this far ahead for the next earnings date
EARNINGS_RISK = 10     # flag/⚠ if earnings fall within this many days
OUTPUT_HTML = "oversold_dashboard.html"
# Market bounce gauge — validated: oversold breadth (RSI14<30) has IC +0.13 with
# 1-4wk returns; calibrated bands from 5y of history (median ~2%).
BREADTH_RSI = 30
GAUGE_ELEVATED = 6.0   # % of universe oversold = elevated (≈80th pct)
GAUGE_EXTREME = 10.0   # ≈90th pct

SIGNAL_RANK = {"DEEP OVERSOLD": 0, "OVERSOLD": 1, "WATCH": 2, "DOWNTREND": 3, "NO SIGNAL": 4}


@dataclass
class Row:
    ticker: str
    price: float
    rsi14: float
    above_trend: bool
    pct_vs_trend: float
    bounce_target: float      # SMA5 price (exit level)
    pct_to_bounce: float      # % move from price up to SMA5
    stop: float
    signal: str
    earnings_date: str = ""   # next earnings date as 'YYYY-MM-DD' (or "")


def analyze(ticker: str, df: pd.DataFrame) -> Row | None:
    if df is None or df.empty:
        return None
    close = df["Close"].dropna()
    if len(close) < TREND_SMA + 5:
        return None
    price = float(close.iloc[-1])
    sma_trend = float(close.rolling(TREND_SMA).mean().iloc[-1])
    sma_exit = float(close.rolling(EXIT_SMA).mean().iloc[-1])
    r14 = float(rsi(close, 14).iloc[-1])
    above = price > sma_trend

    if not above:
        signal = "DOWNTREND"
    elif r14 < RSI_DEEP:
        signal = "DEEP OVERSOLD"
    elif r14 < RSI_OVERSOLD:
        signal = "OVERSOLD"
    elif r14 < RSI_WATCH:
        signal = "WATCH"
    else:
        signal = "NO SIGNAL"

    return Row(
        ticker=ticker, price=price, rsi14=r14, above_trend=above,
        pct_vs_trend=(price / sma_trend - 1) * 100,
        bounce_target=sma_exit, pct_to_bounce=(sma_exit / price - 1) * 100,
        stop=price * (1 - STOP_PCT / 100), signal=signal,
    )


def _fmt_earnings(iso: str) -> str:
    """Next-earnings cell: 'Jul 24 (12d)', amber + ⚠ if within the risk window."""
    if not iso:
        return '<span class="sub" style="display:inline">—</span>'
    d = date.fromisoformat(iso)
    days = (d - date.today()).days
    soon = 0 <= days <= EARNINGS_RISK
    warn = "⚠ " if soon else ""
    cls = "er" if soon else "muted-date"
    return f'<span class="{cls}">{warn}{d:%b %d}</span><span class="sub">in {days}d</span>'


def spy_regime() -> dict:
    """SPY vs its 200-day SMA — the uptrend gate for the bounce gauge."""
    try:
        d = download(["SPY"], "1y", "1d")["SPY"]
        if isinstance(d.columns, pd.MultiIndex):
            d = d.droplevel(0, axis=1)
        c = d["Close"].dropna()
        if len(c) < TREND_SMA:
            return {}
        price = float(c.iloc[-1]); sma = float(c.rolling(TREND_SMA).mean().iloc[-1])
        return dict(risk_on=price > sma, pct=(price / sma - 1) * 100)
    except Exception:
        return {}


def market_gauge(tickers, daily, regime) -> dict:
    """Daily market-state read: oversold breadth + participation + regime."""
    os_n = total = above200 = above50 = 0
    for t in tickers:
        df = daily.get(t)
        if df is None or df.empty:
            continue
        c = df["Close"].dropna()
        if len(c) < TREND_SMA:
            continue
        total += 1
        if float(rsi(c, 14).iloc[-1]) < BREADTH_RSI:
            os_n += 1
        price = float(c.iloc[-1])
        if price > float(c.rolling(TREND_SMA).mean().iloc[-1]):
            above200 += 1
        if price > float(c.rolling(50).mean().iloc[-1]):
            above50 += 1
    total = max(total, 1)
    pct_os = os_n / total * 100
    risk_on = bool(regime.get("risk_on", True)) if regime else True
    if not risk_on:
        state = "RISK-OFF"
    elif pct_os >= GAUGE_EXTREME:
        state = "EXTREME"
    elif pct_os >= GAUGE_ELEVATED:
        state = "ELEVATED"
    else:
        state = "NORMAL"
    return dict(pct_os=pct_os, pct200=above200 / total * 100,
                pct50=above50 / total * 100, risk_on=risk_on, state=state,
                spy_pct=(regime.get("pct") if regime else None))


def render_gauge(g: dict) -> str:
    if not g:
        return ""
    cls = {"EXTREME": "extreme", "ELEVATED": "elevated", "NORMAL": "normal",
           "RISK-OFF": "riskoff"}[g["state"]]
    read = {
        "EXTREME": "🟢 <b>Broad capitulation in an uptrend</b> — 1–4 week bounce odds are high. Strong window to deploy the dip-buys below.",
        "ELEVATED": "🟢 <b>Oversold breadth elevated in an uptrend</b> — bounce odds above average. Good window for the dip-buys below.",
        "NORMAL": "⚪ <b>Calm market</b> — few names oversold. Bounce odds near baseline; be selective and patient.",
        "RISK-OFF": "🔴 <b>SPY below its 200-day</b> — mean-reversion bounces are unreliable in downtrends. Stay defensive; the dip-buys below carry extra risk.",
    }[g["state"]]
    spy = f'SPY {g["spy_pct"]:+.1f}% vs 200-day' if g["spy_pct"] is not None else "SPY n/a"
    regcls = "good" if g["risk_on"] else "bad"
    return f"""
<div class="gauge {cls}">
  <div class="g-top">
    <span class="g-title">📊 Market Bounce Gauge</span>
    <span class="g-state {cls}">{g['state']}</span>
  </div>
  <div class="g-metrics">
    <div class="gm"><div class="gv">{g['pct_os']:.1f}%</div><div class="gl">names oversold (RSI&lt;30)<span>~2% typical · ≥{GAUGE_ELEVATED:.0f}% elevated · ≥{GAUGE_EXTREME:.0f}% extreme</span></div></div>
    <div class="gm"><div class="gv {regcls}">{spy}</div><div class="gl">market regime · risk-{"on" if g['risk_on'] else "off"}</div></div>
    <div class="gm"><div class="gv">{g['pct200']:.0f}%</div><div class="gl">above 200-day<span>long-term breadth</span></div></div>
    <div class="gm"><div class="gv">{g['pct50']:.0f}%</div><div class="gl">above 50-day<span>short-term breadth</span></div></div>
  </div>
  <div class="g-read">{read}</div>
  <div class="g-note">Validated on 5y: oversold breadth has IC +0.13 with 1–4 week returns; uptrend + high-breadth days returned ~2× baseline. Timing signal (when), not selection — pair with the names below (what).</div>
</div>"""


def render(rows: list[Row], generated: datetime, gauge: dict | None = None) -> str:
    rows = sorted(rows, key=lambda r: (SIGNAL_RANK[r.signal], r.rsi14))
    actionable = sum(r.signal in ("DEEP OVERSOLD", "OVERSOLD") for r in rows)

    def badge(text, kind):
        return f'<span class="badge {kind}">{text}</span>'

    sig_kind = {"DEEP OVERSOLD": "deep", "OVERSOLD": "os", "WATCH": "watch",
                "DOWNTREND": "down", "NO SIGNAL": "mixed"}

    trs = []
    for r in rows:
        name = html.escape(TICKER_NAMES.get(r.ticker, ""))
        rsi_kind = "deep" if r.rsi14 < RSI_DEEP else "os" if r.rsi14 < RSI_OVERSOLD else "watch" if r.rsi14 < RSI_WATCH else "mixed"
        trend = badge(f"{r.pct_vs_trend:+.1f}%", "bull" if r.above_trend else "bear")
        trs.append(f"""
      <tr data-rank="{SIGNAL_RANK[r.signal]}">
        <td class="ticker">{r.ticker}<span class="coname">{name}</span></td>
        <td>{badge(r.signal, sig_kind[r.signal])}</td>
        <td class="num">${r.price:,.2f}</td>
        <td class="num">{badge(f"{r.rsi14:.0f}", rsi_kind)}</td>
        <td class="num">{trend}</td>
        <td class="num">${r.bounce_target:,.2f}<span class="sub">+{r.pct_to_bounce:.2f}% to mean</span></td>
        <td class="num">${r.stop:,.2f}<span class="sub">-{STOP_PCT:.0f}%</span></td>
        <td class="num">{_fmt_earnings(r.earnings_date)}</td>
      </tr>""")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Oversold Big-Name Scanner</title>
<style>
  :root {{ --bg:#0d1117; --panel:#161b22; --border:#30363d; --text:#e6edf3;
          --muted:#8b949e; --bull:#2ea043; --bear:#f85149; --watch:#d29922; --mixed:#6e7681; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--text);
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }}
  header {{ padding:24px 32px; border-bottom:1px solid var(--border); }}
  h1 {{ margin:0 0 4px; font-size:22px; }}
  .sub-h {{ color:var(--muted); font-size:13px; }}
  .plan {{ margin:16px 32px 0; padding:14px 18px; background:var(--panel);
          border:1px solid var(--border); border-left:3px solid var(--bull); border-radius:8px;
          font-size:13px; line-height:1.6; }}
  .plan b {{ color:var(--text); }}
  .warn {{ color:var(--watch); }}
  .wrap {{ padding:20px 32px 40px; }}
  table {{ width:100%; border-collapse:collapse; background:var(--panel);
          border:1px solid var(--border); border-radius:10px; overflow:hidden; }}
  th,td {{ padding:10px 14px; text-align:left; font-size:13px; border-bottom:1px solid var(--border); }}
  th {{ color:var(--muted); font-weight:600; text-transform:uppercase; font-size:11px; letter-spacing:.5px; }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  tr:last-child td {{ border-bottom:none; }}
  tr[data-rank="0"] {{ background:rgba(46,160,67,.10); }}
  tr[data-rank="1"] {{ background:rgba(46,160,67,.05); }}
  td.ticker {{ font-weight:700; }}
  .coname {{ display:block; font-weight:400; color:var(--muted); font-size:11px; }}
  .sub {{ display:block; color:var(--muted); font-size:11px; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:6px; font-size:11px; font-weight:700; }}
  .badge.deep {{ background:var(--bull); color:#051b0c; }}
  .badge.os {{ background:rgba(46,160,67,.20); color:var(--bull); }}
  .badge.watch {{ background:rgba(210,153,34,.18); color:var(--watch); }}
  .badge.down,.badge.bear {{ background:rgba(248,81,73,.16); color:var(--bear); }}
  .badge.bull {{ background:rgba(46,160,67,.16); color:var(--bull); }}
  .badge.mixed {{ background:rgba(110,118,129,.18); color:var(--mixed); }}
  .er {{ color:var(--watch); font-size:12px; font-weight:700; }}
  .muted-date {{ color:var(--text); font-size:12px; }}
  footer {{ color:var(--muted); font-size:12px; padding:0 32px 32px; }}
  /* Market bounce gauge */
  .gauge {{ margin:16px 32px 0; border:1px solid var(--border); border-radius:12px;
           padding:16px 20px; background:var(--panel); border-left:4px solid var(--muted); }}
  .gauge.extreme, .gauge.elevated {{ border-left-color:var(--bull); background:linear-gradient(180deg,rgba(46,160,67,.08),var(--panel)); }}
  .gauge.normal {{ border-left-color:var(--muted); }}
  .gauge.riskoff {{ border-left-color:var(--bear); background:linear-gradient(180deg,rgba(248,81,73,.08),var(--panel)); }}
  .g-top {{ display:flex; align-items:center; gap:12px; margin-bottom:14px; }}
  .g-title {{ font-size:16px; font-weight:700; }}
  .g-state {{ font-size:13px; font-weight:800; padding:4px 12px; border-radius:6px; letter-spacing:.5px; }}
  .g-state.extreme {{ background:var(--bull); color:#051b0c; }}
  .g-state.elevated {{ background:rgba(46,160,67,.22); color:var(--bull); }}
  .g-state.normal {{ background:rgba(110,118,129,.20); color:var(--muted); }}
  .g-state.riskoff {{ background:var(--bear); color:#1a0606; }}
  .g-metrics {{ display:flex; gap:28px; flex-wrap:wrap; }}
  .gm {{ min-width:120px; }}
  .gv {{ font-size:22px; font-weight:700; font-variant-numeric:tabular-nums; }}
  .gv.good {{ color:var(--bull); }} .gv.bad {{ color:var(--bear); }}
  .gl {{ color:var(--muted); font-size:12px; margin-top:2px; }}
  .gl span {{ display:block; font-size:10px; opacity:.8; }}
  .g-read {{ margin-top:14px; font-size:14px; line-height:1.5; }}
  .g-note {{ margin-top:8px; color:var(--muted); font-size:11px; line-height:1.5; }}
</style></head><body>
<header>
  <h1>🩸 Oversold Scanner — S&amp;P 500 + Nasdaq 100</h1>
  <div class="sub-h">Mean-reversion dip-buy · Generated {generated:%Y-%m-%d %H:%M} · {len(rows)} names · {actionable} actionable setup(s)</div>
</header>
{render_gauge(gauge)}
<div class="plan">
  <b>The plan (from the backtest):</b> buy the <b>next open</b> on a name flagged OVERSOLD/DEEP OVERSOLD ·
  exit when it <b>closes back above its {EXIT_SMA}-day SMA</b> (the "+% to mean" target) — <b>do not</b> cap at a tiny fixed $ target ·
  hard <b>−{STOP_PCT:.0f}% stop</b> · <span class="warn">never hold a single name through earnings — ⚠ flags reports within {EARNINGS_RISK}d.</span>
</div>
<div class="wrap">
  <table>
    <thead><tr>
      <th>Ticker</th><th>Signal</th><th>Price</th><th>RSI(14)</th>
      <th>vs 200-day</th><th>Bounce target ({EXIT_SMA}d SMA)</th><th>Stop</th><th>Next Earnings</th>
    </tr></thead>
    <tbody>{''.join(trs)}</tbody>
  </table>
</div>
<footer>RSI(14) &lt;{RSI_OVERSOLD:.0f} = oversold, &lt;{RSI_DEEP:.0f} = deep, in an uptrend (above 200-day) — same metric as the gauge.
Backtested edge is thin and bull-market-biased — not investment advice.</footer>
</body></html>"""


def main(tickers: list[str] | None = None,
         daily: dict | None = None) -> int:
    """Build the oversold dashboard. `tickers`/`daily` may be supplied by a
    shared runner (run_daily.py) to avoid re-fetching universe and daily data."""
    print("Oversold scanner — building universe (S&P 500 + Nasdaq 100) ...")
    if tickers is None:
        tickers = get_universe()
    if daily is None:
        print(f"  {len(tickers)} tickers. Downloading 1y daily ...")
        daily = download(tickers, "1y", "1d")
    else:
        print(f"  {len(tickers)} tickers (reusing shared daily download).")
    data = daily
    rows = [r for t in tickers if (r := analyze(t, data.get(t)))]

    # Next earnings date from the dedicated Nasdaq calendar (one batched fetch).
    print(f"Fetching earnings calendar (next {EARNINGS_WINDOW}d) ...")
    emap = ec.upcoming_earnings(days=EARNINGS_WINDOW, tickers=tickers)
    for r in rows:
        r.earnings_date = emap.get(r.ticker, "")

    print("Computing market bounce gauge ...")
    gauge = market_gauge(tickers, daily, spy_regime())

    with open(OUTPUT_HTML, "w") as f:
        f.write(render(rows, datetime.now().astimezone(), gauge))
    n = sum(r.signal in ("DEEP OVERSOLD", "OVERSOLD") for r in rows)
    print(f"\n✅ {OUTPUT_HTML} written · {len(rows)} names · {n} actionable oversold setup(s) today.")
    print(f"   Gauge: {gauge['state']} · {gauge['pct_os']:.1f}% oversold · {gauge['pct200']:.0f}% above 200d")
    for r in sorted(rows, key=lambda x: x.rsi14)[:10]:
        er = r.earnings_date or "—"
        print(f"   {r.ticker:6} RSI14 {r.rsi14:5.1f} | {r.signal:13} | ER {er}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
