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
RSI_DEEP = 5.0     # deeply oversold
RSI_OVERSOLD = 10.0
RSI_WATCH = 20.0
EARNINGS_WINDOW = 90   # look this far ahead for the next earnings date
EARNINGS_RISK = 10     # flag/⚠ if earnings fall within this many days
OUTPUT_HTML = "oversold_dashboard.html"

SIGNAL_RANK = {"DEEP OVERSOLD": 0, "OVERSOLD": 1, "WATCH": 2, "DOWNTREND": 3, "NO SIGNAL": 4}


@dataclass
class Row:
    ticker: str
    price: float
    rsi2: float
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
    r2 = float(rsi(close, 2).iloc[-1])
    above = price > sma_trend

    if not above:
        signal = "DOWNTREND"
    elif r2 < RSI_DEEP:
        signal = "DEEP OVERSOLD"
    elif r2 < RSI_OVERSOLD:
        signal = "OVERSOLD"
    elif r2 < RSI_WATCH:
        signal = "WATCH"
    else:
        signal = "NO SIGNAL"

    return Row(
        ticker=ticker, price=price, rsi2=r2, above_trend=above,
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


def render(rows: list[Row], generated: datetime) -> str:
    rows = sorted(rows, key=lambda r: (SIGNAL_RANK[r.signal], r.rsi2))
    actionable = sum(r.signal in ("DEEP OVERSOLD", "OVERSOLD") for r in rows)

    def badge(text, kind):
        return f'<span class="badge {kind}">{text}</span>'

    sig_kind = {"DEEP OVERSOLD": "deep", "OVERSOLD": "os", "WATCH": "watch",
                "DOWNTREND": "down", "NO SIGNAL": "mixed"}

    trs = []
    for r in rows:
        name = html.escape(TICKER_NAMES.get(r.ticker, ""))
        rsi_kind = "deep" if r.rsi2 < RSI_DEEP else "os" if r.rsi2 < RSI_OVERSOLD else "watch" if r.rsi2 < RSI_WATCH else "mixed"
        trend = badge(f"{r.pct_vs_trend:+.1f}%", "bull" if r.above_trend else "bear")
        trs.append(f"""
      <tr data-rank="{SIGNAL_RANK[r.signal]}">
        <td class="ticker">{r.ticker}<span class="coname">{name}</span></td>
        <td>{badge(r.signal, sig_kind[r.signal])}</td>
        <td class="num">${r.price:,.2f}</td>
        <td class="num">{badge(f"{r.rsi2:.0f}", rsi_kind)}</td>
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
</style></head><body>
<header>
  <h1>🩸 Oversold Scanner — S&amp;P 500 + Nasdaq 100</h1>
  <div class="sub-h">Mean-reversion dip-buy · Generated {generated:%Y-%m-%d %H:%M} · {len(rows)} names · {actionable} actionable setup(s)</div>
</header>
<div class="plan">
  <b>The plan (from the backtest):</b> buy the <b>next open</b> on a name flagged OVERSOLD/DEEP OVERSOLD ·
  exit when it <b>closes back above its {EXIT_SMA}-day SMA</b> (the "+% to mean" target) — <b>do not</b> cap at a tiny fixed $ target ·
  hard <b>−{STOP_PCT:.0f}% stop</b> · <span class="warn">never hold a single name through earnings — ⚠ flags reports within {EARNINGS_RISK}d.</span>
</div>
<div class="wrap">
  <table>
    <thead><tr>
      <th>Ticker</th><th>Signal</th><th>Price</th><th>RSI(2)</th>
      <th>vs 200-day</th><th>Bounce target ({EXIT_SMA}d SMA)</th><th>Stop</th><th>Next Earnings</th>
    </tr></thead>
    <tbody>{''.join(trs)}</tbody>
  </table>
</div>
<footer>RSI(2) &lt;{RSI_OVERSOLD:.0f} = oversold, &lt;{RSI_DEEP:.0f} = deep, in an uptrend (above 200-day).
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

    with open(OUTPUT_HTML, "w") as f:
        f.write(render(rows, datetime.now().astimezone()))
    n = sum(r.signal in ("DEEP OVERSOLD", "OVERSOLD") for r in rows)
    print(f"\n✅ {OUTPUT_HTML} written · {len(rows)} names · {n} actionable oversold setup(s) today.")
    for r in sorted(rows, key=lambda x: x.rsi2)[:10]:
        er = r.earnings_date or "—"
        print(f"   {r.ticker:6} RSI2 {r.rsi2:5.1f} | {r.signal:13} | ER {er}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
