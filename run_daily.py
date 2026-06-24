#!/usr/bin/env python3
"""
Daily market scan — one command, both dashboards.

Runs:
  1. ema_scanner       -> ema_dashboard.html       (trend / EMA-stack setups)
  2. oversold_scanner  -> oversold_dashboard.html  (mean-reversion dip-buys)

and writes index.html linking both. Schedule this after the close (4pm ET).
"""

from __future__ import annotations

from datetime import datetime

import bottleneck_scanner
import ema_scanner
import momentum_scanner
import next_nvidia
import oversold_scanner


def write_index(generated: datetime) -> None:
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Daily Market Scan</title>
<style>
  :root {{ --bg:#0d1117; --panel:#161b22; --border:#30363d; --text:#e6edf3;
          --muted:#8b949e; --bull:#2ea043; --blue:#388bfd; --purple:#a371f7; --teal:#39c5cf; --orange:#f78166; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--text); min-height:100vh;
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }}
  header {{ padding:40px 32px 8px; }}
  h1 {{ margin:0; font-size:26px; }}
  .sub {{ color:var(--muted); font-size:13px; margin-top:6px; }}
  .cards {{ display:flex; gap:20px; padding:28px 32px; flex-wrap:wrap; }}
  a.card {{ flex:1; min-width:300px; text-decoration:none; color:inherit;
           background:var(--panel); border:1px solid var(--border); border-radius:14px;
           padding:24px; transition:border-color .15s, transform .15s; }}
  a.card:hover {{ transform:translateY(-2px); }}
  a.card.trend:hover {{ border-color:var(--bull); }}
  a.card.rev:hover {{ border-color:var(--blue); }}
  a.card.mom:hover {{ border-color:var(--purple); }}
  .emoji {{ font-size:32px; }}
  .ct {{ font-size:19px; font-weight:700; margin:10px 0 6px; }}
  .cd {{ color:var(--muted); font-size:13px; line-height:1.6; }}
  .tag {{ display:inline-block; margin-top:14px; font-size:12px; font-weight:700;
         padding:4px 10px; border-radius:6px; }}
  .trend .tag {{ background:rgba(46,160,67,.18); color:var(--bull); }}
  .rev .tag {{ background:rgba(56,139,253,.18); color:var(--blue); }}
  .mom .tag {{ background:rgba(163,113,247,.18); color:var(--purple); }}
  a.card.fund:hover {{ border-color:var(--teal); }}
  .fund .tag {{ background:rgba(57,197,207,.16); color:var(--teal); }}
  a.card.theme:hover {{ border-color:var(--orange); }}
  .theme .tag {{ background:rgba(247,129,102,.16); color:var(--orange); }}
  footer {{ color:var(--muted); font-size:12px; padding:8px 32px 40px; }}
</style></head><body>
<header>
  <h1>📊 Daily Market Scan</h1>
  <div class="sub">Generated {generated:%A, %B %d %Y · %H:%M}</div>
</header>
<div class="cards">
  <a class="card trend" href="ema_dashboard.html">
    <div class="emoji">📈</div>
    <div class="ct">EMA Stack Scanner</div>
    <div class="cd">Trend-following. S&amp;P 500 + Nasdaq 100 ranked 0–10 by weekly/daily
      EMA alignment, with entry zones, scale-out targets and EMA9-break freshness.
      Hold days–weeks; exit on a close below the EMA50.</div>
    <span class="tag">TREND · OPEN DASHBOARD →</span>
  </a>
  <a class="card rev" href="oversold_dashboard.html">
    <div class="emoji">🩸</div>
    <div class="ct">Oversold Big-Name Scanner</div>
    <div class="cd">Mean-reversion. Big liquid names that closed oversold (RSI-2) inside an
      uptrend — buy the dip, exit on the bounce to the 5-day mean, −3% stop,
      with earnings warnings. Hold ~1–2 days.</div>
    <span class="tag">MEAN-REVERSION · OPEN DASHBOARD →</span>
  </a>
  <a class="card mom" href="momentum_dashboard.html">
    <div class="emoji">🚀</div>
    <div class="ct">Momentum Rotation</div>
    <div class="cd">Concentrated top-5 by 6-month momentum, sized by signal strength,
      with a SPY 200-day regime switch to cash in downtrends. A sample portfolio
      to rebalance ~monthly.</div>
    <span class="tag">MOMENTUM · OPEN DASHBOARD →</span>
  </a>
  <a class="card fund" href="next_nvidia_dashboard.html">
    <div class="emoji">🔎</div>
    <div class="ct">Next-NVIDIA Screen</div>
    <div class="cd">Fundamentals, not price. Ranks the universe against NVDA's pre-explosion
      fingerprint — high gross margin (the moat), heavy R&amp;D, accelerating revenue and a
      wide gross-to-net gap (dormant operating leverage). Tiered, with a QoQ staircase to
      separate a real inflection from one lumpy year. A research watchlist, not a buy list.</div>
    <span class="tag">FUNDAMENTAL · OPEN DASHBOARD →</span>
  </a>
  <a class="card theme" href="bottleneck_dashboard.html">
    <div class="emoji">🏗️</div>
    <div class="ct">AI Buildout Bottlenecks</div>
    <div class="cd">Thematic. Maps the data-center boom's supply constraints — power, grid,
      cooling, HBM, optical, fab equipment — to the public companies that relieve each, then
      ranks them by where demand is rotating now (relative strength + momentum). A research
      map of the AI supply chain, not a forecast.</div>
    <span class="tag">THEMATIC · OPEN DASHBOARD →</span>
  </a>
</div>
<footer>Three price edges (trend / dip / momentum) + a fundamental lens + a thematic map. Not investment advice.</footer>
</body></html>"""
    with open("index.html", "w") as f:
        f.write(html)


def main() -> int:
    print("=" * 60)
    print("DAILY MARKET SCAN")
    print("=" * 60)

    # Both scanners use the same universe and 1-year daily history, so fetch
    # them once here and share — the EMA scanner adds its own weekly pull.
    print("\nFetching shared universe + daily history (once) ...")
    tickers = ema_scanner.get_universe()
    daily = ema_scanner.download(tickers, ema_scanner.DAILY_PERIOD, "1d")
    print(f"  {len(tickers)} tickers · daily history cached for both scanners.")

    print("\n[A] Trend — EMA stack scanner")
    print("-" * 60)
    ema_scanner.main(tickers=tickers, daily=daily)

    print("\n[B] Mean-reversion — oversold big-name scanner")
    print("-" * 60)
    oversold_scanner.main(tickers=tickers, daily=daily)

    print("\n[C] Momentum — rotation portfolio")
    print("-" * 60)
    momentum_scanner.main(tickers=tickers, daily=daily)

    # Fundamentals can't share the price download (yfinance has no batch
    # financials endpoint) and barely move day to day, so the heavy per-ticker
    # scan runs WEEKLY (Mondays); other days republish last Monday's dashboard.
    print("\n[D] Fundamentals — next-NVIDIA screen (weekly)")
    print("-" * 60)
    next_nvidia.main(tickers=tickers, weekly=True)

    print("\n[E] Thematic — AI buildout bottlenecks")
    print("-" * 60)
    bottleneck_scanner.main(tickers=tickers, daily=daily)

    write_index(datetime.now().astimezone())
    print("\n" + "=" * 60)
    print("✅ Both dashboards built. Open index.html")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
