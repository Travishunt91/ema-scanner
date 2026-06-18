#!/usr/bin/env python3
"""
Momentum rotation dashboard — concentrated top-N, signal-weighted, regime-gated.

Each daily run, rank the S&P 500 + Nasdaq 100 by 6-month (skip-1-month) momentum,
hold the TOP_N strongest names in uptrends, sized by signal strength (momentum-
weighted, capped). Switch to CASH when the broad market (SPY) is below its
200-day. Same method as momentum_sim.py — used forward, which is its valid use.

Writes momentum_dashboard.html.
"""

from __future__ import annotations

import html
import json
import os
import urllib.request
from datetime import date, datetime

import pandas as pd

from ema_scanner import TICKER_NAMES, download, get_universe

MOM_LOOKBACK = 126      # ~6 months trailing return
MOM_SKIP = 21           # skip the most recent ~month
TOP_N = 5               # concentrated portfolio size
MAX_WEIGHT = 0.40       # cap any single position
TREND_LEN = 200         # uptrend filter (SMA, matches the backtest)
SAMPLE_CAPITAL = 100_000.0
BENCH_N = 10            # how many "next in line" names to show
OUTPUT_HTML = "momentum_dashboard.html"
STATE_FILE = "momentum_state.json"
# Previous run's state is chained through the published Pages file so it survives
# GitHub's ephemeral runners (no local file on a fresh checkout).
PAGES_STATE_URL = "https://travishunt91.github.io/ema-scanner/momentum_state.json"
WEIGHT_EPS = 1.0        # ignore weight wiggles smaller than this (pp)


def momentum_pct(close: pd.Series):
    if len(close) < MOM_LOOKBACK + MOM_SKIP + 1:
        return None
    recent = float(close.iloc[-1 - MOM_SKIP])
    base = float(close.iloc[-1 - MOM_SKIP - MOM_LOOKBACK])
    return (recent / base - 1) * 100.0 if base > 0 else None


def spy_regime() -> dict:
    try:
        d = download(["SPY"], "1y", "1d")["SPY"]
        if isinstance(d.columns, pd.MultiIndex):
            d = d.droplevel(0, axis=1)
        c = d["Close"].dropna()
        if len(c) < TREND_LEN:
            return {}
        price = float(c.iloc[-1]); sma = float(c.rolling(TREND_LEN).mean().iloc[-1])
        return dict(risk_on=price > sma, pct=(price / sma - 1) * 100)
    except Exception:
        return {}


def rank_candidates(tickers, daily):
    """All names in an uptrend with positive momentum, ranked strongest first."""
    cands = []
    for t in tickers:
        df = daily.get(t)
        if df is None or df.empty:
            continue
        c = df["Close"].dropna()
        if len(c) < TREND_LEN + 5:
            continue
        mom = momentum_pct(c)
        if mom is None or mom <= 0:
            continue
        price = float(c.iloc[-1])
        sma200 = float(c.rolling(TREND_LEN).mean().iloc[-1])
        if price <= sma200:                       # uptrend filter
            continue
        cands.append(dict(ticker=t, name=TICKER_NAMES.get(t, ""),
                          momentum=mom, price=price))
    cands.sort(key=lambda x: x["momentum"], reverse=True)
    return cands


def weighted_top(cands):
    top = cands[:TOP_N]
    if not top:
        return []
    raw = [c["momentum"] for c in top]
    w = [x / sum(raw) for x in raw]
    w = [min(x, MAX_WEIGHT) for x in w]
    w = [x / sum(w) for x in w]
    for c, wi in zip(top, w):
        c["weight"] = wi * 100
        c["dollars"] = SAMPLE_CAPITAL * wi
        c["shares"] = int(SAMPLE_CAPITAL * wi / c["price"]) if c["price"] else 0
    return top


def load_prev_state() -> dict | None:
    """Yesterday's portfolio: local file first (local runs), else the live Pages
    copy (cloud runs chain through the deployed momentum_state.json)."""
    if os.path.exists(STATE_FILE):
        try:
            return json.load(open(STATE_FILE))
        except Exception:
            pass
    try:
        req = urllib.request.Request(PAGES_STATE_URL, headers={"User-Agent": "momentum-scanner"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def save_state(picks, regime) -> None:
    json.dump({
        "date": date.today().isoformat(),
        "risk_on": bool(regime.get("risk_on", True)) if regime else True,
        "holdings": {p["ticker"]: round(p["weight"], 1) for p in picks},
    }, open(STATE_FILE, "w"))


def compute_moves(picks, regime, prev) -> dict | None:
    """Annotate each pick with its change vs last run; return buy/sell/trim moves."""
    if not prev:
        return None  # baseline run, nothing to diff against
    prev_h = prev.get("holdings", {})
    cur = {p["ticker"]: p["weight"] for p in picks}
    for p in picks:
        pw = prev_h.get(p["ticker"])
        if pw is None:
            p["change"] = ("NEW", None)
        else:
            d = p["weight"] - pw
            p["change"] = ("ADD", d) if d > WEIGHT_EPS else ("TRIM", d) if d < -WEIGHT_EPS else ("HOLD", d)
    sells = sorted([t for t in prev_h if t not in cur])
    now_cash = bool(regime and not regime["risk_on"])
    return dict(
        buys=[p["ticker"] for p in picks if p["change"][0] == "NEW"],
        sells=sells,
        adds=[(p["ticker"], p["change"][1]) for p in picks if p["change"][0] == "ADD"],
        trims=[(p["ticker"], p["change"][1]) for p in picks if p["change"][0] == "TRIM"],
        now_cash=now_cash,
        was_cash=not prev.get("risk_on", True),
    )


def _moves_box(moves) -> str:
    if moves is None:
        return '<div class="moves base">Baseline portfolio set — moves will be tracked from the next run.</div>'
    chips = []
    if moves["now_cash"]:
        if moves["sells"]:
            chips.append(f'<span class="mv sell">SELL ALL · {", ".join(moves["sells"])}</span>')
        chips.append('<span class="mv cashm">→ CASH (market risk-off)</span>')
    else:
        for t in moves["buys"]:
            chips.append(f'<span class="mv buy">BUY {t}</span>')
        for t in moves["sells"]:
            chips.append(f'<span class="mv sell">SELL {t}</span>')
        for t, d in moves["adds"]:
            chips.append(f'<span class="mv add">ADD {t} +{d:.0f}%</span>')
        for t, d in moves["trims"]:
            chips.append(f'<span class="mv trim">TRIM {t} {d:.0f}%</span>')
    inner = " ".join(chips) if chips else '<span class="mv hold">No changes — hold current portfolio.</span>'
    return f'<div class="moves"><span class="moves-h">Moves vs last run:</span> {inner}</div>'


def _change_badge(p) -> str:
    if "change" not in p:
        return '<span class="chg none">—</span>'
    kind, d = p["change"]
    if kind == "NEW":
        return '<span class="chg new">NEW</span>'
    if kind == "ADD":
        return f'<span class="chg add">+{d:.0f}%</span>'
    if kind == "TRIM":
        return f'<span class="chg trim">{d:.0f}%</span>'
    return '<span class="chg hold">hold</span>'


def render(picks, bench, regime, generated, moves=None) -> str:
    reg = regime or {}
    if reg:
        cls = "on" if reg["risk_on"] else "off"
        chip = f'<span class="chip {cls}">SPY {reg["pct"]:+.1f}% vs 200-day · risk-{"on" if reg["risk_on"] else "off"}</span>'
    else:
        chip = ""

    moves_html = _moves_box(moves)
    if reg and not reg["risk_on"]:
        body = f"""
    {moves_html}
    <div class="cash">💵 <b>CASH — broad market is below its 200-day.</b>
      Momentum rotation sits out downtrends, so no positions are suggested today.
      Re-engage when SPY reclaims its 200-day average.</div>"""
    elif not picks:
        body = '<div class="cash">No qualifying names (need positive momentum in an uptrend).</div>'
    else:
        prows = "".join(f"""
        <tr>
          <td class="rank">{i+1}</td>
          <td class="tk">{p['ticker']}<span class="co">{html.escape(p['name'])}</span></td>
          <td>{_change_badge(p)}</td>
          <td class="num w">{p['weight']:.0f}%</td>
          <td class="num">${p['dollars']:,.0f}</td>
          <td class="num">${p['price']:,.2f}</td>
          <td class="num">{p['shares']:,}</td>
          <td class="num mom">+{p['momentum']:.0f}%</td>
        </tr>""" for i, p in enumerate(picks))
        brows = "".join(f"""
        <tr>
          <td class="rank">{TOP_N+i+1}</td>
          <td class="tk">{b['ticker']}<span class="co">{html.escape(b['name'])}</span></td>
          <td class="num">${b['price']:,.2f}</td>
          <td class="num mom">+{b['momentum']:.0f}%</td>
        </tr>""" for i, b in enumerate(bench))
        body = f"""
    {moves_html}
    <table class="tbl">
      <thead><tr><th>#</th><th>Ticker</th><th>Change</th><th class="num">Weight</th><th class="num">Allocate</th>
        <th class="num">Price</th><th class="num">~Shares</th><th class="num">6m Mom</th></tr></thead>
      <tbody>{prows}</tbody>
    </table>
    <div class="bench-h">Next in line (rebalance candidates)</div>
    <table class="tbl bench">
      <thead><tr><th>#</th><th>Ticker</th><th class="num">Price</th><th class="num">6m Mom</th></tr></thead>
      <tbody>{brows}</tbody>
    </table>"""

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Momentum Rotation</title>
<style>
  :root {{ --bg:#0d1117; --panel:#161b22; --border:#30363d; --text:#e6edf3; --muted:#8b949e;
          --bull:#2ea043; --bear:#f85149; --blue:#388bfd; }}
  body {{ margin:0; background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif; }}
  header {{ padding:24px 32px; border-bottom:1px solid var(--border); }}
  h1 {{ margin:0 0 4px; font-size:22px; }} .sub {{ color:var(--muted); font-size:13px; }}
  .chip {{ font-size:11px; font-weight:700; padding:3px 9px; border-radius:6px; margin-left:10px; }}
  .chip.on {{ background:rgba(46,160,67,.18); color:var(--bull); }}
  .chip.off {{ background:rgba(248,81,73,.18); color:var(--bear); }}
  .wrap {{ padding:22px 32px 40px; max-width:880px; }}
  .tbl {{ width:100%; border-collapse:collapse; background:var(--panel); border:1px solid var(--border); border-radius:10px; overflow:hidden; }}
  .tbl th, .tbl td {{ padding:9px 12px; font-size:13px; border-bottom:1px solid var(--border); text-align:left; }}
  .tbl th {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.5px; }}
  .tbl td.num, .tbl th.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  .tbl tr:last-child td {{ border-bottom:none; }}
  .rank {{ color:var(--muted); font-weight:700; width:24px; }}
  .tk {{ font-weight:700; }} .co {{ display:block; font-weight:400; color:var(--muted); font-size:11px; }}
  .w {{ font-weight:700; color:var(--blue); }} .mom {{ color:var(--bull); }}
  .bench-h {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.5px; margin:22px 0 8px; }}
  .bench {{ opacity:.85; }}
  .cash {{ background:rgba(248,81,73,.10); border:1px solid rgba(248,81,73,.35); border-radius:10px;
          padding:18px 20px; font-size:15px; }} .cash b {{ color:var(--bear); }}
  .note {{ color:var(--muted); font-size:12px; margin-top:18px; line-height:1.6; }}
  .moves {{ background:var(--panel); border:1px solid var(--border); border-radius:10px;
           padding:12px 16px; margin-bottom:16px; font-size:13px; line-height:2; }}
  .moves.base {{ color:var(--muted); }}
  .moves-h {{ color:var(--muted); font-weight:700; margin-right:6px; }}
  .mv {{ display:inline-block; padding:3px 9px; border-radius:6px; font-size:12px; font-weight:700; margin:0 4px 4px 0; }}
  .mv.buy {{ background:rgba(46,160,67,.20); color:var(--bull); }}
  .mv.sell {{ background:rgba(248,81,73,.20); color:var(--bear); }}
  .mv.add {{ background:rgba(46,160,67,.12); color:var(--bull); }}
  .mv.trim {{ background:rgba(210,153,34,.18); color:#d29922; }}
  .mv.hold, .mv.cashm {{ background:rgba(110,118,129,.18); color:var(--muted); }}
  .chg {{ font-size:11px; font-weight:700; padding:2px 7px; border-radius:5px; }}
  .chg.new {{ background:var(--bull); color:#051b0c; }}
  .chg.add {{ background:rgba(46,160,67,.18); color:var(--bull); }}
  .chg.trim {{ background:rgba(210,153,34,.18); color:#d29922; }}
  .chg.hold, .chg.none {{ background:rgba(110,118,129,.16); color:var(--muted); }}
</style></head><body>
<header>
  <h1>🚀 Momentum Rotation — Top {TOP_N}{chip}</h1>
  <div class="sub">S&amp;P 500 + Nasdaq 100 · weighted by 6-month momentum · ${SAMPLE_CAPITAL:,.0f} sample · Generated {generated:%Y-%m-%d %H:%M}</div>
</header>
<div class="wrap">{body}
  <div class="note">Forward-looking momentum: ranks names in an uptrend (above the 200-day SMA) by trailing
    6-month return (skipping the most recent month), sized by signal strength (capped {MAX_WEIGHT*100:.0f}%).
    Switches to CASH when SPY is below its 200-day. <b>Rebalance ~monthly, not daily</b> — day-to-day
    reshuffles are noise. Same method as the backtest; survivorship bias only affected historical returns,
    not forward selection. Not investment advice.</div>
</div></body></html>"""


def main(tickers=None, daily=None) -> int:
    print("Momentum scanner — building universe (S&P 500 + Nasdaq 100) ...")
    if tickers is None:
        tickers = get_universe()
    if daily is None:
        print(f"  {len(tickers)} tickers. Downloading 1y daily ...")
        daily = download(tickers, "1y", "1d")
    else:
        print(f"  {len(tickers)} tickers (reusing shared daily download).")

    cands = rank_candidates(tickers, daily)
    picks = weighted_top(cands)
    bench = cands[TOP_N:TOP_N + BENCH_N]
    regime = spy_regime()

    prev = load_prev_state()
    moves = compute_moves(picks, regime, prev)   # annotates picks with 'change'

    with open(OUTPUT_HTML, "w") as f:
        f.write(render(picks, bench, regime, datetime.now().astimezone(), moves))
    save_state(picks, regime)

    state = "CASH (risk-off)" if regime and not regime["risk_on"] else f"{len(picks)} names"
    print(f"\n✅ {OUTPUT_HTML} written · regime: {state}")
    for p in picks:
        chg = p.get("change", ("", None))[0]
        print(f"   {p['ticker']:6} {p['weight']:>4.0f}%  mom +{p['momentum']:.0f}%  {chg}")
    if moves:
        if moves["buys"]:
            print(f"   BUY:  {', '.join(moves['buys'])}")
        if moves["sells"]:
            print(f"   SELL: {', '.join(moves['sells'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
