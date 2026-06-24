#!/usr/bin/env python3
"""
AI Buildout Bottleneck dashboard — where the data-center boom is supply-constrained,
and which public companies are positioned to relieve each bottleneck.

This is a CURATED supply-chain map (domain knowledge) + a DEMAND ranking (objective
price signals: relative strength vs SPY, 3/6-month momentum, proximity to 52-week
high, trend). It surfaces where capital is rotating *now* across the AI buildout —
it is NOT a validated predictive model. Treat it as a research map, not a forecast.

Writes bottleneck_dashboard.html.
"""

from __future__ import annotations

import html
from datetime import datetime

import numpy as np
import pandas as pd

from ema_scanner import download

OUTPUT_HTML = "bottleneck_dashboard.html"
PERIOD = "2y"

# Curated bottleneck map: each constraint in the AI/data-center buildout, why it's
# a bottleneck, and the public "solvers" positioned to relieve it.
CATEGORIES = [
    dict(key="power", emoji="⚡", name="Power Generation",
         why="The #1 constraint. AI data centers need gigawatts of reliable, 24/7 baseload power. "
             "Nuclear, gas and independent power producers that can actually deliver electricity win.",
         tickers=["CEG", "VST", "NRG", "TLN", "GEV", "NEE", "SO", "D", "PEG",
                  "SMR", "OKLO", "BWXT", "LEU"]),
    dict(key="electrical", emoji="🔌", name="Electrical Equipment & Grid",
         why="Transformers, switchgear and power-distribution gear are in multi-year shortage — "
             "you can't energize a data center without them.",
         tickers=["ETN", "VRT", "POWL", "HUBB", "NVT", "PWR", "GEV"]),
    dict(key="cooling", emoji="❄️", name="Cooling & Thermal",
         why="Next-gen GPUs run too hot for air. Liquid/immersion cooling is becoming mandatory "
             "at rack densities the industry has never built before.",
         tickers=["VRT", "MOD", "SMCI"]),
    dict(key="compute", emoji="🖥️", name="Compute / Accelerators",
         why="The GPUs and custom silicon at the core. Supply is the binding constraint on how fast "
             "the buildout can scale.",
         tickers=["NVDA", "AMD", "AVGO", "MRVL", "TSM", "ARM"]),
    dict(key="memory", emoji="🧠", name="Memory / HBM",
         why="High-bandwidth memory (HBM) is sold out years forward — every accelerator needs stacks of it.",
         tickers=["MU", "WDC", "STX", "SNDK"]),
    dict(key="network", emoji="🌐", name="Networking / Optical",
         why="Moving data between tens of thousands of GPUs needs massive optical interconnect bandwidth.",
         tickers=["ANET", "AVGO", "CIEN", "COHR", "LITE", "CRDO"]),
    dict(key="equip", emoji="🏭", name="Semiconductor Equipment",
         why="The machines that make the chips. Every leading-edge wafer and HBM stack flows through them.",
         tickers=["ASML", "AMAT", "LRCX", "KLAC", "TER", "ENTG"]),
    dict(key="infra", emoji="🏢", name="Data-Center Infrastructure & REITs",
         why="The physical buildings, racks and colocation capacity the boom is consuming.",
         tickers=["EQIX", "DLR", "DELL", "SMCI"]),
    dict(key="build", emoji="🏗️", name="Construction & Engineering",
         why="Someone has to build the data centers and electrical infrastructure — specialized contractors are booked solid.",
         tickers=["PWR", "MTZ", "FIX", "ACM", "J"]),
    dict(key="materials", emoji="⛏️", name="Materials (Copper)",
         why="Electrification of the buildout is copper-intensive; supply is structurally tight.",
         tickers=["FCX"]),
]

# Names for tickers that aren't in the S&P/Nasdaq universe map.
NAMES = {
    "CEG": "Constellation Energy", "VST": "Vistra", "NRG": "NRG Energy", "TLN": "Talen Energy",
    "GEV": "GE Vernova", "NEE": "NextEra Energy", "SO": "Southern Co", "D": "Dominion Energy",
    "PEG": "Public Service Ent.", "SMR": "NuScale Power", "OKLO": "Oklo", "BWXT": "BWX Technologies",
    "LEU": "Centrus Energy", "ETN": "Eaton", "VRT": "Vertiv", "POWL": "Powell Industries",
    "HUBB": "Hubbell", "NVT": "nVent Electric", "PWR": "Quanta Services", "MOD": "Modine Mfg",
    "SMCI": "Super Micro", "NVDA": "Nvidia", "AMD": "AMD", "AVGO": "Broadcom", "MRVL": "Marvell",
    "TSM": "TSMC", "ARM": "Arm Holdings", "MU": "Micron", "WDC": "Western Digital", "STX": "Seagate",
    "SNDK": "Sandisk", "ANET": "Arista Networks", "CIEN": "Ciena", "COHR": "Coherent",
    "LITE": "Lumentum", "CRDO": "Credo Technology", "ASML": "ASML", "AMAT": "Applied Materials",
    "LRCX": "Lam Research", "KLAC": "KLA Corp", "TER": "Teradyne", "ENTG": "Entegris",
    "EQIX": "Equinix", "DLR": "Digital Realty", "DELL": "Dell Technologies", "MTZ": "MasTec",
    "FIX": "Comfort Systems", "ACM": "AECOM", "J": "Jacobs Solutions", "FCX": "Freeport-McMoRan",
}


def all_tickers():
    seen = []
    for c in CATEGORIES:
        for t in c["tickers"]:
            if t not in seen:
                seen.append(t)
    return seen


def metrics(close: pd.Series, spy_m3: float):
    if len(close) < 130:
        return None
    px = float(close.iloc[-1])
    mom3 = px / float(close.iloc[-64]) - 1
    mom6 = px / float(close.iloc[-127]) - 1
    hi52 = float(close.tail(252).max())
    dist52 = px / hi52 - 1
    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else float("nan")
    sma50 = float(close.rolling(50).mean().iloc[-1])
    trend = (px > sma200) + (px > sma50) if np.isfinite(sma200) else (px > sma50)
    return dict(price=px, mom3=mom3, mom6=mom6, dist52=dist52,
                rs3=mom3 - spy_m3, trend=trend / 2)


def build(daily, spy_m3):
    out = {}
    for t in all_tickers():
        df = daily.get(t)
        if df is None or df.empty:
            continue
        m = metrics(df["Close"].dropna(), spy_m3)
        if m:
            out[t] = m
    # Demand score: within-set percentile ranks, weighted.
    if not out:
        return out
    frame = pd.DataFrame(out).T
    def rk(col):
        return frame[col].rank(pct=True)
    frame["demand"] = (0.30 * rk("rs3") + 0.25 * rk("mom6") + 0.20 * rk("mom3")
                       + 0.15 * rk("dist52") + 0.10 * frame["trend"]) * 100
    for t in out:
        out[t]["demand"] = float(frame.loc[t, "demand"])
    return out


def render(data, generated) -> str:
    # Category heat = average demand score of its members (present in data).
    cats = []
    for c in CATEGORIES:
        members = [(t, data[t]) for t in c["tickers"] if t in data]
        if not members:
            continue
        members.sort(key=lambda x: x[1]["demand"], reverse=True)
        heat = float(np.mean([m[1]["demand"] for m in members]))
        cats.append((c, members, heat))
    cats.sort(key=lambda x: x[2], reverse=True)

    def dcol(v):
        return "hot" if v >= 66 else "warm" if v >= 40 else "cool"

    # Heat overview bars
    heat_rows = "".join(f"""
      <div class="heat-row">
        <div class="heat-label">{c['emoji']} {c['name']}</div>
        <div class="heat-bar"><div class="heat-fill {dcol(heat)}" style="width:{heat:.0f}%"></div></div>
        <div class="heat-val">{heat:.0f}</div>
      </div>""" for c, _, heat in cats)

    sections = ""
    for c, members, heat in cats:
        rows = "".join(f"""
        <tr>
          <td class="tk">{t}<span class="co">{html.escape(NAMES.get(t, ''))}</span></td>
          <td class="num"><span class="dscore {dcol(m['demand'])}">{m['demand']:.0f}</span></td>
          <td class="num">${m['price']:,.2f}</td>
          <td class="num {'pos' if m['rs3']>=0 else 'neg'}">{m['rs3']*100:+.0f}%</td>
          <td class="num {'pos' if m['mom3']>=0 else 'neg'}">{m['mom3']*100:+.0f}%</td>
          <td class="num {'pos' if m['mom6']>=0 else 'neg'}">{m['mom6']*100:+.0f}%</td>
          <td class="num">{m['dist52']*100:+.0f}%</td>
        </tr>""" for t, m in members)
        sections += f"""
    <div class="cat">
      <div class="cat-head"><span class="cat-name">{c['emoji']} {c['name']}</span>
        <span class="cat-heat {dcol(heat)}">heat {heat:.0f}</span></div>
      <div class="cat-why">{c['why']}</div>
      <table class="tbl">
        <thead><tr><th>Ticker</th><th class="num">Demand</th><th class="num">Price</th>
          <th class="num">RS vs SPY (3m)</th><th class="num">3m</th><th class="num">6m</th><th class="num">vs 52w hi</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>AI Buildout Bottlenecks</title>
<style>
  :root {{ --bg:#0d1117; --panel:#161b22; --border:#30363d; --text:#e6edf3; --muted:#8b949e;
          --bull:#2ea043; --bear:#f85149; --hot:#f78166; --warm:#d29922; --cool:#388bfd; }}
  body {{ margin:0; background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif; }}
  header {{ padding:24px 32px; border-bottom:1px solid var(--border); }}
  h1 {{ margin:0 0 4px; font-size:22px; }} .sub {{ color:var(--muted); font-size:13px; }}
  .wrap {{ padding:22px 32px 40px; max-width:1000px; }}
  .overview {{ background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:18px 20px; margin-bottom:24px; }}
  .ov-h {{ font-size:14px; font-weight:700; margin-bottom:14px; }}
  .heat-row {{ display:flex; align-items:center; gap:12px; margin:7px 0; }}
  .heat-label {{ width:230px; font-size:13px; }}
  .heat-bar {{ flex:1; height:10px; background:rgba(110,118,129,.18); border-radius:5px; overflow:hidden; }}
  .heat-fill {{ height:100%; border-radius:5px; }}
  .heat-fill.hot {{ background:var(--hot); }} .heat-fill.warm {{ background:var(--warm); }} .heat-fill.cool {{ background:var(--cool); }}
  .heat-val {{ width:30px; text-align:right; font-variant-numeric:tabular-nums; font-weight:700; font-size:13px; }}
  .cat {{ margin-bottom:26px; }}
  .cat-head {{ display:flex; align-items:center; gap:12px; }}
  .cat-name {{ font-size:17px; font-weight:700; }}
  .cat-heat {{ font-size:11px; font-weight:700; padding:3px 9px; border-radius:6px; }}
  .cat-heat.hot {{ background:rgba(247,129,102,.22); color:var(--hot); }}
  .cat-heat.warm {{ background:rgba(210,153,34,.18); color:var(--warm); }}
  .cat-heat.cool {{ background:rgba(56,139,253,.18); color:var(--cool); }}
  .cat-why {{ color:var(--muted); font-size:12.5px; line-height:1.5; margin:6px 0 12px; max-width:760px; }}
  .tbl {{ width:100%; border-collapse:collapse; background:var(--panel); border:1px solid var(--border); border-radius:10px; overflow:hidden; }}
  .tbl th, .tbl td {{ padding:8px 12px; font-size:13px; border-bottom:1px solid var(--border); text-align:left; }}
  .tbl th {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.5px; }}
  .tbl td.num, .tbl th.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  .tbl tr:last-child td {{ border-bottom:none; }}
  .tk {{ font-weight:700; }} .co {{ display:block; font-weight:400; color:var(--muted); font-size:11px; }}
  .pos {{ color:var(--bull); }} .neg {{ color:var(--bear); }}
  .dscore {{ font-weight:800; padding:2px 8px; border-radius:5px; }}
  .dscore.hot {{ background:var(--hot); color:#1a0a06; }}
  .dscore.warm {{ background:rgba(210,153,34,.20); color:var(--warm); }}
  .dscore.cool {{ background:rgba(110,118,129,.18); color:var(--muted); }}
  .note {{ color:var(--muted); font-size:12px; line-height:1.6; margin-top:8px; }}
</style></head><body>
<header>
  <h1>🏗️ AI Buildout Bottleneck Scanner</h1>
  <div class="sub">Where the data-center boom is supply-constrained, and who relieves it · Generated {generated:%Y-%m-%d %H:%M}</div>
</header>
<div class="wrap">
  <div class="overview">
    <div class="ov-h">🔥 Bottleneck Heat — which constraint the market is bidding up now (0–100 demand)</div>
    {heat_rows}
  </div>
  {sections}
  <div class="note"><b>How to read this:</b> "Demand" (0–100) ranks each name within the AI-buildout set by
    relative strength vs SPY (3m), 3- &amp; 6-month momentum, proximity to its 52-week high, and trend — i.e. where
    capital is rotating. "Heat" is a category's average demand. <b>This is a curated thematic map + a momentum
    ranking, not a validated forecast</b> — it tells you where demand is showing up, not what will win. Momentum
    can reverse hard; size accordingly. Not investment advice.</div>
</div></body></html>"""


def main(tickers=None, daily=None) -> int:
    print("AI bottleneck scanner — curated data-center supply chain")
    need = all_tickers() + ["SPY"]
    have = daily or {}
    missing = [t for t in need if t not in have]
    if missing:
        print(f"  downloading {len(missing)} names ({PERIOD}) ...")
        fetched = download(missing, PERIOD, "1d")
        have = {**have, **fetched}

    spy = have.get("SPY")
    if spy is not None and isinstance(spy.columns, pd.MultiIndex):
        spy = spy.droplevel(0, axis=1)
    spy_c = spy["Close"].dropna() if spy is not None else None
    spy_m3 = (float(spy_c.iloc[-1]) / float(spy_c.iloc[-64]) - 1) if spy_c is not None and len(spy_c) > 64 else 0.0

    data = build(have, spy_m3)
    with open(OUTPUT_HTML, "w") as f:
        f.write(render(data, datetime.now().astimezone()))

    print(f"\n✅ {OUTPUT_HTML} written · {len(data)} names scored")
    top = sorted(data.items(), key=lambda kv: kv[1]["demand"], reverse=True)[:8]
    for t, m in top:
        print(f"   {t:6} demand {m['demand']:5.0f} | RS3 {m['rs3']*100:+5.0f}% | 6m {m['mom6']*100:+5.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
