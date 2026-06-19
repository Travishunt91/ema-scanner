#!/usr/bin/env python3
"""
"Next NVIDIA" fundamental screener.

Reverse-engineers NVDA's pre-explosion fingerprint (FY2023 state, the year
before the FY2024 AI ignition) and scores the S&P 500 + Nasdaq 100 universe
for companies that look the same TODAY.

The thesis (from NVDA's own numbers):
  - A high gross margin (~57%) was ALREADY in place before revenue exploded
    -> pricing power / moat must exist FIRST.
  - Heavy R&D (~27% of revenue) -> building a platform, not milking a product.
  - Net margin was still LOW (16%) -> operating leverage was dormant.
  - When revenue doubled, fixed costs didn't, so net margin went 16% -> 49%
    and EPS +584%. The GAP between gross and net margin is the stored upside.

So we screen for: high gross margin + heavy R&D + accelerating revenue growth
+ a wide gross-to-net gap + a revenue base small enough to still 5-10x.

Not investment advice. A matching fingerprint is necessary, not sufficient.
"""
from __future__ import annotations
import sys, time, warnings, urllib.request
import numpy as np
import yfinance as yf

warnings.filterwarnings("ignore")

# Fundamentals only move on earnings, so the expensive per-ticker scan runs
# WEEKLY (Mondays). On other weekdays the cloud job re-downloads last Monday's
# published dashboard and republishes it unchanged -- mirroring how the momentum
# scanner chains state through the live Pages copy.
PAGES_HTML_URL = "https://travishunt91.github.io/ema-scanner/next_nvidia_dashboard.html"
PAGES_CSV_URL = "https://travishunt91.github.io/ema-scanner/next_nvidia_screen.csv"


def _is_refresh_day() -> bool:
    """True on Mondays (Central time), when we do the full fundamental scan."""
    from datetime import datetime
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("America/Chicago"))
    except Exception:
        now = datetime.now()
    return now.weekday() == 0


def _republish_from_pages() -> bool:
    """Pull last Monday's published dashboard + CSV and write them locally so the
    collect-for-Pages step republishes them. Returns False if unavailable."""
    ok = False
    for url, dest in ((PAGES_HTML_URL, "next_nvidia_dashboard.html"),
                      (PAGES_CSV_URL, "next_nvidia_screen.csv")):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "next-nvidia"})
            with urllib.request.urlopen(req, timeout=20) as r:
                data = r.read()
            with open(dest, "wb") as f:
                f.write(data)
            ok = True
        except Exception as e:
            print(f"  could not fetch {dest} from Pages ({e}); will regenerate.")
            return False
    return ok

# --- NVDA pre-explosion reference (FY2023) ---
REF = dict(gm=56.9, nm=16.2, rd=27.2, gNow=60.0)  # gNow: target accel growth


def metrics(tk: str):
    """Pull latest-year fundamentals for one ticker. None on failure."""
    try:
        t = yf.Ticker(tk)
        af = t.financials
        if af is None or af.empty:
            return None

        def g(names):
            for n in ([names] if isinstance(names, str) else names):
                if n in af.index:
                    return af.loc[n]
            return None

        rev = g(["Total Revenue", "Operating Revenue"])
        gp, ni, rd = g("Gross Profit"), g("Net Income"), g("Research And Development")
        if rev is None or gp is None or ni is None:
            return None

        cols = list(af.columns)[::-1]                      # oldest -> newest
        rev_v = [rev[c] for c in cols]
        gpv = [gp[c] for c in cols]
        niv = [ni[c] for c in cols]
        rdv = [rd[c] if rd is not None and c in rd.index else np.nan for c in cols]

        revL = rev_v[-1]
        if not revL or revL <= 0:
            return None
        gm = gpv[-1] / revL * 100
        nm = niv[-1] / revL * 100
        rdi = (rdv[-1] / revL * 100) if not np.isnan(rdv[-1]) else 0.0

        yoy = [(rev_v[i] / rev_v[i - 1] - 1) * 100
               for i in range(1, len(rev_v)) if rev_v[i - 1] > 0]
        g_now = yoy[-1] if yoy else np.nan
        g_prev = yoy[-2] if len(yoy) > 1 else np.nan
        accel = (g_now - g_prev) if not np.isnan(g_prev) else 0.0

        return dict(tk=tk, revB=revL / 1e9, gm=gm, nm=nm, rd=rdi,
                    gNow=g_now, accel=accel, gap=gm - nm)
    except Exception:
        return None


def score(m) -> float:
    """0-100 similarity to the NVDA-pre-explosion fingerprint."""
    if m is None or np.isnan(m["gNow"]):
        return 0.0
    s = 0.0
    # 1. Gross margin >= 55% is the moat gate (35 pts, ramps from 40%)
    s += 35 * np.clip((m["gm"] - 40) / (60 - 40), 0, 1)
    # 2. Revenue growth, accelerating (25 pts for >=25%, bonus for accel)
    s += 20 * np.clip(m["gNow"] / 40, 0, 1)
    s += 5 * np.clip(m["accel"] / 15, 0, 1)
    # 3. Heavy R&D investment (15 pts, >=15% of rev)
    s += 15 * np.clip(m["rd"] / 15, 0, 1)
    # 4. Operating-leverage runway: wide gross-net gap (15 pts)
    s += 15 * np.clip(m["gap"] / 40, 0, 1)
    # 5. Small-enough base to still 5-10x (10 pts, sweet spot 1-40B)
    if m["revB"] < 1:
        s += 4
    elif m["revB"] <= 40:
        s += 10
    elif m["revB"] <= 80:
        s += 5
    return round(s, 1)


TIERS = {
    "tier1": ("Tier 1 · proven leverage", "Moat (gross margin >=50%) with a POSITIVE but still-low net "
              "margin and real growth -- leverage proven and with runway. The true NVDA-FY2023 shape."),
    "tier2": ("Tier 2 · unproven leverage", "High gross margin + growth, but net margin near/below zero. "
              "Leverage is hoped-for, not proven. Higher ceiling, real chance of a deep drawdown."),
    "fired": ("Already fired", "Net margin already high (>=35%) -- the operating leverage is largely "
              "realized. Mid-explosion, not pre. Less asymmetry left (today's NVDA lives here)."),
    "artifact": ("Screen artifact", "Extreme losses or R&D (clinical-stage biotech / cash-burner). The "
                 "fingerprint matches mechanically but the business dynamics don't. Ignore."),
    "lowmoat": ("Low moat", "Gross margin under 50% -- not enough pricing power to throw off NVDA-style "
                "operating leverage even if revenue scales."),
    "watch": ("Watch", "Has the moat but growth is slow -- no inflection underway."),
}


def classify(m) -> str:
    """Bucket a name by the financial SHAPE (not its GICS label)."""
    gm, nm, g, rd = m["gm"], m["nm"], m["gNow"], m["rd"]
    if np.isnan(g):
        return "watch"
    if nm < -50 or rd > 60:          # clinical-stage burner / screen noise
        return "artifact"
    if gm < 50:
        return "lowmoat"
    if nm >= 35:                      # leverage already cashed in
        return "fired"
    if g < 15:
        return "watch"
    return "tier1" if nm > 5 else "tier2"


def qoq_data(tk: str):
    """Last ~5 quarters: revenue, sequential QoQ %, gross & net margin + a verdict."""
    try:
        t = yf.Ticker(tk)
        qf = t.quarterly_financials
        if qf is None or qf.empty:
            return None

        def s(names):
            for n in ([names] if isinstance(names, str) else names):
                if n in qf.index:
                    return qf.loc[n]
            return None

        rev, gp, ni = s(["Total Revenue", "Operating Revenue"]), s("Gross Profit"), s("Net Income")
        if rev is None or ni is None:
            return None
        cols = list(qf.columns)[::-1]
        qs, prev = [], None
        for c in cols:
            r = rev[c]
            if r is None or (isinstance(r, float) and np.isnan(r)) or r <= 0:
                continue
            qs.append(dict(
                date=str(c.date()), rev=r / 1e6,
                qoq=((r / prev - 1) * 100) if prev else None,
                gm=(gp[c] / r * 100) if gp is not None and not np.isnan(gp[c]) else None,
                nm=(ni[c] / r * 100) if not np.isnan(ni[c]) else None))
            prev = r
        if len(qs) < 3:
            return None
        qoqs = [q["qoq"] for q in qs if q["qoq"] is not None]
        nms = [q["nm"] for q in qs if q["nm"] is not None]
        verdict = []
        verdict.append("QoQ positive & holding" if all(q > 0 for q in qoqs[-3:]) else "QoQ choppy")
        verdict.append("re-accelerating" if qoqs[-1] > np.mean(qoqs[:-1]) else "cooling")
        verdict.append("margin stair-steps up" if len(nms) >= 3 and nms[-1] > nms[0] else "margin flat/down")
        return dict(tk=tk, quarters=qs, verdict=verdict)
    except Exception:
        return None


def get_basket():
    try:
        from ema_scanner import get_universe
        return sorted(set(get_universe()))
    except Exception:
        return None


TIER_BADGE = {"tier1": "bull", "tier2": "watch", "fired": "fired",
              "artifact": "bear", "lowmoat": "mixed", "watch": "mixed"}


def _spark(vals, neg_ok=False):
    """Tiny inline bar sparkline (positive scale)."""
    clean = [v for v in vals if v is not None]
    if not clean:
        return ""
    lo = min(0, min(clean)) if neg_ok else 0
    hi = max(clean) or 1
    span = (hi - lo) or 1
    bars = []
    for v in vals:
        if v is None:
            bars.append('<i class="sp" style="height:2px;opacity:.3"></i>')
            continue
        h = max(2, round((v - lo) / span * 26))
        cls = "sp" + (" dn" if v < 0 else "")
        bars.append(f'<i class="{cls}" style="height:{h}px" title="{v:.1f}"></i>')
    return '<span class="spark">' + "".join(bars) + "</span>"


def render_html(rows, qoqs, generated):
    import html as _h
    counts = {k: sum(1 for r in rows if r["tier"] == k) for k in TIERS}
    cards = "".join(
        f'<div class="card {TIER_BADGE[k]}"><div class="n">{counts[k]}</div>'
        f'<div class="l">{TIERS[k][0].split(" · ")[0].split(" -")[0]}</div></div>'
        for k in ["tier1", "tier2", "fired", "artifact"])

    trs = []
    for m in rows:
        t = m["tier"]
        label = TIERS[t][0]
        nvda = " title=\"today's NVDA: leverage already cashed in\"" if m["tk"] == "NVDA" else ""
        trs.append(f"""<tr class="t-{t}">
      <td class="ticker"{nvda}>{m['tk']}</td>
      <td class="score-cell"><span class="score" data-sort="{m['score']}">{m['score']:.0f}</span></td>
      <td data-sort="{ {'tier1':5,'tier2':4,'fired':3,'watch':2,'lowmoat':1,'artifact':0}[t] }">
          <span class="badge {TIER_BADGE[t]}" title="{_h.escape(TIERS[t][1])}">{_h.escape(label)}</span></td>
      <td class="num">{m['revB']:.1f}</td>
      <td class="num">{m['gm']:.1f}</td>
      <td class="num">{m['nm']:.1f}</td>
      <td class="num">{m['gap']:.1f}</td>
      <td class="num">{m['rd']:.1f}</td>
      <td class="num">{m['gNow']:.0f}</td>
      <td class="num">{m['accel']:+.0f}</td>
    </tr>""")

    qcards = []
    for q in qoqs:
        revs = [x["rev"] for x in q["quarters"]]
        nms = [x["nm"] for x in q["quarters"]]
        last_qoq = next((x["qoq"] for x in reversed(q["quarters"]) if x["qoq"] is not None), None)
        vtags = "".join(f'<span class="vtag">{_h.escape(v)}</span>' for v in q["verdict"])
        good = q["verdict"][0].startswith("QoQ positive") and "stair-steps" in q["verdict"][2]
        qcards.append(f"""<div class="qcard {'ok' if good else ''}">
      <div class="qhead"><b>{q['tk']}</b><span class="qoq">{f'+{last_qoq:.1f}% QoQ' if last_qoq else ''}</span></div>
      <div class="qrow"><span class="qlab">Rev</span>{_spark(revs)}</div>
      <div class="qrow"><span class="qlab">Net&nbsp;M%</span>{_spark(nms, neg_ok=True)}
           <span class="qnm">{nms[0]:.0f}→{nms[-1]:.0f}%</span></div>
      <div class="vtags">{vtags}</div>
    </div>""")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Next-NVIDIA Fundamental Screen</title>
<style>
  :root {{ --bg:#0d1117; --panel:#161b22; --border:#30363d; --text:#e6edf3; --muted:#8b949e;
           --bull:#2ea043; --bear:#f85149; --watch:#d29922; --mixed:#6e7681; --fired:#388bfd; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--text);
          font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }}
  header {{ padding:24px 32px; border-bottom:1px solid var(--border); }}
  h1 {{ margin:0 0 4px; font-size:22px; }} h2 {{ font-size:15px; margin:28px 0 12px; }}
  .sub {{ color:var(--muted); font-size:13px; max-width:900px; line-height:1.5; }}
  .cards {{ display:flex; gap:16px; padding:20px 32px; flex-wrap:wrap; }}
  .card {{ background:var(--panel); border:1px solid var(--border); border-radius:10px;
           padding:16px 20px; min-width:130px; }}
  .card .n {{ font-size:28px; font-weight:700; }}
  .card .l {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.5px; }}
  .card.bull .n {{ color:var(--bull); }} .card.watch .n {{ color:var(--watch); }}
  .card.bear .n {{ color:var(--bear); }} .card.fired .n {{ color:var(--fired); }}
  .wrap {{ padding:0 32px 40px; }}
  table {{ width:100%; border-collapse:collapse; background:var(--panel);
           border:1px solid var(--border); border-radius:10px; overflow:hidden; }}
  th,td {{ padding:9px 13px; text-align:left; font-size:13px; border-bottom:1px solid var(--border); }}
  th {{ color:var(--muted); font-weight:600; text-transform:uppercase; font-size:11px;
        letter-spacing:.5px; cursor:pointer; user-select:none; }}
  th:hover {{ color:var(--text); }} tr:last-child td {{ border-bottom:none; }}
  td.ticker {{ font-weight:700; }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  tr.t-tier1 {{ background:rgba(46,160,67,.07); }}
  tr.t-tier2 {{ background:rgba(210,153,34,.05); }}
  tr.t-artifact td {{ opacity:.55; }}
  .score {{ display:inline-block; min-width:30px; text-align:center; padding:3px 7px; border-radius:6px;
            font-weight:800; font-variant-numeric:tabular-nums; background:rgba(110,118,129,.2); }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:6px; font-size:11px; font-weight:700;
            cursor:help; white-space:nowrap; }}
  .badge.bull {{ background:var(--bull); color:#051b0c; }}
  .badge.watch {{ background:rgba(210,153,34,.22); color:var(--watch); }}
  .badge.fired {{ background:rgba(56,139,253,.2); color:var(--fired); }}
  .badge.bear {{ background:rgba(248,81,73,.18); color:var(--bear); }}
  .badge.mixed {{ background:rgba(110,118,129,.18); color:var(--mixed); }}
  .qgrid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:14px; }}
  .qcard {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:14px 16px; }}
  .qcard.ok {{ border-color:var(--bull); box-shadow:0 0 0 1px rgba(46,160,67,.3); }}
  .qhead {{ display:flex; justify-content:space-between; align-items:baseline; margin-bottom:10px; }}
  .qhead b {{ font-size:16px; }} .qoq {{ color:var(--bull); font-size:12px; font-variant-numeric:tabular-nums; }}
  .qrow {{ display:flex; align-items:flex-end; gap:8px; margin:6px 0; }}
  .qlab {{ color:var(--muted); font-size:11px; width:42px; }}
  .qnm {{ color:var(--muted); font-size:11px; font-variant-numeric:tabular-nums; }}
  .spark {{ display:inline-flex; align-items:flex-end; gap:3px; height:28px; }}
  .sp {{ display:inline-block; width:9px; background:var(--bull); border-radius:2px 2px 0 0; }}
  .sp.dn {{ background:var(--bear); }}
  .vtags {{ margin-top:10px; display:flex; flex-wrap:wrap; gap:5px; }}
  .vtag {{ font-size:10px; color:var(--muted); border:1px solid var(--border); border-radius:5px; padding:2px 6px; }}
  footer {{ color:var(--muted); font-size:12px; padding:0 32px 36px; line-height:1.6; max-width:900px; }}
</style></head><body>
<header>
  <h1>🔎 Next-NVIDIA Fundamental Screen</h1>
  <div class="sub">S&amp;P 500 + Nasdaq 100 · Generated {generated:%Y-%m-%d %H:%M} · {len(rows)} names ranked
  against NVDA's pre-explosion (FY2023) fingerprint: a high gross margin already in place (the moat),
  heavy R&amp;D, accelerating revenue, and a wide gross-to-net gap (dormant operating leverage).
  <b>A matching fingerprint is necessary, not sufficient.</b></div>
</header>
<div class="cards">{cards}</div>
<div class="wrap">
  <h2>Full ranked screen <span style="color:var(--muted);font-weight:400">— click any header to re-sort</span></h2>
  <table id="scan"><thead><tr>
    <th onclick="sortBy(0,'s')">Ticker</th>
    <th onclick="sortBy(1,'d')" title="0-100 similarity to NVDA's FY2023 fingerprint">Score ▾</th>
    <th onclick="sortBy(2,'d')" title="Bucket by financial shape, not GICS sector">Tier</th>
    <th onclick="sortBy(3,'n')" title="Trailing revenue, $B">Rev $B</th>
    <th onclick="sortBy(4,'n')" title="Gross margin = pricing power / moat">GM %</th>
    <th onclick="sortBy(5,'n')" title="Net margin">NM %</th>
    <th onclick="sortBy(6,'n')" title="Gross minus net = dormant operating leverage">Gap</th>
    <th onclick="sortBy(7,'n')" title="R&amp;D as % of revenue">R&amp;D %</th>
    <th onclick="sortBy(8,'n')" title="Latest annual revenue growth YoY">Rev YoY</th>
    <th onclick="sortBy(9,'n')" title="Change in YoY growth vs prior year (+ = accelerating)">Accel</th>
  </tr></thead><tbody>{''.join(trs)}</tbody></table>

  <h2>QoQ staircase — top finalists <span style="color:var(--muted);font-weight:400">— green border = real sequential inflection</span></h2>
  <div class="qgrid">{''.join(qcards)}</div>
</div>
<footer>
  <b>How to read it:</b> Tier 1 = moat + <i>proven</i>, still-low net margin (NVDA-FY2023 shape).
  Tier 2 = moat + growth but unproven (low/negative net) leverage. "Already fired" = leverage realized,
  less upside. "Screen artifact" = clinical-stage biotech / cash-burner that matches mechanically — ignore.
  The QoQ staircase separates a real, sequential inflection from one lumpy year.
  <br>Not investment advice — research/educational use only. Concentrate the research, diversify the dollars.
</footer>
<script>
  let dir=1;
  function sortBy(col,type){{
    const tb=document.querySelector('#scan tbody'); const rows=[...tb.rows]; dir=-dir;
    rows.sort((a,b)=>{{
      let x,y;
      if(type==='d'){{ x=+a.cells[col].querySelector('[data-sort]').dataset.sort;
                       y=+b.cells[col].querySelector('[data-sort]').dataset.sort; }}
      else if(type==='n'){{ x=parseFloat(a.cells[col].innerText.replace(/[^0-9.\\-]/g,''))||0;
                            y=parseFloat(b.cells[col].innerText.replace(/[^0-9.\\-]/g,''))||0; }}
      else {{ x=a.cells[col].innerText.trim(); y=b.cells[col].innerText.trim(); }}
      return x>y?dir:x<y?-dir:0;
    }});
    rows.forEach(r=>tb.appendChild(r));
  }}
</script>
</body></html>"""


def run(full=False, tickers=None, weekly=False):
    """Score the universe, write next_nvidia_screen.csv + next_nvidia_dashboard.html.

    `tickers` lets run_daily pass the already-fetched universe so we don't
    re-scrape Wikipedia. Fundamentals are still pulled per-ticker (yfinance has
    no batch endpoint for financials), so the full run takes a few minutes.

    `weekly=True` (the cloud daily job) only does the full scan on Mondays;
    other weekdays it republishes last Monday's dashboard from Pages.
    """
    if weekly and not _is_refresh_day():
        print("Not a refresh day (weekly cadence) — republishing last Monday's screen.")
        if _republish_from_pages():
            print("Republished next_nvidia_dashboard.html + CSV from Pages.")
            return None
        print("Pages copy unavailable — falling back to a full scan.")

    if tickers is not None:
        basket = sorted(set(tickers))
    elif full:
        basket = get_basket()
        if not basket:
            print("Could not load universe; using basket.")
            basket = None
    else:
        basket = None
    if basket is None:
        basket = ["NVDA", "AMD", "AVGO", "PLTR", "SMCI", "ARM", "VRT", "CRWD",
                  "SNOW", "NET", "DDOG", "MDB", "ANET", "MRVL", "TSM", "NOW",
                  "PANW", "ASML", "CDNS", "SNPS"]

    print(f"Scoring {len(basket)} tickers against the NVDA-FY2023 fingerprint...\n")
    rows = []
    for i, tk in enumerate(basket, 1):
        m = metrics(tk)
        if m:
            m["score"] = score(m)
            rows.append(m)
        if len(basket) > 50 and i % 25 == 0:
            print(f"  ...{i}/{len(basket)}")
            time.sleep(0.3)

    rows.sort(key=lambda r: r["score"], reverse=True)
    for m in rows:
        m["tier"] = classify(m)

    hdr = f'{"TK":<6}{"Score":>6}{"Tier":>10}{"RevB":>7}{"GM%":>7}{"NM%":>7}{"Gap":>7}{"RD%":>7}{"gYoY":>7}{"accel":>7}'
    print("\n" + hdr)
    print("-" * len(hdr))
    for m in rows[:60]:
        print(f'{m["tk"]:<6}{m["score"]:>6.0f}{m["tier"]:>10}{m["revB"]:>7.1f}{m["gm"]:>7.1f}'
              f'{m["nm"]:>7.1f}{m["gap"]:>7.1f}{m["rd"]:>7.1f}{m["gNow"]:>7.0f}{m["accel"]:>+7.0f}')

    # --- CSV ---
    import csv
    from datetime import datetime
    with open("next_nvidia_screen.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "score", "tier", "rev_B", "gross_margin", "net_margin",
                    "gap", "rd_pct", "rev_yoy", "accel"])
        for m in rows:
            w.writerow([m["tk"], m["score"], m["tier"], round(m["revB"], 2), round(m["gm"], 1),
                        round(m["nm"], 1), round(m["gap"], 1), round(m["rd"], 1),
                        round(m["gNow"], 1), round(m["accel"], 1)])
    print("\nWrote next_nvidia_screen.csv")

    # --- QoQ for the top finalists (skip artifacts/low-moat) ---
    finalists = [m for m in rows if m["tier"] in ("tier1", "tier2", "fired")][:18]
    print(f"Pulling QoQ trends for {len(finalists)} finalists...")
    qoqs = []
    for m in finalists:
        q = qoq_data(m["tk"])
        if q:
            qoqs.append(q)

    # --- HTML dashboard ---
    htmlout = render_html(rows, qoqs, datetime.now())
    with open("next_nvidia_dashboard.html", "w") as f:
        f.write(htmlout)
    print("Wrote next_nvidia_dashboard.html")
    return rows


def main(tickers=None, weekly=False):
    return run(full="--full" in sys.argv[1:], tickers=tickers, weekly=weekly)


if __name__ == "__main__":
    main()
