#!/usr/bin/env python3
"""
Daily EMA Stack Scanner
=======================

Scans the combined S&P 500 + Nasdaq 100 universe for bullish EMA-stack setups
across weekly and daily timeframes, then writes a sortable HTML dashboard.

EMAs computed (both timeframes): 9, 21, 33, 50, 200

Signals
-------
WEEKLY : FULL BULL (9>21>33>50>200), FULL BEAR (reverse), else MIXED
DAILY  : full-bull stack, close>EMA9 (BUY), close<EMA21 (SELL), % distance to EMA9
COMBINED:
    STRONG BUY  = weekly FULL BULL + daily BUY fired
    WATCH       = weekly FULL BULL + price pulling back within 3% of EMA9/EMA21
    NO SIGNAL   = everything else

Run once daily after the close (4pm ET). See README for scheduling.
"""

from __future__ import annotations

import html
import io
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
import yfinance as yf

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
EMA_PERIODS = [9, 21, 33, 50, 200]

# Daily lookback honors the 1-year spec. Weekly needs many more bars for a
# meaningful EMA-200 (200 weekly bars ~= 4 years), so we pull extra weekly
# history — the daily timeframe stays at the requested 1-year window.
DAILY_PERIOD = "1y"
WEEKLY_PERIOD = "5y"
MONTHLY_PERIOD = "max"   # full history; monthly EMA200 needs ~17y to be valid

PULLBACK_PCT = 3.0  # "WATCH" proximity threshold to EMA9/EMA21

# Scale-out plan (from backtest.py / winrate_lab.py): banking ~1/3 of the position
# at +5% and trailing the rest with the EMA50 stop lifts win rate ~29%->38% while
# keeping ~63% of the edge — the best comfort-for-cost trade.
SCALE_OUT_PCT = 5.0
SCALE_OUT_FRAC = "⅓"

# Score: % above EMA9 at which the proximity reward decays to 0. A fresh breakout
# sitting right at the EMA9 scores the full 3 pts; one extended this far scores 0.
PROX_MAX_EXT = 8.0

OUTPUT_HTML = "ema_dashboard.html"

# Signal ordering for the sorted dashboard (lower = stronger / higher up)
SIGNAL_RANK = {"STRONG BUY": 0, "WATCH": 1, "NO SIGNAL": 2}


# --------------------------------------------------------------------------- #
# Universe
# --------------------------------------------------------------------------- #
def _clean_symbols(symbols) -> list[str]:
    """Normalize ticker symbols for yfinance (BRK.B -> BRK-B)."""
    out = []
    for s in symbols:
        s = str(s).strip().upper()
        if not s or s == "NAN":
            continue
        out.append(s.replace(".", "-"))
    return out


# Populated as a side effect of get_universe(): {ticker: company name}
TICKER_NAMES: dict[str, str] = {}


def get_universe() -> list[str]:
    """Fetch S&P 500 + Nasdaq 100 tickers from Wikipedia (with fallback).

    Also fills the module-level TICKER_NAMES map with company names.
    """
    tickers: set[str] = set()

    # (url, ticker column, name column)
    sources = [
        ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol", "Security"),
        ("https://en.wikipedia.org/wiki/Nasdaq-100", "Ticker", "Company"),
    ]
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
        )
    }
    for url, col, name_col in sources:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                html = resp.read().decode("utf-8")
            tables = pd.read_html(io.StringIO(html))
            for tbl in tables:
                if col in tbl.columns:
                    tickers.update(_clean_symbols(tbl[col].tolist()))
                    if name_col in tbl.columns:
                        for sym, name in zip(tbl[col], tbl[name_col]):
                            sym = str(sym).strip().upper().replace(".", "-")
                            if sym and sym != "NAN":
                                TICKER_NAMES.setdefault(sym, str(name).strip())
                    break
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"  ! Could not load {url}: {exc}", file=sys.stderr)

    if not tickers:
        print("  ! Falling back to a small built-in watchlist.", file=sys.stderr)
        tickers.update(
            _clean_symbols(
                [
                    "AAPL", "MSFT", "NVDA", "AMZN", "GOOG", "META", "AVGO",
                    "MRVL", "SMCI", "INTC", "AMBA", "ALGM", "ON", "BE",
                    "BB", "TSLA", "AMD",
                ]
            )
        )

    return sorted(tickers)


# --------------------------------------------------------------------------- #
# EMA / analysis
# --------------------------------------------------------------------------- #
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def stack_label(emas: dict[int, float]) -> str:
    """Return FULL BULL / FULL BEAR / MIXED for an EMA value set."""
    vals = [emas[p] for p in EMA_PERIODS]
    if any(pd.isna(v) for v in vals):
        return "MIXED"
    if all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)):
        return "FULL BULL"
    if all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)):
        return "FULL BEAR"
    return "MIXED"


def _ordered_pairs(emas: dict[int, float]) -> int:
    """How many of the 4 adjacent EMA orderings hold for a bull stack (0-4)."""
    if not emas:
        return 0
    vals = [emas[p] for p in EMA_PERIODS]
    if any(pd.isna(v) for v in vals):
        return 0
    return sum(1 for i in range(len(vals) - 1) if vals[i] > vals[i + 1])


def compute_score(res: "Result") -> tuple[float, dict]:
    """Composite 0-10 score rolling up every indicator. Returns (score, parts)."""
    parts: dict[str, float] = {}

    # Weekly trend (max 3) — graded by how much of the stack is bullish
    parts["Weekly trend"] = round(_ordered_pairs(res.weekly_emas) / 4 * 3, 2)

    # Daily trend (max 2)
    parts["Daily trend"] = round(_ordered_pairs(res.daily_emas) / 4 * 2, 2)

    # Proximity to EMA9 (max 3) — the freshest, least-extended breakouts score
    # highest. Reward decays linearly from 3 (price right at EMA9) to 0 once price
    # is PROX_MAX_EXT% above it (chasing). Below EMA9 = no breakout = 0.
    p = res.pct_from_ema9
    if res.buy_signal and not pd.isna(p) and p >= 0:
        parts["Proximity to EMA9"] = round(max(0.0, 3.0 * (1 - p / PROX_MAX_EXT)), 2)
    else:
        parts["Proximity to EMA9"] = 0.0

    # Daily EMA9 break freshness (max 1)
    db = res.daily_break_bars_ago
    parts["Fresh break"] = 1.0 if db is not None and db <= 1 else 0.5 if db is not None and db <= 5 else 0.0

    # Above weekly EMA9 (max 1)
    wb = res.weekly_break_bars_ago
    parts["Weekly >EMA9"] = 1.0 if wb is not None and wb <= 1 else 0.75 if wb is not None else 0.0

    score = round(sum(parts.values()), 1)
    return min(score, 10.0), parts


@dataclass
class Result:
    ticker: str
    price: float = float("nan")
    weekly_label: str = "MIXED"
    daily_full_bull: bool = False
    buy_signal: bool = False
    sell_signal: bool = False
    pct_from_ema9: float = float("nan")
    pct_from_ema21: float = float("nan")
    combined: str = "NO SIGNAL"
    error: str = ""
    daily_emas: dict = field(default_factory=dict)
    weekly_emas: dict = field(default_factory=dict)
    monthly_label: str = "MIXED"
    monthly_emas: dict = field(default_factory=dict)
    # Most recent upward cross of price through EMA9 (None = currently below EMA9)
    daily_break_bars_ago: int | None = None
    daily_break_date: object = None
    weekly_break_bars_ago: int | None = None
    weekly_break_date: object = None
    score: float = 0.0
    score_parts: dict = field(default_factory=dict)
    entry_low: float = float("nan")    # daily EMA21 (lower bound of buy zone)
    entry_high: float = float("nan")   # daily EMA9 (upper bound / trigger)
    entry_stop: float = float("nan")   # suggested stop (below EMA50)
    entry_status: str = ""             # IN ZONE / PULLBACK / BELOW
    scaleout_basis: float = float("nan")   # assumed entry price the target hangs off
    scaleout_target: float = float("nan")  # price to bank the scale-out leg


def _last_emas(close: pd.Series) -> dict[int, float]:
    return {p: float(ema(close, p).iloc[-1]) for p in EMA_PERIODS}


def ema9_break(close: pd.Series):
    """When did price most recently cross UP through EMA9 and stay above?

    Returns (bars_ago, date) for the start of the current above-EMA9 run, or
    (None, None) if the latest close is at/below EMA9 (no active break).
    bars_ago == 0 means the break happened on the most recent bar.
    """
    e9 = ema(close, 9)
    above = close > e9
    if not bool(above.iloc[-1]):
        return None, None
    # Walk back from the last bar to the first bar of the current above-run.
    i = len(above) - 1
    while i > 0 and bool(above.iloc[i - 1]):
        i -= 1
    bars_ago = (len(above) - 1) - i
    return bars_ago, close.index[i]


def analyze(ticker: str, daily: pd.DataFrame, weekly: pd.DataFrame,
            monthly: pd.DataFrame | None = None) -> Result:
    res = Result(ticker=ticker)

    if daily is None or daily.empty or "Close" in daily and daily["Close"].dropna().empty:
        res.error = "no daily data"
        return res

    dclose = daily["Close"].dropna()
    wclose = weekly["Close"].dropna() if weekly is not None and not weekly.empty else pd.Series(dtype=float)
    mclose = monthly["Close"].dropna() if monthly is not None and not monthly.empty else pd.Series(dtype=float)

    if len(dclose) < 50:
        res.error = "insufficient daily history"
        return res

    res.price = float(dclose.iloc[-1])

    # Daily
    d = _last_emas(dclose)
    res.daily_emas = d
    res.daily_full_bull = stack_label(d) == "FULL BULL"
    res.buy_signal = res.price > d[9]
    res.sell_signal = res.price < d[21]
    res.pct_from_ema9 = (res.price - d[9]) / d[9] * 100.0
    res.pct_from_ema21 = (res.price - d[21]) / d[21] * 100.0
    res.daily_break_bars_ago, res.daily_break_date = ema9_break(dclose)

    # Entry zone: the dynamic-support band between daily EMA9 and EMA21.
    res.entry_low = min(d[9], d[21])
    res.entry_high = max(d[9], d[21])
    res.entry_stop = d[50]  # stop a bit under the EMA50, deeper structural support
    if res.price > res.entry_high:
        res.entry_status = "PULLBACK"   # extended above; wait for a dip into the band
    elif res.price >= res.entry_low:
        res.entry_status = "IN ZONE"    # already sitting in the buy zone
    else:
        res.entry_status = "BELOW"      # under support; no clean long entry

    # Scale-out target hangs off the realistic entry: current price if you can buy
    # in the zone now, else the EMA9 you'd buy on a pullback. None if no entry.
    if res.entry_status == "IN ZONE":
        res.scaleout_basis = res.price
    elif res.entry_status == "PULLBACK":
        res.scaleout_basis = res.entry_high  # EMA9 in a bull stack
    if not pd.isna(res.scaleout_basis):
        res.scaleout_target = res.scaleout_basis * (1 + SCALE_OUT_PCT / 100)

    # Weekly
    if len(wclose) >= 5:
        w = _last_emas(wclose)
        res.weekly_emas = w
        res.weekly_label = stack_label(w)
        res.weekly_break_bars_ago, res.weekly_break_date = ema9_break(wclose)

    # Monthly (higher-timeframe regime; EMA200 needs ~17y so it's often partial)
    if len(mclose) >= 5:
        m = _last_emas(mclose)
        res.monthly_emas = m
        res.monthly_label = stack_label(m)

    # Combined
    weekly_bull = res.weekly_label == "FULL BULL"
    near_ema = (
        abs(res.pct_from_ema9) <= PULLBACK_PCT
        or abs(res.pct_from_ema21) <= PULLBACK_PCT
    )
    if weekly_bull and res.buy_signal:
        res.combined = "STRONG BUY"
    elif weekly_bull and near_ema:
        res.combined = "WATCH"
    else:
        res.combined = "NO SIGNAL"

    res.score, res.score_parts = compute_score(res)
    return res


# --------------------------------------------------------------------------- #
# Data download
# --------------------------------------------------------------------------- #
def download(tickers: list[str], period: str, interval: str) -> dict[str, pd.DataFrame]:
    """Batch-download OHLC, returning {ticker: DataFrame}."""
    frames: dict[str, pd.DataFrame] = {}
    batch_size = 100
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        print(f"    downloading {interval} {i + 1}-{i + len(batch)} of {len(tickers)} ...")
        data = yf.download(
            batch,
            period=period,
            interval=interval,
            group_by="ticker",
            # Split-adjusted but NOT dividend-adjusted, to match what charting
            # tools (e.g. TradingView) show. Dividend-adjusting drags historical
            # prices down and can fabricate an EMA stack on high-yield names.
            auto_adjust=False,
            threads=True,
            progress=False,
        )
        for t in batch:
            try:
                if len(batch) == 1:
                    frames[t] = data
                else:
                    frames[t] = data[t]
            except Exception:
                frames[t] = pd.DataFrame()
        time.sleep(0.5)  # be polite to the API
    return frames


# --------------------------------------------------------------------------- #
# HTML dashboard
# --------------------------------------------------------------------------- #
def _fmt_pct(v: float) -> str:
    if pd.isna(v):
        return "—"
    return f"{v:+.2f}%"


def _fmt_break(bars_ago, date, unit: str) -> tuple[str, int]:
    """Render the EMA9 break freshness cell -> (html, sort_key).

    sort_key is bars_ago for sorting (fresher first); 9999 for 'below'.
    """
    if bars_ago is None:
        return '<span class="badge mixed">below</span>', 9999
    when = f"{date:%b %d}" if date is not None else ""
    if bars_ago == 0:
        label, kind = ("TODAY" if unit == "d" else "THIS WK"), "fresh"
    elif bars_ago == 1:
        label, kind = f"1{unit} ago", "fresh"
    else:
        label, kind = f"{bars_ago}{unit} ago", "bull"
    sub = f' <span class="when">{when}</span>' if when else ""
    return f'<span class="badge {kind}">{label}</span>{sub}', bars_ago


def _fmt_entry(r) -> str:
    """Render the entry-zone cell: price band + status tag, stop in tooltip."""
    if pd.isna(r.entry_low) or pd.isna(r.entry_high):
        return "—"
    kind = {"IN ZONE": "fresh", "PULLBACK": "watch", "BELOW": "mixed"}[r.entry_status]
    tip = f"Buy zone EMA21–EMA9. Suggested stop below EMA50 ${r.entry_stop:,.2f}"
    band = f"${r.entry_low:,.2f}–${r.entry_high:,.2f}"
    return (
        f'<span class="badge {kind}" title="{tip}">{r.entry_status}</span>'
        f'<span class="band">{band}</span>'
    )


def _fmt_scaleout(r) -> str:
    """Render the scale-out target: bank a third at +5%, trail the rest."""
    if pd.isna(r.scaleout_target):
        return "—"
    basis_lbl = "from price" if r.entry_status == "IN ZONE" else "from EMA9"
    tip = (
        f"Sell {SCALE_OUT_FRAC} at +{SCALE_OUT_PCT:g}% (${r.scaleout_target:,.2f}, "
        f"{basis_lbl} ${r.scaleout_basis:,.2f}); trail the rest with the EMA50 stop. "
        f"Backtest: lifts win rate ~29%→38% while keeping ~63% of the edge."
    )
    return (
        f'<span class="badge bull" title="{tip}">${r.scaleout_target:,.2f}</span>'
        f'<span class="band">sell {SCALE_OUT_FRAC} @ +{SCALE_OUT_PCT:g}%</span>'
    )


def render_html(results: list[Result], generated: datetime) -> str:
    results = sorted(
        results,
        key=lambda r: (-r.score, SIGNAL_RANK.get(r.combined, 9)),
    )

    counts = {"STRONG BUY": 0, "WATCH": 0, "NO SIGNAL": 0}
    for r in results:
        counts[r.combined] = counts.get(r.combined, 0) + 1

    rows = []
    for r in results:
        if r.error:
            continue
        sig_class = {
            "STRONG BUY": "strong-buy",
            "WATCH": "watch",
            "NO SIGNAL": "no-signal",
        }[r.combined]

        def badge(text, kind):
            return f'<span class="badge {kind}">{text}</span>'

        stack_badge = {
            "FULL BULL": lambda: badge("FULL BULL", "bull"),
            "FULL BEAR": lambda: badge("FULL BEAR", "bear"),
            "MIXED": lambda: badge("MIXED", "mixed"),
        }
        monthly_badge = stack_badge[r.monthly_label]()
        weekly_badge = stack_badge[r.weekly_label]()

        daily_stack = badge("BULL", "bull") if r.daily_full_bull else badge("—", "mixed")
        buy = badge("BUY", "bull") if r.buy_signal else ""
        sell = badge("SELL", "bear") if r.sell_signal else ""
        daily_sig = (buy + " " + sell).strip() or "—"

        d_break_html, d_break_key = _fmt_break(r.daily_break_bars_ago, r.daily_break_date, "d")
        w_break_html, w_break_key = _fmt_break(r.weekly_break_bars_ago, r.weekly_break_date, "w")

        s_tier = "s-hi" if r.score >= 8 else "s-mid" if r.score >= 5 else "s-lo"
        s_tip = " · ".join(f"{k}: {v:g}" for k, v in r.score_parts.items())

        rows.append(
            f"""
      <tr class="{sig_class}" data-signal="{SIGNAL_RANK[r.combined]}">
        <td class="ticker">{r.ticker}<span class="coname">{html.escape(TICKER_NAMES.get(r.ticker, ""))}</span></td>
        <td class="signal">{badge(r.combined, sig_class)}</td>
        <td class="score-cell" data-sort="{r.score}"><span class="score {s_tier}" title="{s_tip}">{r.score:g}</span><span class="score-max">/10</span></td>
        <td>${r.price:,.2f}</td>
        <td class="entry-cell">{_fmt_entry(r)}</td>
        <td class="entry-cell">{_fmt_scaleout(r)}</td>
        <td>{monthly_badge}</td>
        <td>{weekly_badge}</td>
        <td>{daily_stack}</td>
        <td>{daily_sig}</td>
        <td data-sort="{d_break_key}">{d_break_html}</td>
        <td data-sort="{w_break_key}">{w_break_html}</td>
        <td class="num">{_fmt_pct(r.pct_from_ema9)}</td>
        <td class="num">{_fmt_pct(r.pct_from_ema21)}</td>
      </tr>"""
        )

    errored = [r for r in results if r.error]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EMA Stack Scanner</title>
<style>
  :root {{
    --bg: #0d1117; --panel: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e;
    --bull: #2ea043; --bear: #f85149; --watch: #d29922; --mixed: #6e7681;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--bg); color: var(--text);
         font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
  header {{ padding: 24px 32px; border-bottom: 1px solid var(--border); }}
  h1 {{ margin: 0 0 4px; font-size: 22px; }}
  .sub {{ color: var(--muted); font-size: 13px; }}
  .cards {{ display: flex; gap: 16px; padding: 20px 32px; flex-wrap: wrap; }}
  .card {{ background: var(--panel); border: 1px solid var(--border);
          border-radius: 10px; padding: 16px 20px; min-width: 140px; }}
  .card .n {{ font-size: 28px; font-weight: 700; }}
  .card .l {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .5px; }}
  .card.sb .n {{ color: var(--bull); }}
  .card.w .n {{ color: var(--watch); }}
  .wrap {{ padding: 0 32px 40px; }}
  table {{ width: 100%; border-collapse: collapse; background: var(--panel);
          border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }}
  th, td {{ padding: 10px 14px; text-align: left; font-size: 13px;
           border-bottom: 1px solid var(--border); }}
  th {{ color: var(--muted); font-weight: 600; text-transform: uppercase;
       font-size: 11px; letter-spacing: .5px; cursor: pointer; user-select: none; }}
  th:hover {{ color: var(--text); }}
  tr:last-child td {{ border-bottom: none; }}
  tr.strong-buy {{ background: rgba(46,160,67,.08); }}
  tr.watch {{ background: rgba(210,153,34,.06); }}
  td.ticker {{ font-weight: 700; }}
  .coname {{ display: block; font-weight: 400; color: var(--muted); font-size: 11px;
            max-width: 180px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 6px;
           font-size: 11px; font-weight: 700; }}
  .badge.bull {{ background: rgba(46,160,67,.18); color: var(--bull); }}
  .badge.bear {{ background: rgba(248,81,73,.18); color: var(--bear); }}
  .badge.watch {{ background: rgba(210,153,34,.18); color: var(--watch); }}
  .badge.mixed {{ background: rgba(110,118,129,.18); color: var(--mixed); }}
  .badge.strong-buy {{ background: var(--bull); color: #051b0c; }}
  .badge.no-signal {{ background: rgba(110,118,129,.18); color: var(--mixed); }}
  .badge.fresh {{ background: var(--watch); color: #1b1303; animation: pulse 1.8s ease-in-out infinite; }}
  @keyframes pulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: .55; }} }}
  .when {{ color: var(--muted); font-size: 11px; margin-left: 4px; }}
  .score-cell {{ white-space: nowrap; }}
  .score {{ display: inline-block; min-width: 34px; text-align: center; padding: 3px 8px;
           border-radius: 6px; font-weight: 800; font-size: 14px; font-variant-numeric: tabular-nums; }}
  .score.s-hi {{ background: var(--bull); color: #051b0c; }}
  .score.s-mid {{ background: rgba(210,153,34,.22); color: var(--watch); }}
  .score.s-lo {{ background: rgba(110,118,129,.18); color: var(--mixed); }}
  .score-max {{ color: var(--muted); font-size: 11px; margin-left: 2px; }}
  .entry-cell {{ white-space: nowrap; }}
  .band {{ display: block; color: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; margin-top: 2px; }}
  footer {{ color: var(--muted); font-size: 12px; padding: 0 32px 32px; }}
</style>
</head>
<body>
<header>
  <h1>📈 EMA Stack Scanner</h1>
  <div class="sub">S&amp;P 500 + Nasdaq 100 &nbsp;·&nbsp; Generated {generated:%Y-%m-%d %H:%M %Z} &nbsp;·&nbsp; {len(rows)} symbols analyzed</div>
</header>

<div class="cards">
  <div class="card sb"><div class="n">{counts['STRONG BUY']}</div><div class="l">Strong Buy</div></div>
  <div class="card w"><div class="n">{counts['WATCH']}</div><div class="l">Watch</div></div>
  <div class="card"><div class="n">{counts['NO SIGNAL']}</div><div class="l">No Signal</div></div>
</div>

<div class="wrap">
  <table id="scan">
    <thead>
      <tr>
        <th onclick="sortBy(0,'s')">Ticker</th>
        <th onclick="sortBy(1,'sig')">Signal</th>
        <th onclick="sortBy(2,'d')" title="Composite 0-10 score across all indicators. Hover a score for the breakdown.">Score ▾</th>
        <th onclick="sortBy(3,'n')">Price</th>
        <th onclick="sortBy(4,'s')" title="Pullback buy zone (daily EMA21–EMA9). IN ZONE = price in the band now; PULLBACK = wait for a dip; BELOW = under support. Hover for stop.">Entry Zone</th>
        <th onclick="sortBy(5,'n')" title="Suggested scale-out: bank a third at +5% off your entry, trail the rest with the EMA50 stop. Backtest-derived comfort/edge sweet spot.">Scale-out</th>
        <th onclick="sortBy(6,'s')" title="Monthly EMA(9/21/33/50/200) stack — the highest-timeframe regime. EMA200 needs ~17y of history, so newer names may read MIXED.">Monthly Stack</th>
        <th onclick="sortBy(7,'s')">Weekly Stack</th>
        <th onclick="sortBy(8,'s')">Daily Stack</th>
        <th onclick="sortBy(9,'s')">Daily Signal</th>
        <th onclick="sortBy(10,'d')" title="Bars since price last crossed up through the daily EMA9 (TODAY = broke today)">Daily &gt;EMA9</th>
        <th onclick="sortBy(11,'d')" title="Bars since price last crossed up through the weekly EMA9">Weekly &gt;EMA9</th>
        <th onclick="sortBy(12,'n')">% vs EMA9</th>
        <th onclick="sortBy(13,'n')">% vs EMA21</th>
      </tr>
    </thead>
    <tbody>{''.join(rows)}
    </tbody>
  </table>
</div>

<footer>
  EMAs: 9 / 21 / 33 / 50 / 200 on daily &amp; weekly. WATCH threshold = within {PULLBACK_PCT:.0f}% of EMA9/EMA21.
  {f"{len(errored)} symbols skipped (no/insufficient data)." if errored else ""}
  <br>Not investment advice — for research and educational use only.
</footer>

<script>
  let dir = 1;
  function sortBy(col, type) {{
    const tb = document.querySelector('#scan tbody');
    const rows = [...tb.rows];
    dir = -dir;
    rows.sort((a, b) => {{
      let x = a.cells[col].innerText.trim();
      let y = b.cells[col].innerText.trim();
      if (type === 'sig') {{ x = +a.dataset.signal; y = +b.dataset.signal; }}
      else if (type === 'd') {{ x = +a.cells[col].dataset.sort; y = +b.cells[col].dataset.sort; }}
      else if (type === 'n') {{ x = parseFloat(x.replace(/[^0-9.\\-]/g,'')) || 0;
                                y = parseFloat(y.replace(/[^0-9.\\-]/g,'')) || 0; }}
      return x > y ? dir : x < y ? -dir : 0;
    }});
    rows.forEach(r => tb.appendChild(r));
  }}
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(tickers: list[str] | None = None,
         daily: dict | None = None) -> int:
    """Build the EMA dashboard. `tickers`/`daily` may be supplied by a shared
    runner (run_daily.py) to avoid re-fetching the universe and daily history."""
    print("EMA Stack Scanner")
    print("=================")
    print("[1/4] Building universe (S&P 500 + Nasdaq 100) ...")
    if tickers is None:
        tickers = get_universe()
    print(f"      {len(tickers)} unique tickers.")

    print("[2/4] Downloading daily history ...")
    if daily is None:
        daily = download(tickers, DAILY_PERIOD, "1d")
    else:
        print("      (reusing shared daily download)")
    print("[3/4] Downloading weekly + monthly history ...")
    weekly = download(tickers, WEEKLY_PERIOD, "1wk")
    monthly = download(tickers, MONTHLY_PERIOD, "1mo")

    print("[4/4] Analyzing ...")
    results = [analyze(t, daily.get(t), weekly.get(t), monthly.get(t)) for t in tickers]

    ok = [r for r in results if not r.error]
    sb = sum(r.combined == "STRONG BUY" for r in ok)
    w = sum(r.combined == "WATCH" for r in ok)
    print(f"      {len(ok)} analyzed · {sb} STRONG BUY · {w} WATCH")

    html = render_html(results, datetime.now().astimezone())
    with open(OUTPUT_HTML, "w") as f:
        f.write(html)
    print(f"\n✅ Dashboard written to {OUTPUT_HTML}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
