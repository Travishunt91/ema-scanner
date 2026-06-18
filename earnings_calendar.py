#!/usr/bin/env python3
"""
Dedicated earnings calendar via Nasdaq's public API (free, no key).

Builds a {ticker: 'YYYY-MM-DD'} map of upcoming earnings dates so the scanners
can warn before you hold a single name through its report. Results are cached to
earnings_cache.json for the day to avoid hammering the API on repeated runs.
Falls back to yfinance per-ticker if Nasdaq is unreachable.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from datetime import date, datetime, timedelta

NASDAQ_URL = "https://api.nasdaq.com/api/calendar/earnings?date={date}"
CACHE_FILE = "earnings_cache.json"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/",
}


def _norm(sym: str) -> str:
    return str(sym).strip().upper().replace(".", "-")


def fetch_day(day: str) -> dict[str, str]:
    """All tickers reporting on `day` (YYYY-MM-DD) -> {ticker: day}."""
    req = urllib.request.Request(NASDAQ_URL.format(date=day), headers=HEADERS)
    with urllib.request.urlopen(req, timeout=20) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    rows = ((payload or {}).get("data") or {}).get("rows") or []
    return {_norm(r["symbol"]): day for r in rows if r.get("symbol")}


def _from_nasdaq(days: int) -> dict[str, str]:
    out: dict[str, str] = {}
    today = date.today()
    for i in range(days + 1):
        d = today + timedelta(days=i)
        if d.weekday() >= 5:      # skip weekends (markets closed, empty anyway)
            continue
        try:
            for sym, ds in fetch_day(d.isoformat()).items():
                out.setdefault(sym, ds)   # keep the earliest upcoming date
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"  ! earnings fetch failed for {d}: {exc}")
        time.sleep(0.15)  # be polite over a wide date window
    return out


def _from_yfinance(tickers: list[str], days: int) -> dict[str, str]:
    import yfinance as yf  # local import; only used on fallback
    out: dict[str, str] = {}
    horizon = date.today() + timedelta(days=days)
    for t in tickers or []:
        try:
            ed = yf.Ticker(t).get_earnings_dates(limit=8)
            if ed is None or ed.empty:
                continue
            fut = [d for d in ed.index.date if date.today() <= d <= horizon]
            if fut:
                out[_norm(t)] = min(fut).isoformat()
        except Exception:
            continue
    return out


def upcoming_earnings(days: int = 10, tickers: list[str] | None = None,
                      use_cache: bool = True) -> dict[str, str]:
    """Map of {ticker: next earnings date} within `days`, filtered to `tickers`."""
    today_key = date.today().isoformat()
    if use_cache and os.path.exists(CACHE_FILE):
        try:
            cached = json.load(open(CACHE_FILE))
            if cached.get("date") == today_key and cached.get("days", 0) >= days:
                full = cached["map"]
                return {t: full[t] for t in tickers if t in full} if tickers else full
        except Exception:
            pass

    full = _from_nasdaq(days)
    if not full and tickers:          # Nasdaq blocked -> fall back
        print("  ! Nasdaq calendar empty; falling back to yfinance.")
        full = _from_yfinance(tickers, days)

    if use_cache and full:
        try:
            json.dump({"date": today_key, "days": days, "map": full},
                      open(CACHE_FILE, "w"))
        except Exception:
            pass

    return {t: full[t] for t in tickers if t in full} if tickers else full


def warning(ticker: str, emap: dict[str, str]) -> str:
    """Render a warning string if `ticker` reports soon, else ''."""
    ds = emap.get(_norm(ticker))
    if not ds:
        return ""
    days = (datetime.fromisoformat(ds).date() - date.today()).days
    when = datetime.fromisoformat(ds).strftime("%b %d")
    return f"⚠ ER {when} ({days}d)" if days >= 0 else ""


if __name__ == "__main__":
    m = upcoming_earnings(days=10)
    print(f"{len(m)} tickers reporting in the next 10 days.")
    for t, d in sorted(m.items(), key=lambda kv: kv[1])[:25]:
        print(f"  {t:6} {d}")
