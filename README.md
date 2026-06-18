# EMA Stack Scanner

Scans the **S&P 500 + Nasdaq 100** universe for bullish EMA-stack setups and writes a
sortable HTML dashboard.

## What it computes

EMAs **9 / 21 / 33 / 50 / 200** on both **daily** and **weekly** timeframes.

| Check | Logic |
|-------|-------|
| Weekly stack | `FULL BULL` (9>21>33>50>200), `FULL BEAR` (reverse), else `MIXED` |
| Daily stack | full-bull true/false |
| BUY signal | close > daily EMA9 |
| SELL signal | close < daily EMA21 |
| % from EMA9 | distance of price to daily EMA9 |
| **STRONG BUY** | weekly FULL BULL **and** daily BUY fired |
| **WATCH** | weekly FULL BULL **and** price within 3% of EMA9/EMA21 |
| **NO SIGNAL** | everything else |

The dashboard (`ema_dashboard.html`) is sorted strongest-signal-first; every column header is clickable to re-sort.

## Run

```bash
pip install yfinance pandas lxml
python3 ema_scanner.py
open ema_dashboard.html
```

Takes a few minutes (batched yfinance downloads). Ticker lists are pulled live from
Wikipedia, with a small built-in fallback if that fails.

## Notes

- **Lookback:** daily = 1 year (per spec). Weekly is pulled at 5 years so the weekly
  EMA-200 has enough bars to be meaningful (a 200-period EMA on weekly bars needs ~4 years).
- Not investment advice — research/educational use only.

## Schedule daily at 4:00pm ET (after the close)

`launchd` (macOS) — runs at 16:05 America/New_York. Adjust the path and, if your machine
is not on ET, the hour. Save as `~/Library/LaunchAgents/com.ema.scanner.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.ema.scanner</string>
  <key>ProgramArguments</key>
  <array>
    <string>/opt/homebrew/bin/python3</string>
    <string>/Users/travishunt/Claude/ema-scanner/ema_scanner.py</string>
  </array>
  <key>WorkingDirectory</key><string>/Users/travishunt/Claude/ema-scanner</string>
  <key>StartCalendarInterval</key>
  <dict><key>Hour</key><integer>16</integer><key>Minute</key><integer>5</integer></dict>
  <key>StandardOutPath</key><string>/tmp/ema-scanner.log</string>
  <key>StandardErrorPath</key><string>/tmp/ema-scanner.err</string>
</dict></plist>
```

```bash
launchctl load ~/Library/LaunchAgents/com.ema.scanner.plist
```

Or with cron (weekdays 4:05pm local): `5 16 * * 1-5 cd /Users/travishunt/Claude/ema-scanner && /opt/homebrew/bin/python3 ema_scanner.py`
