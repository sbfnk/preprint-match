#!/usr/bin/env python3
"""Weekly visitor-stats snapshot for preprints.epiforecasts.io.

Reads the analytics `hits` table and emits a Markdown report: this-week-vs-last-week,
recent trend, traffic-channel breakdown, top referrers and pages, regions and devices.

Usage:
    # Analyse a local copy of the DB
    python3 analytics_snapshot.py --db /tmp/analytics_live.db

    # Pull the live DB off the Fly volume first, then analyse
    python3 analytics_snapshot.py --pull

The --pull path shells out to flyctl and needs either FLY_API_TOKEN /
FLY_ACCESS_TOKEN in the environment, or a token in ~/.fly/config.yml.
"""
import argparse
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone

APP = "preprint-match"
REMOTE_DB = "/data/analytics.db"

# Referrer host -> channel. Checked as substrings, first match wins.
CHANNELS = [
    ("search", ("google", "bing", "duckduckgo", "yahoo", "ecosia", "qwant",
                "startpage", "baidu", "yandex", "search.brave", "sogou")),
    ("ai assistant", ("chatgpt", "perplexity", "copilot", "gemini", "claude",
                       "doubao", "yuanbao", "phind", "you.com", "poe.com",
                       "tencent", "feishu", "kimi")),
    ("social", ("linkedin", "bsky", "bluesky", "twitter", "t.co", "x.com",
                "mastodon", "facebook", "reddit", "slack", "teams", "office",
                "whatsapp", "telegram")),
    ("plagiarism check", ("turnitin", "ithenticate", "crossref", "scribbr",
                          "quillbot")),
    ("code/docs", ("github",)),
]


def classify(referrer):
    if not referrer:
        return "direct"
    r = referrer.lower()
    for name, hosts in CHANNELS:
        if any(h in r for h in hosts):
            return name
    return "other referral"


def pull_live_db(dest):
    """SFTP the live analytics DB off the Fly volume to `dest`."""
    env = dict(os.environ)
    if not env.get("FLY_API_TOKEN") and not env.get("FLY_ACCESS_TOKEN"):
        cfg = os.path.expanduser("~/.fly/config.yml")
        if os.path.exists(cfg):
            with open(cfg) as fh:
                m = re.search(r"access_token:\s*(\S+)", fh.read())
                if m:
                    env["FLY_ACCESS_TOKEN"] = m.group(1)
    cmd = ["flyctl", "ssh", "sftp", "get", REMOTE_DB, dest, "--app", APP]
    subprocess.run(cmd, check=True, env=env)
    return dest


def q(conn, sql, params=()):
    return conn.execute(sql, params).fetchall()


def bar(n, total, width=24):
    if not total:
        return ""
    filled = round(width * n / total)
    return "█" * filled + "·" * (width - filled)


def report(db_path):
    conn = sqlite3.connect(db_path)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    out = []
    p = out.append

    total_all, tmin, tmax = q(conn,
        "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM hits")[0]

    p(f"# Visitor snapshot — {now:%Y-%m-%d}")
    p("")
    p(f"Site: **preprints.epiforecasts.io** · "
      f"{total_all:,} hits on record ({tmin[:10]} → {tmax[:10]})")
    p("")

    # --- This week vs last week (rolling 7-day windows) ---
    def window(days_ago_start, days_ago_end):
        start = (now - timedelta(days=days_ago_start)).strftime("%Y-%m-%d %H:%M:%S")
        end = (now - timedelta(days=days_ago_end)).strftime("%Y-%m-%d %H:%M:%S")
        return q(conn,
            "SELECT COUNT(*) FROM hits WHERE timestamp >= ? AND timestamp < ?",
            (start, end))[0][0]

    this_week = window(7, 0)
    last_week = window(14, 7)
    prev_week = window(21, 14)
    if last_week:
        pct = 100.0 * (this_week - last_week) / last_week
        arrow = "▲" if pct >= 0 else "▼"
        change = f"{arrow} {pct:+.0f}% vs previous 7 days"
    else:
        change = "(no prior week to compare)"

    p("## This week")
    p("")
    p(f"- **{this_week:,}** hits in the last 7 days — {change}")
    p(f"- Previous three 7-day windows: {prev_week:,} → {last_week:,} → **{this_week:,}**")
    p(f"- Daily average (last 7d): **{this_week / 7:.0f}**/day")
    p("")

    # --- Daily trend, last 14 days ---
    p("## Daily hits (last 14 days)")
    p("")
    p("```")
    since = (now - timedelta(days=14)).strftime("%Y-%m-%d")
    daily = q(conn,
        "SELECT DATE(timestamp) d, COUNT(*) c FROM hits WHERE timestamp >= ? "
        "GROUP BY d ORDER BY d", (since,))
    dmax = max((c for _, c in daily), default=1)
    for d, c in daily:
        p(f"{d}  {c:>4}  {bar(c, dmax)}")
    p("```")
    p("")

    # --- Monthly trend ---
    p("## Monthly trend")
    p("")
    p("| Month | Hits | Hits/active day |")
    p("|---|---:|---:|")
    for month, hits, days in q(conn,
        "SELECT strftime('%Y-%m', timestamp) m, COUNT(*), "
        "COUNT(DISTINCT DATE(timestamp)) FROM hits GROUP BY m ORDER BY m"):
        p(f"| {month} | {hits:,} | {hits / days:.0f} |")
    p("")

    # --- Channel breakdown, last 30 days ---
    since30 = (now - timedelta(days=30)).strftime("%Y-%m-%d")
    rows = q(conn,
        "SELECT referrer FROM hits WHERE timestamp >= ?", (since30,))
    counts = {}
    for (ref,) in rows:
        counts[classify(ref)] = counts.get(classify(ref), 0) + 1
    tot30 = sum(counts.values()) or 1
    p("## Traffic channels (last 30 days)")
    p("")
    p("| Channel | Hits | Share |")
    p("|---|---:|---:|")
    for name, c in sorted(counts.items(), key=lambda x: -x[1]):
        p(f"| {name} | {c:,} | {100 * c / tot30:.0f}% |")
    p("")

    # --- Top non-search referrers, last 30 days ---
    refs = q(conn,
        "SELECT referrer, COUNT(*) c FROM hits WHERE timestamp >= ? "
        "AND referrer IS NOT NULL AND referrer != '' "
        "GROUP BY referrer ORDER BY c DESC LIMIT 40", (since30,))
    social = [(r, c) for r, c in refs
              if classify(r) in ("social", "ai assistant", "other referral", "code/docs")]
    if social:
        p("## Notable referrers — social / AI / other (last 30 days)")
        p("")
        p("| Referrer | Hits |")
        p("|---|---:|")
        for r, c in social[:15]:
            p(f"| {r} | {c} |")
        p("")

    # --- Top pages, last 30 days ---
    p("## Top pages (last 30 days)")
    p("")
    p("| Page | Hits |")
    p("|---|---:|")
    for path, c in q(conn,
        "SELECT path, COUNT(*) c FROM hits WHERE timestamp >= ? "
        "GROUP BY path ORDER BY c DESC LIMIT 15", (since30,)):
        p(f"| {path} | {c} |")
    p("")

    # --- Regions & devices, last 30 days ---
    p("## Edge regions & devices (last 30 days)")
    p("")
    regions = q(conn,
        "SELECT region, COUNT(*) c FROM hits WHERE timestamp >= ? "
        "GROUP BY region ORDER BY c DESC LIMIT 8", (since30,))
    p("Top Fly edge regions (proxy for geography): "
      + ", ".join(f"{r} ({c})" for r, c in regions))
    p("")
    devices = q(conn,
        "SELECT device, COUNT(*) c FROM hits WHERE timestamp >= ? "
        "GROUP BY device ORDER BY c DESC", (since30,))
    dtot = sum(c for _, c in devices) or 1
    p("Devices: " + ", ".join(f"{d} {100 * c / dtot:.0f}%" for d, c in devices))
    p("")

    conn.close()
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", help="path to a local analytics.db")
    ap.add_argument("--pull", action="store_true",
                    help="fetch the live DB off the Fly volume first")
    ap.add_argument("-o", "--output", help="write report to this file (else stdout)")
    args = ap.parse_args()

    if args.pull:
        db = args.db or os.path.join(tempfile.gettempdir(), "analytics_live.db")
        print(f"Pulling live DB -> {db}", file=sys.stderr)
        pull_live_db(db)
    elif args.db:
        db = args.db
    else:
        ap.error("give --db PATH or --pull")

    md = report(db)
    if args.output:
        with open(args.output, "w") as fh:
            fh.write(md + "\n")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(md)


if __name__ == "__main__":
    main()
