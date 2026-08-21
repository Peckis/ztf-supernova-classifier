"""Is the pipeline healthy? One command, one screen, one verdict.

    python status.py            # human summary
    python status.py --events   # plus the recent event log
    python status.py --quiet    # print nothing, just set the exit code

Exit codes, so this works from a monitor or a shell prompt too:
    0  OK or DEGRADED   nothing needs you today
    1  FAILING          a stage is erroring; the loop is still turning
    2  STALLED          the loop is not running at all, or cannot continue

The important half of this is what a log file cannot tell you. Logs only record
runs that HAPPENED. The likeliest real failure -- cron removed, machine
rebooted, disk full, lock left stale -- produces no log line at all, just
silence that looks exactly like a quiet night. So the checks below compare what
should have happened against what did, rather than only reading back what was
written.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).parent
DB = REPO / "data" / "pipeline.db"
EVENTS = REPO / "logs" / "events.tsv"
DASHBOARD = REPO / "dashboard" / "index.html"
LOCK = REPO / ".nightly.lock"

# Severity ladder. A check reports the worst thing it finds; the overall verdict
# is the worst across all checks.
OK, DEGRADED, FAILING, STALLED = "OK", "DEGRADED", "FAILING", "STALLED"
RANK = {OK: 0, DEGRADED: 1, FAILING: 2, STALLED: 3}
EXIT = {OK: 0, DEGRADED: 0, FAILING: 1, STALLED: 2}

# A night of downloads plus stamps needs headroom; below this the next run
# cannot complete, which is a stop-the-world problem rather than a warning.
DISK_CRITICAL_GB = 20
DISK_WARN_GB = 60
# Missing one night is a weather night. Missing two days of RUNS means the
# scheduler itself is gone.
RUN_STALE_HOURS = 36
RUN_CRITICAL_HOURS = 72


def now() -> datetime:
    return datetime.now(timezone.utc)


def parse_events(limit_days: int = 14) -> list[tuple]:
    if not EVENTS.exists():
        return []
    cutoff = now() - timedelta(days=limit_days)
    out = []
    for line in EVENTS.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split("\t")
        if len(parts) != 4:
            continue
        try:
            ts = datetime.strptime(parts[0], "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc)
        except ValueError:
            continue
        if ts >= cutoff:
            out.append((ts, parts[1], parts[2], parts[3]))
    return out


def humanise(delta: timedelta) -> str:
    h = delta.total_seconds() / 3600
    if h < 1:
        return f"{delta.total_seconds()/60:.0f} min ago"
    if h < 48:
        return f"{h:.0f}h ago"
    return f"{h/24:.1f} days ago"


def check_scheduler(events) -> tuple[str, list[str]]:
    """Did the loop actually run recently? This is the check that matters."""
    runs = [e for e in events if e[2] == "run" and e[3].startswith(("started",))]
    if not runs:
        return STALLED, ["no run recorded in the last 14 days -- "
                         "is cron installed? (bash install_cron.sh)"]
    last = max(r[0] for r in runs)
    age = now() - last
    line = f"last run        : {last:%Y-%m-%d %H:%M} UTC ({humanise(age)})"
    if age > timedelta(hours=RUN_CRITICAL_HOURS):
        return STALLED, [line + "  <- NOT RUNNING",
                         "  check: crontab -l | grep nightly ; systemctl status cron"]
    if age > timedelta(hours=RUN_STALE_HOURS):
        return FAILING, [line + "  <- overdue, expected daily"]
    return OK, [line]


def check_last_outcome(events) -> tuple[str, list[str]]:
    # "skipped" is a terminal outcome, not a death. nightly.sh exits early and
    # cleanly when it finds the lock held -- which is what should happen while a
    # backfill is running -- and that path returns before the "finished" event
    # at the end of the script. Counting only "finished" made a correct skip
    # look like a crashed run, i.e. a monitor crying wolf at its own design.
    finished = [e for e in events if e[2] == "run"
                and ("finished" in e[3] or e[3].startswith("skipped"))]
    started = [e for e in events if e[2] == "run" and e[3] == "started"]
    if not finished:
        if started:
            # started but never finished: either still going, or it died
            last = max(s[0] for s in started)
            if now() - last < timedelta(hours=6):
                return OK, ["last outcome    : still running"]
            return FAILING, ["last outcome    : started but never finished "
                             "-- previous run died mid-way"]
        return DEGRADED, ["last outcome    : unknown"]
    last = max(finished, key=lambda e: e[0])
    if last[1] == "ERROR":
        return FAILING, [f"last outcome    : {last[3]}"]
    if last[3].startswith("skipped"):
        return OK, ["last outcome    : skipped (another run held the lock)"]
    return OK, ["last outcome    : finished cleanly"]


def check_errors(events) -> tuple[str, list[str]]:
    errs = [e for e in events if e[1] == "ERROR"]
    warns = [e for e in events if e[1] == "WARN"]
    lines = [f"last 14 days    : {len(errs)} errors, {len(warns)} warnings"]
    if errs:
        return FAILING, lines
    if len(warns) > 20:
        return DEGRADED, lines + ["  (a lot of warnings -- run --events to see)"]
    return OK, lines


def check_disk() -> tuple[str, list[str]]:
    target = DB.parent if DB.parent.exists() else REPO
    free = shutil.disk_usage(target).free / 1e9
    line = f"free disk       : {free:.0f} GB"
    if free < DISK_CRITICAL_GB:
        return STALLED, [line + "  <- too low to download another night"]
    if free < DISK_WARN_GB:
        return DEGRADED, [line + "  (getting tight)"]
    return OK, [line]


def check_data() -> tuple[str, list[str]]:
    """Is data actually flowing, or is the loop spinning on nothing?"""
    if not DB.exists():
        return STALLED, ["database        : missing -- nothing has ingested yet"]
    try:
        c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        alerts, nights = c.execute(
            "SELECT COUNT(*), COUNT(DISTINCT night) FROM alerts").fetchone()
        last_night = c.execute("SELECT MAX(night) FROM runs").fetchone()[0]
        labelled = c.execute("SELECT COUNT(DISTINCT oid) FROM alerts "
                             "WHERE tns_label IS NOT NULL").fetchone()[0]
        ready = c.execute("SELECT COUNT(DISTINCT oid) FROM alerts WHERE "
                          "tns_label IS NOT NULL AND stamp_path IS NOT NULL"
                          ).fetchone()[0]
        c.close()
    except sqlite3.Error as exc:
        return STALLED, [f"database        : unreadable ({exc})"]

    lines = [f"nights ingested : {nights}  (latest {last_night})",
             f"alerts scored   : {alerts:,}",
             f"TNS-confirmed   : {labelled:,}  ({ready:,} with stamps kept)"]
    if alerts == 0:
        return DEGRADED, lines + ["  nothing scored yet -- run a backfill"]
    return OK, lines


def check_dashboard() -> tuple[str, list[str]]:
    if not DASHBOARD.exists():
        return DEGRADED, ["dashboard       : never built"]
    age = now() - datetime.fromtimestamp(DASHBOARD.stat().st_mtime, timezone.utc)
    line = f"dashboard       : rebuilt {humanise(age)}"
    if age > timedelta(hours=RUN_CRITICAL_HOURS):
        return FAILING, [line + "  <- stale"]
    return OK, [line]


def check_stale_lock() -> tuple[str, list[str]]:
    """A lock left behind by a killed run blocks every future run, silently."""
    if not LOCK.exists():
        return OK, []
    age = now() - datetime.fromtimestamp(LOCK.stat().st_mtime, timezone.utc)
    if age > timedelta(hours=24):
        return DEGRADED, [
            f"lock file       : {humanise(age)} old",
            "  if runs are being skipped: rm .nightly.lock"]
    return OK, []


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", action="store_true",
                        help="also print recent warnings and errors")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--quiet", action="store_true",
                        help="no output, exit code only")
    args = parser.parse_args(argv)

    events = parse_events(args.days)
    results = [
        check_scheduler(events), check_last_outcome(events),
        check_errors(events), check_data(), check_dashboard(),
        check_disk(), check_stale_lock(),
    ]
    verdict = max((r[0] for r in results), key=lambda v: RANK[v])

    if not args.quiet:
        banner = {OK: "OK", DEGRADED: "DEGRADED  (running, worth a look)",
                  FAILING: "FAILING   (a stage is erroring)",
                  STALLED: "STALLED   (not running / cannot continue)"}[verdict]
        print(f"pipeline status : {banner}")
        print(f"checked         : {now():%Y-%m-%d %H:%M} UTC")
        print("-" * 60)
        for _, lines in results:
            for line in lines:
                print(line)

        if args.events:
            notable = [e for e in events if e[1] in ("WARN", "ERROR")]
            print("-" * 60)
            if not notable:
                print(f"no warnings or errors in {args.days} days")
            else:
                print(f"warnings and errors, last {args.days} days "
                      f"(newest first):")
                for ts, level, stage, msg in sorted(notable, reverse=True)[:25]:
                    print(f"  {ts:%m-%d %H:%M}  {level:<5} {stage:<9} {msg}")
        elif verdict != OK:
            print("-" * 60)
            print("run 'python status.py --events' for the detail")

    return EXIT[verdict]


if __name__ == "__main__":
    sys.exit(main())
