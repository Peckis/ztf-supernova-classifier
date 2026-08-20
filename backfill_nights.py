"""Ingest a run of past nights, so the dashboard has real numbers today.

The live loop would take months to accumulate labels. The archive holds 2,198
nights going back to 2018, and TNS has already followed those up -- so ingesting
older nights gives the accuracy plot months of hindsight immediately.

    python backfill_nights.py --nights 8 --start 20250101 --end 20260315
    python backfill_nights.py --nights 12 --strategy smallest --dry-run

Downloads one night at a time and deletes it before fetching the next, so peak
disk is one tarball plus the kept stamps. Nights already in the runs table are
skipped, so an interrupted run resumes for free.

On picking nights: --strategy spread is the default and takes them evenly across
the window. --strategy smallest downloads far less, but small nights are small
because of weather, so it quietly selects for poor observing conditions -- fine
for a plumbing test, misleading for a benchmark.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from ingest_night import DB_PATH, connect  # noqa: E402

INDEX_URL = "https://ztf.uw.edu/alerts/public/"
DATA = Path("data")
SIZE_UNITS = {"": 1e-9, "K": 1e-6, "M": 1e-3, "G": 1.0, "T": 1000.0}
ROW = re.compile(
    r'ztf_public_(\d{8})\.tar\.gz</a></td>'
    r'<td class="indexcollastmod">[^<]*</td>'
    r'<td class="indexcolsize">\s*([\d.]+)\s*([KMGT]?)\s*</td>')


def available_nights(min_gb: float = 0.5) -> list[tuple[str, float]]:
    """Every night in the archive index with its size in GB.

    Nights with no observations are published as 74-byte placeholders, so
    anything tiny is filtered out rather than downloaded and found empty.
    """
    html = requests.get(INDEX_URL, timeout=120).text
    out = []
    for date, value, unit in ROW.findall(html):
        gb = float(value) * SIZE_UNITS[unit]
        if gb >= min_gb:
            out.append((date, gb))
    return sorted(out)


def choose(nights, start, end, count, strategy, max_gb):
    pool = [(d, g) for d, g in nights if start <= d <= end and g <= max_gb]
    if not pool:
        return []
    if strategy == "smallest":
        return sorted(sorted(pool, key=lambda x: x[1])[:count])
    # spread: evenly spaced across the window so weather and season average out
    if count >= len(pool):
        return pool
    step = len(pool) / count
    return [pool[int(i * step)] for i in range(count)]


def already_done(db: Path) -> set[str]:
    if not db.exists():
        return set()
    conn = connect(db)
    done = {r[0] for r in conn.execute("SELECT night FROM runs")}
    conn.close()
    return done


def free_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1e9


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nights", type=int, default=8)
    parser.add_argument("--start", default="20250101")
    parser.add_argument("--end", default="20260315",
                        help="leave >=4 months for TNS to follow up")
    parser.add_argument("--strategy", choices=["spread", "smallest"],
                        default="spread")
    parser.add_argument("--max-gb", type=float, default=12.0,
                        help="skip nights larger than this")
    parser.add_argument("--keep-stamps", type=int, default=1000)
    parser.add_argument("--max-ndethist", type=int, default=2)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--dry-run", action="store_true",
                        help="list the nights and totals, download nothing")
    args = parser.parse_args(argv)

    print("reading the archive index ...")
    nights = available_nights()
    print(f"{len(nights)} nights with data, {nights[0][0]} .. {nights[-1][0]}")

    picked = choose(nights, args.start, args.end, args.nights,
                    args.strategy, args.max_gb)
    if not picked:
        sys.exit("no nights matched -- widen --start/--end or --max-gb")

    done = already_done(args.db)
    todo = [(d, g) for d, g in picked if d not in done]
    total = sum(g for _, g in todo)

    print(f"\nselected {len(picked)} nights ({args.strategy}), "
          f"{len(picked) - len(todo)} already ingested")
    for d, g in todo:
        print(f"  {d}  {g:5.1f} GB")
    print(f"  total download: {total:.0f} GB")

    have = free_gb(DATA)
    print(f"free disk: {have:.0f} GB "
          f"(needs ~{max((g for _, g in todo), default=0):.0f} GB at a time, "
          f"tarballs are deleted after each night)")
    if todo and have < max(g for _, g in todo) * 1.5:
        sys.exit("not enough free disk for the largest night")

    if args.dry_run:
        print("\n(dry run -- nothing downloaded)")
        return 0
    if not todo:
        print("\nnothing to do")
        return 0

    started = time.perf_counter()
    ok = failed = 0
    for i, (night, gb) in enumerate(todo, start=1):
        print(f"\n{'='*70}\n[{i}/{len(todo)}] night {night} ({gb:.1f} GB)\n{'='*70}")
        cmd = [sys.executable, "-u", "ingest_night.py", "--date", night,
               "--keep-stamps", str(args.keep_stamps),
               "--max-ndethist", str(args.max_ndethist)]
        result = subprocess.run(cmd)
        if result.returncode == 0:
            ok += 1
        else:
            failed += 1
            print(f"  night {night} failed (exit {result.returncode}), continuing")
        # a stale tarball would otherwise accumulate if ingest died mid-run
        leftover = DATA / "archive" / f"ztf_public_{night}.tar.gz"
        if leftover.exists():
            leftover.unlink()

    mins = (time.perf_counter() - started) / 60
    print(f"\n{'='*70}")
    print(f"backfill done: {ok} nights ingested, {failed} failed, "
          f"{mins:.0f} min")

    conn = sqlite3.connect(args.db)
    n, nights_done = conn.execute(
        "SELECT COUNT(*), COUNT(DISTINCT night) FROM alerts").fetchone()
    conn.close()
    print(f"database now holds {n:,} scored alerts over {nights_done} nights")
    print("\nnext:")
    print("  python label_from_tns.py --full     # attach TNS ground truth")
    print("  python export_dashboard.py          # build the plot and gallery")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
