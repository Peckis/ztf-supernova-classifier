"""Phase 2, step 2: fill in ground truth as TNS confirms it.

This is the delayed-label half of the continual-learning loop. ingest_night.py
scored these alerts when they arrived; over the following days and weeks TNS
publishes spectroscopic classifications for a few of them, and this job joins
those back onto the stored predictions.

    python label_from_tns.py --delta 20260814   # one day's changes (nightly)
    python label_from_tns.py --full             # whole catalogue (backfill)
    python label_from_tns.py --csv local.csv    # a file already on disk

TNS publishes daily delta files retained for two weeks, so the nightly path
stays small. Use --full to backfill or to recover after a gap longer than that.

Labels are applied per OBJECT, not per alert: one oid can have many alerts
across many nights, and a spectrum classifies the object, so every row for that
oid gets the same truth.

Types get revised -- an unclassified AT becomes "SN Ia" weeks later, and
occasionally a classification is corrected. Rows are therefore updated whenever
the type has changed, not only when they are currently NULL.

Requires the TNS credentials used by fetch_tns.py:
    TNS_BOT_ID, TNS_BOT_NAME, TNS_API_KEY
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fetch_tns import classify, download, read_tns_csv  # noqa: E402
from ingest_night import DB_PATH, connect  # noqa: E402

DATA = Path("data")


def tns_rows_for_known_oids(tns: pd.DataFrame, known: set[str]) -> pd.DataFrame:
    """Explode internal_names and keep only objects we have alerts for."""
    sub = tns[tns.internal_names.notna()][
        ["name", "name_prefix", "type", "internal_names"]].copy()
    sub["oid"] = sub.internal_names.str.split(",")
    sub = sub.explode("oid")
    sub["oid"] = sub.oid.str.strip()
    sub = sub[sub.oid.isin(known)]
    if sub.empty:
        return sub.assign(tns_label=None, tns_name=None)

    sub["tns_label"] = sub.type.map(classify)
    sub["tns_name"] = (sub.name_prefix.fillna("").astype(str)
                       + sub.name.astype(str))
    # a typed row beats an untyped one for the same object
    return (sub.sort_values("tns_label")
               .drop_duplicates("oid", keep="first")
               .rename(columns={"type": "tns_type"}))


def apply_labels(conn: sqlite3.Connection, rows: pd.DataFrame) -> dict:
    """Write labels onto every alert belonging to each labelled object."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    prev = pd.read_sql(
        "SELECT oid, tns_name, tns_type FROM alerts GROUP BY oid", conn
    ).set_index("oid")
    existing = prev.tns_type.to_dict()
    existing_name = prev.tns_name.to_dict()

    updates, new, revised = [], 0, 0
    for r in rows.itertuples():
        before = existing.get(r.oid)
        after = r.tns_type if isinstance(r.tns_type, str) else None
        # Compare the NAME too, not just the type. An object reported to TNS but
        # not yet classified has type NULL on both sides, so a type-only check
        # skips it -- and we would never record that it had been reported at
        # all, which is exactly the population we are waiting on spectra for.
        if before == after and existing_name.get(r.oid) == r.tns_name:
            continue
        if before is None and after is not None:
            new += 1
        elif before is not None:
            revised += 1
        label = None if pd.isna(r.tns_label) else int(r.tns_label)
        updates.append((r.tns_name, after, label, now, r.oid))

    if updates:
        conn.executemany(
            "UPDATE alerts SET tns_name=?, tns_type=?, tns_label=?, "
            "labelled_at=? WHERE oid=?", updates)
        conn.commit()
    return {"objects_updated": len(updates), "new": new, "revised": revised}


def fetch_missing_p_sn(conn: sqlite3.Connection, limit: int, rate: float,
                       workers: int) -> int:
    """Get ALeRCE's baseline score, but only for objects that now have truth.

    Deliberately not fetched at ingest time: 55,000 alerts a night would be
    hours of API calls. The accuracy plot only ever compares the two models on
    LABELLED objects, and TNS confirms a handful per night, so this stays cheap.
    """
    todo = pd.read_sql(
        "SELECT DISTINCT oid FROM alerts "
        "WHERE tns_label IS NOT NULL AND p_sn IS NULL LIMIT ?",
        conn, params=(limit,))
    if todo.empty:
        return 0

    import fetch_data
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from alerce.core import Alerce
    from tqdm import tqdm

    fetch_data.LIMITER = fetch_data.RateLimiter(rate)

    def one(oid):
        try:
            p, _ = fetch_data.fetch_p_sn(Alerce(), oid)
            return oid, p
        except Exception:
            return oid, None

    got = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, o) for o in todo.oid]
        for f in tqdm(as_completed(futures), total=len(futures),
                      desc="baseline p_sn", unit="obj"):
            oid, p = f.result()
            if p is not None:
                got.append((float(p), oid))

    if got:
        conn.executemany("UPDATE alerts SET p_sn=? WHERE oid=?", got)
        conn.commit()
    return len(got)


def summarise(conn: sqlite3.Connection) -> None:
    d = pd.read_sql(
        "SELECT tns_label, COUNT(DISTINCT oid) n FROM alerts "
        "WHERE tns_label IS NOT NULL GROUP BY tns_label", conn)
    total = pd.read_sql("SELECT COUNT(DISTINCT oid) n FROM alerts", conn).n[0]
    labelled = int(d.n.sum()) if not d.empty else 0
    print(f"\nobjects in db      : {total:,}")
    print(f"with a TNS label   : {labelled:,} ({labelled/max(total,1):.2%})")
    for r in d.itertuples():
        kind = {1: "confirmed SN", 0: "confirmed not-SN"}.get(r.tns_label, "?")
        print(f"  {kind:<18}: {r.n:,}")

    untyped = pd.read_sql(
        "SELECT COUNT(DISTINCT oid) n FROM alerts "
        "WHERE tns_name IS NOT NULL AND tns_label IS NULL", conn).n[0]
    print(f"  in TNS, untyped   : {untyped:,}  (reported, no spectrum yet)")

    # Only meaningful once the nightly delta job is running, where labelled_at
    # tracks when TNS published the classification. On a --full backfill it just
    # measures how long ago the alert was, so it is reported as such.
    lag = pd.read_sql(
        "SELECT AVG(julianday(labelled_at) - jd) d FROM alerts "
        "WHERE tns_label IS NOT NULL AND jd IS NOT NULL", conn).d[0]
    if lag is not None:
        print(f"  mean time alert->label recorded: {lag:.0f} days "
              f"(only a true label lag when run nightly, not on a backfill)")

    ready = pd.read_sql(
        "SELECT COUNT(DISTINCT oid) n FROM alerts WHERE tns_label IS NOT NULL "
        "AND stamp_path IS NOT NULL", conn).n[0]
    print(f"\nlabelled AND have a stored stamp: {ready:,}  <- retraining fuel")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--delta", metavar="YYYYMMDD",
                       help="one day's TNS changes (kept 2 weeks)")
    group.add_argument("--full", action="store_true",
                       help="whole catalogue; use to backfill or after a gap")
    group.add_argument("--csv", type=Path, help="a TNS csv already on disk")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--p-sn-limit", type=int, default=500,
                        help="max baseline lookups this run (0 to skip)")
    parser.add_argument("--rate", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(argv)

    if not args.db.exists():
        sys.exit(f"{args.db} does not exist -- run ingest_night.py first")
    conn = connect(args.db)

    known = set(pd.read_sql("SELECT DISTINCT oid FROM alerts", conn).oid)
    if not known:
        sys.exit("no alerts in the database yet")
    print(f"{len(known):,} distinct objects in the database")

    if args.csv:
        path = args.csv
    else:
        filename = ("tns_public_objects.csv.zip" if args.full
                    else f"tns_public_objects_{args.delta}.csv.zip")
        path = download(filename, DATA / "tns")

    tns = read_tns_csv(path)
    print(f"TNS rows in this file: {len(tns):,}")

    rows = tns_rows_for_known_oids(tns, known)
    print(f"matching our objects : {len(rows):,}")

    stats = apply_labels(conn, rows)
    print(f"updated {stats['objects_updated']:,} objects "
          f"({stats['new']:,} newly labelled, {stats['revised']:,} revised)")

    if args.p_sn_limit:
        n = fetch_missing_p_sn(conn, args.p_sn_limit, args.rate, args.workers)
        print(f"fetched baseline p_sn for {n:,} objects")

    summarise(conn)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
