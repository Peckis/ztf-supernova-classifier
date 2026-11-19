"""Upgrade presumed negatives to confirmed ones using external catalogues.

The dashboard's controls are unlabelled alerts called not-supernovae on a
base-rate argument, which is true in aggregate and unproven for any one object.
Crossmatching against SIMBAD, Milliquas and AAVSO VSX replaces that inference
with evidence where evidence exists.

    python confirm_negatives.py --limit 5000

The trap this is built around: a catalogue match is not automatically evidence
of absence. Supernovae explode inside galaxies, so a catalogued GALAXY match is
exactly what a real one looks like, and SIMBAD contains supernovae outright.
Hence an allowlist of types positively incompatible with being a supernova,
with everything else left presumed rather than guessed at.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from ingest_night import DB_PATH, connect  # noqa: E402

# in order of authority; first confident verdict wins
CATALOGUES = [
    ("SIMBAD", "vizier:simbad", "main_type"),
    ("Milliquas", "vizier:VII/294/catalog", None),   # every entry is an AGN/QSO
    ("VSX", "vizier:B/vsx/vsx", None),               # every entry is a variable
]

# types that positively cannot be a supernova: it is neither a star nor a
# persistent nucleus. Matched case-insensitively against main_type.
STELLAR = (
    "star", "*", "eclbin", "highpm", "sb*", "chempec", "horbranch", "rscvnv",
    "bydrav", "rgb", "**", "variable", "rrlyrae", "cepheid", "mira", "wolfrayet",
    "whitedwarf", "brightgiant", "supergiant", "ttauri", "yso", "flarestar",
    "pulsatingv", "rotatingv", "eruptivev", "cataclyv", "novacand", "bluestraggler",
    "carbonstar", "sstar", "post-agb", "planetarynebula",
)
AGN_LIKE = (
    "seyfert", "qso", "agn", "bllac", "blazar", "liner", "activegalaxy",
    "quasar", "radiogalaxy",
)
# ambiguous or outright a supernova: never a negative, whatever else matched
NEVER_NEGATIVE = ("supernova", "sn", "snr", "galaxy", "galgroup", "galcluster",
                  "interactinggal", "transient", "gravlens")


def classify_type(main_type) -> str | None:
    """SIMBAD main_type -> 'star' / 'agn' / None (not usable as a negative)."""
    if not isinstance(main_type, str) or not main_type.strip():
        return None
    t = main_type.strip().lower().replace(" ", "").replace("_", "")

    # rejections first: "SNRemnant" would otherwise slip through on another rule
    for bad in NEVER_NEGATIVE:
        if t == bad or t.startswith(bad):
            return None
    if any(a in t for a in AGN_LIKE):
        return "agn"
    if t.endswith("*") or any(s in t for s in STELLAR):
        return "star"
    return None


def ensure_columns(conn: sqlite3.Connection) -> None:
    have = {r[1] for r in conn.execute("PRAGMA table_info(alerts)")}
    for col in ("cat_class", "cat_source"):
        if col not in have:
            conn.execute(f"ALTER TABLE alerts ADD COLUMN {col} TEXT")
    conn.commit()


def require_astroquery():
    """Fail loudly if the crossmatch dependency is missing.

    Imported inside crossmatch(), the per-batch `except Exception` treated a
    missing package like a transient query error: every batch "failed", nothing
    was confirmed, and the script exited 0 nightly for days.
    """
    try:
        from astroquery.xmatch import XMatch  # noqa: F401
    except ImportError:
        sys.exit("astroquery is not installed -- the catalogue crossmatch "
                 "cannot run.\n  fix: uv pip install astroquery\n"
                 "  (setup_server.sh installs it for new deployments)")


def crossmatch(frame: pd.DataFrame, table: str, radius: float):
    from astropy import units as u
    from astropy.table import Table
    from astroquery.xmatch import XMatch
    res = XMatch.query(cat1=Table.from_pandas(frame[["oid", "ra", "dec"]]),
                       cat2=table, max_distance=radius * u.arcsec,
                       colRA1="ra", colDec1="dec")
    return res.to_pandas() if len(res) else pd.DataFrame()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--limit", type=int, default=20000,
                        help="objects to process this run")
    parser.add_argument("--batch", type=int, default=2000,
                        help="rows per X-Match upload")
    parser.add_argument("--radius", type=float, default=1.5,
                        help="match radius in arcsec")
    parser.add_argument("--controls-per-night", type=int, default=300,
                        help="match the dashboard's control sample size")
    parser.add_argument("--all", action="store_true",
                        help="crossmatch every unlabelled object, not just the "
                             "controls the dashboard evaluates")
    args = parser.parse_args(argv)

    require_astroquery()
    if not args.db.exists():
        sys.exit(f"{args.db} does not exist -- run ingest_night.py first")
    conn = connect(args.db)
    ensure_columns(conn)

    # only the per-night control sample is ever measured, so crossmatching all
    # several hundred thousand alerts would load a free CDS service for nothing
    if args.all:
        todo = pd.read_sql("""
            SELECT oid, MAX(ra) ra, MAX(dec) dec FROM alerts
            WHERE cat_class IS NULL AND ra IS NOT NULL AND tns_label IS NULL
            GROUP BY oid LIMIT ?
        """, conn, params=(args.limit,))
    else:
        todo = pd.read_sql("""
            SELECT oid, ra, dec FROM (
                SELECT oid, ra, dec,
                       ROW_NUMBER() OVER (PARTITION BY night ORDER BY candid) rn
                FROM alerts
                WHERE cat_class IS NULL AND ra IS NOT NULL
                  AND tns_label IS NULL AND tns_name IS NULL
            ) WHERE rn <= ? LIMIT ?
        """, conn, params=(args.controls_per_night, args.limit))
    if todo.empty:
        print("nothing to do -- every unlabelled object already has a verdict")
        return 0
    print(f"{len(todo):,} objects to check against {len(CATALOGUES)} catalogues")

    verdicts: dict[str, tuple[str, str]] = {}
    for name, table, type_col in CATALOGUES:
        pending = todo[~todo.oid.isin(verdicts)]
        if pending.empty:
            break
        found = 0
        for start in range(0, len(pending), args.batch):
            chunk = pending.iloc[start:start + args.batch]
            try:
                res = crossmatch(chunk, table, args.radius)
            except Exception as exc:
                print(f"  {name}: batch failed ({type(exc).__name__}), skipping")
                continue
            if res.empty:
                continue
            # keep the closest match per object
            res = res.sort_values("angDist").drop_duplicates("oid", keep="first")
            for row in res.itertuples():
                if row.oid in verdicts:
                    continue
                if type_col:
                    kind = classify_type(getattr(row, type_col, None))
                else:
                    # a Milliquas or VSX entry IS the classification
                    kind = "agn" if name == "Milliquas" else "star"
                if kind:
                    verdicts[row.oid] = (kind, name)
                    found += 1
        print(f"  {name:<10} confirmed {found:,} objects as not-supernovae")

    if not verdicts:
        print("\nno confirmations this run")
        return 0

    conn.executemany("UPDATE alerts SET cat_class=?, cat_source=? WHERE oid=?",
                     [(k, s, o) for o, (k, s) in verdicts.items()])
    conn.commit()

    total, confirmed = conn.execute(
        "SELECT COUNT(DISTINCT oid), "
        "COUNT(DISTINCT CASE WHEN cat_class IS NOT NULL THEN oid END) "
        "FROM alerts").fetchone()
    by_kind = pd.read_sql(
        "SELECT cat_class, cat_source, COUNT(DISTINCT oid) n FROM alerts "
        "WHERE cat_class IS NOT NULL GROUP BY cat_class, cat_source", conn)
    print(f"\nconfirmed this run : {len(verdicts):,}")
    print(f"confirmed in total : {confirmed:,} of {total:,} objects "
          f"({confirmed/max(total,1):.1%})")
    print(by_kind.to_string(index=False))
    print("\nThese are now evidence-backed negatives rather than base-rate "
          "assumptions.\nObjects matching a galaxy, a transient, or a SIMBAD "
          "supernova are deliberately\nleft unconfirmed -- a supernova sits in "
          "a galaxy, so such a match proves nothing.")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
