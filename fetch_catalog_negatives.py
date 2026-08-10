"""Hard negatives from external catalogues: known AGN and variable stars.

Why: our negatives are ~80% single-detection stream junk, which is easy. A
nuisance-only model (5 photometric numbers, no image) already reaches AUC 0.92
on this data, meaning the classes differ in brightness and environment as much
as in being supernovae. AGN and variable stars fix that -- they are point
sources, often bright, often sitting on galaxies, i.e. the cases genuinely
confusable with a supernova.

Crucially these labels come from spectroscopic/photometric catalogues
(Milliquas, AAVSO VSX), not from any alert classifier, so they stay
non-circular in exactly the way TNS labels do.

Two stages, because stage 2 competes with other fetches for the ALeRCE rate
limit and stage 1 does not touch ALeRCE at all:

    python fetch_catalog_negatives.py positions --n 8000
    python fetch_catalog_negatives.py fetch --n 6000 --workers 3 --rate 2
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA = Path("data")
POSITIONS = DATA / "catalog_positions.csv"

# VII/294 = Million Quasars (Flesch 2023); B/vsx = AAVSO Variable Star Index
CATALOGS = {
    "AGN": ("VII/294", ["RAJ2000", "DEJ2000"]),
    "VS": ("B/vsx", ["RAJ2000", "DEJ2000"]),
}
# X-Match needs the full table path, not the catalogue id that Vizier accepts
XMATCH_TABLES = {
    "AGN": "vizier:VII/294/catalog",   # Million Quasars
    "VS": "vizier:B/vsx/vsx",          # AAVSO Variable Star Index
}
# ZTF sees the northern sky; below about -30 deg there is nothing to match
DEC_MIN, DEC_MAX = -28.0, 85.0


def sample_positions(n_per_class: int, patches: int, radius_deg: float,
                     seed: int, only: list[str] | None = None) -> pd.DataFrame:
    """Cone-search several random sky patches and pool the results.

    Sampling patches rather than taking the catalogue's first N rows matters:
    these catalogues are ordered by position, so the head of the table is one
    small region of sky with its own extinction and crowding.
    """
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astroquery.vizier import Vizier
    from tqdm import tqdm

    rng = np.random.default_rng(seed)
    rows = []

    for label, (catalog, cols) in CATALOGS.items():
        if only and label not in only:
            continue
        v = Vizier(columns=cols, row_limit=max(2000, n_per_class // 2))
        got = []
        bar = tqdm(range(patches), desc=f"vizier[{label}]", unit="patch")
        for _ in bar:
            ra = rng.uniform(0, 360)
            # uniform on the sphere, not on dec, or patches cluster at the pole
            dec = np.degrees(np.arcsin(rng.uniform(
                np.sin(np.radians(DEC_MIN)), np.sin(np.radians(DEC_MAX)))))
            centre = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
            try:
                res = v.query_region(centre, radius=radius_deg * u.deg,
                                     catalog=catalog)
            except Exception:
                continue
            if not res:
                continue
            t = res[0].to_pandas()
            if len(t):
                got.append(t)
                bar.set_postfix(rows=sum(len(g) for g in got), refresh=False)
            if sum(len(g) for g in got) >= n_per_class:
                break
        bar.close()

        if not got:
            print(f"  {label}: nothing returned")
            continue
        df = pd.concat(got, ignore_index=True)
        df = df.rename(columns={"RAJ2000": "ra", "DEJ2000": "dec"})
        df = df[["ra", "dec"]].dropna()
        df = df[(df.dec > DEC_MIN) & (df.dec < DEC_MAX)]
        df = df.drop_duplicates().head(n_per_class)
        df["cat_class"] = label
        rows.append(df)
        print(f"  {label}: {len(df)} positions")

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def resolve_to_oids(positions: pd.DataFrame, radius_arcsec: float,
                    rate: float, workers: int) -> pd.DataFrame:
    """Cone-search ALeRCE at each catalogue position to find the ZTF object."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from alerce.core import Alerce
    from tqdm import tqdm
    import fetch_data

    fetch_data.LIMITER = fetch_data.RateLimiter(rate)

    def one(row):
        client = Alerce()
        try:
            df = fetch_data._retry(
                client.query_objects, survey="ztf", ra=float(row.ra),
                dec=float(row.dec), radius=radius_arcsec, page_size=1,
                format="pandas")
        except Exception:
            return None
        if df is None or len(df) == 0:
            return None
        return {"oid": df.oid.iloc[0], "cat_class": row.cat_class,
                "cat_ra": row.ra, "cat_dec": row.dec}

    out = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, r) for r in positions.itertuples(index=False)]
        bar = tqdm(as_completed(futures), total=len(futures),
                   desc="alerce cone-search", unit="pos")
        for f in bar:
            r = f.result()
            if r:
                out.append(r)
            bar.set_postfix(matched=len(out), refresh=False)
    return pd.DataFrame(out).drop_duplicates("oid")


def label_existing(manifest_path: Path, radius_arcsec: float) -> int:
    """Name the hard negatives we already have by crossmatching them outward.

    This is the efficient direction. Going catalogue -> ZTF wastes most of the
    catalogue: only 12% of Milliquas AGN and 32% of VSX variables ever vary
    enough to trigger a 5-sigma ZTF alert, so the yield is terrible. Going
    ZTF -> catalogue keeps every object we already paid to fetch, and anything
    without a match simply stays an unlabelled hard negative rather than being
    thrown away.

    Uses the CDS X-Match service, which does the whole table in one request
    instead of one query per object.
    """
    from astropy import units as u
    from astropy.table import Table
    from astroquery.xmatch import XMatch

    if not manifest_path.exists():
        sys.exit(f"{manifest_path} missing")
    d = pd.read_csv(manifest_path)
    if "ra" not in d.columns or "dec" not in d.columns:
        sys.exit(f"{manifest_path.name} has no ra/dec columns to crossmatch")

    d = d.dropna(subset=["ra", "dec"]).reset_index(drop=True)
    print(f"crossmatching {len(d)} objects at {radius_arcsec}\"")
    tab = Table.from_pandas(d[["oid", "ra", "dec"]])

    d["cat_class"] = "unknown"
    for label, table in XMATCH_TABLES.items():
        try:
            res = XMatch.query(cat1=tab, cat2=table,
                               max_distance=radius_arcsec * u.arcsec,
                               colRA1="ra", colDec1="dec")
        except Exception as exc:
            print(f"  {label}: X-Match failed ({type(exc).__name__}: "
                  f"{str(exc)[:90]}), skipping")
            continue
        matched = set(res["oid"].tolist()) if len(res) else set()
        # a variable star that is also a catalogued AGN is rare; first label wins
        newly = d.oid.isin(matched) & (d.cat_class == "unknown")
        d.loc[newly, "cat_class"] = label
        print(f"  {label}: {int(newly.sum())} objects matched")

    d.to_csv(manifest_path, index=False)
    print(f"\nupdated {manifest_path}")
    print(d.cat_class.value_counts().to_string())
    print("\n'unknown' rows are still perfectly good hard negatives -- they are")
    print("long-span objects, which physically cannot be supernovae.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["positions", "fetch", "label"])
    parser.add_argument("--n", type=int, default=6000)
    parser.add_argument("--patches", type=int, default=60)
    parser.add_argument("--radius-deg", type=float, default=0.4,
                        help="sky-patch radius for the Vizier sampling")
    parser.add_argument("--match-arcsec", type=float, default=1.5,
                        help="ALeRCE cone-search radius when resolving a "
                             "catalogue position to a ZTF object")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--rate", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--classes", nargs="+", choices=list(CATALOGS),
                        help="restrict the positions stage to these classes")
    parser.add_argument("--manifest", type=Path,
                        default=DATA / "manifest_hard_negatives.csv",
                        help="manifest to annotate in the 'label' stage")
    args = parser.parse_args()

    if args.stage == "positions":
        pos = sample_positions(args.n, args.patches, args.radius_deg, args.seed,
                               only=args.classes)
        if pos.empty:
            sys.exit("no catalogue positions retrieved")
        DATA.mkdir(exist_ok=True)
        if POSITIONS.exists():
            # accumulate across runs -- AGN are sparser than variable stars, so
            # getting a usable number of them takes several passes
            old = pd.read_csv(POSITIONS)
            pos = (pd.concat([old, pos], ignore_index=True)
                     .drop_duplicates(subset=["ra", "dec"]))
            print(f"merged with existing: {len(old)} -> {len(pos)} positions")
        pos.to_csv(POSITIONS, index=False)
        print(f"\nwrote {POSITIONS} ({len(pos)} positions)")
        print(pos.cat_class.value_counts().to_string())
        print("\nNext (run when no other ALeRCE fetch is active):")
        print("  python fetch_catalog_negatives.py fetch --n 6000")
        return 0

    if args.stage == "label":
        return label_existing(args.manifest, args.match_arcsec)

    if not POSITIONS.exists():
        sys.exit(f"{POSITIONS} missing -- run the 'positions' stage first")
    positions = pd.read_csv(POSITIONS).sample(frac=1, random_state=args.seed)
    positions = positions.head(args.n)

    matched = resolve_to_oids(positions, args.match_arcsec, args.rate, args.workers)
    if matched.empty:
        sys.exit("no catalogue position matched a ZTF object")
    print(f"\n{len(matched)} of {len(positions)} positions matched a ZTF object "
          f"({len(matched)/len(positions):.0%})")

    # never let a TNS-confirmed supernova in as a negative
    tns_path = DATA / "tns_ztf_all.csv"
    if tns_path.exists():
        tns = pd.read_csv(tns_path)
        confirmed = set(tns[tns.tns_label == 1].oid)
        before = len(matched)
        matched = matched[~matched.oid.isin(confirmed)]
        if before != len(matched):
            print(f"dropped {before - len(matched)} TNS-confirmed SNe")

    oid_file = DATA / "catalog_negative_oids.csv"
    matched.to_csv(oid_file, index=False)
    print(f"wrote {oid_file}")
    print(matched.cat_class.value_counts().to_string())

    import fetch_data
    fetch_data.LIMITER = fetch_data.RateLimiter(args.rate)
    manifest = fetch_data.build_from_oids(
        oid_file, len(matched), DATA, "catalog_negatives",
        workers=args.workers, first_detection=True)
    if manifest.empty:
        return 1

    # carry the catalogue class through so benchmark.py can break FP rate down
    manifest = manifest.merge(matched[["oid", "cat_class"]], on="oid", how="left")
    manifest.to_csv(DATA / "manifest_catalog_negatives.csv", index=False)
    print(f"\nwrote {DATA / 'manifest_catalog_negatives.csv'} ({len(manifest)} rows)")
    print(manifest.cat_class.value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
