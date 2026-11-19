"""Join TNS ground-truth classifications onto our ALeRCE candidates.

Writes data/tns_labels.csv separately rather than into manifest.csv, which
fetch_data.py rewrites wholesale and would silently wipe.

Needs TNS_BOT_ID, TNS_BOT_NAME, TNS_API_KEY in the environment.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

BASE = "https://www.wis-tns.org/system/files/tns_public_objects"
DATA = Path("data")

SN_PREFIXES = ("SN ", "SLSN")

# a type in neither group (including a blank one, i.e. reported but not yet
# classified) stays unlabelled rather than being guessed at
NON_SN_TYPES = {
    "AGN", "Blazar", "CV", "Galaxy", "Impostor-SN", "LBV", "Light-Echo",
    "M dwarf", "Nova", "QSO", "Star", "TDE", "Varstar", "WR", "XRB",
}


def _load_dotenv() -> None:
    """Read .env if the variables are not already exported.

    nightly.sh sources .env, so cron had credentials and a hand-run command did
    not. Values already in the environment win, so an export still overrides.
    """
    path = Path(__file__).parent / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def credentials() -> tuple[str, str, str]:
    _load_dotenv()
    bot_id = os.environ.get("TNS_BOT_ID", "")
    bot_name = os.environ.get("TNS_BOT_NAME", "")
    api_key = os.environ.get("TNS_API_KEY", "")
    missing = [n for n, v in [("TNS_BOT_ID", bot_id), ("TNS_BOT_NAME", bot_name),
                              ("TNS_API_KEY", api_key)] if not v]
    if missing:
        sys.exit(f"missing environment variable(s): {', '.join(missing)}\n"
                 "Create a bot at https://www.wis-tns.org (profile -> Bots) and set them.")
    return bot_id, bot_name, api_key


def download(filename: str, dest: Path) -> Path:
    """Fetch one TNS csv.zip and unpack the csv inside it."""
    bot_id, bot_name, api_key = credentials()
    # TNS rejects scripted requests without this exact User-Agent shape
    marker = json.dumps({"tns_id": int(bot_id), "type": "bot", "name": bot_name})
    headers = {"user-agent": f"tns_marker{marker}"}

    url = f"{BASE}/{filename}"
    print(f"downloading {url}")
    response = requests.post(url, headers=headers, data={"api_key": api_key}, timeout=300)
    if response.status_code != 200:
        sys.exit(f"TNS returned {response.status_code}: {response.text[:300]}")

    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        name = archive.namelist()[0]
        archive.extract(name, dest)
    path = dest / name
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def read_tns_csv(path: Path) -> pd.DataFrame:
    """Read the TNS csv, tolerating the leading metadata line it sometimes has."""
    df = pd.read_csv(path, low_memory=False)
    if "internal_names" not in df.columns:
        # first row was a version/date banner, not the header
        df = pd.read_csv(path, low_memory=False, skiprows=1)
    if "internal_names" not in df.columns:
        sys.exit(f"unexpected TNS csv columns: {list(df.columns)[:10]}")
    return df


def classify(tns_type) -> float:
    """Map a TNS type string to 1 (SN), 0 (confirmed not-SN), or NaN (unknown)."""
    if not isinstance(tns_type, str) or not tns_type.strip():
        return float("nan")
    t = tns_type.strip()
    if t == "SN" or t.startswith(SN_PREFIXES):
        return 1.0
    if t in NON_SN_TYPES:
        return 0.0
    return float("nan")


def build_labels(tns: pd.DataFrame, oids: set[str] | None) -> pd.DataFrame:
    """Explode internal_names and keep ZTF rows.

    oids=None keeps every ZTF-named object in TNS, not just our candidates --
    that is the pool to build a training set from.
    """
    subset = tns[["name", "name_prefix", "type", "internal_names"]].copy()
    subset = subset[subset.internal_names.notna()]

    # "ZTF18abc, ATLAS18xyz" -> one row per internal name
    subset["oid"] = subset.internal_names.str.split(",")
    subset = subset.explode("oid")
    subset["oid"] = subset.oid.str.strip()
    if oids is None:
        subset = subset[subset.oid.str.startswith("ZTF", na=False)]
    else:
        subset = subset[subset.oid.isin(oids)]

    subset["tns_label"] = subset.type.map(classify)
    subset["tns_name"] = (subset.name_prefix.fillna("").astype(str)
                          + subset.name.astype(str))
    out = subset[["oid", "tns_name", "type", "tns_label"]].rename(
        columns={"type": "tns_type"})
    # one object can appear under several internal names; keep the labeled row
    return out.sort_values("tns_label").drop_duplicates("oid", keep="first")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path,
                        help="use a local TNS csv instead of downloading")
    parser.add_argument("--delta", metavar="YYYYMMDD",
                        help="fetch one day's delta file instead of the full catalogue")
    parser.add_argument("--out", type=Path, default=DATA / "tns_labels.csv")
    parser.add_argument("--all-ztf", action="store_true",
                        help="export every ZTF-named TNS object, not just our "
                             "current candidates -- the pool to build a real "
                             "training set from")
    args = parser.parse_args(argv)

    if args.all_ztf:
        oids = None
        if args.out == DATA / "tns_labels.csv":
            args.out = DATA / "tns_ztf_all.csv"
    else:
        oids = set(pd.read_csv(DATA / "manifest.csv").oid)

    if args.csv:
        path = args.csv
    else:
        filename = (f"tns_public_objects_{args.delta}.csv.zip" if args.delta
                    else "tns_public_objects.csv.zip")
        path = download(filename, DATA / "tns")

    tns = read_tns_csv(path)
    print(f"TNS rows: {len(tns):,}")

    labels = build_labels(tns, oids)
    labels.to_csv(args.out, index=False)

    matched = len(labels)
    positives = int((labels.tns_label == 1).sum())
    negatives = int((labels.tns_label == 0).sum())
    unlabeled = int(labels.tns_label.isna().sum())

    print(f"\nwrote {args.out}")
    if oids is None:
        print(f"ZTF-named TNS objects : {matched}")
        print(f"  confirmed SN        : {positives}   <- available positives")
        print(f"  confirmed not-SN    : {negatives}")
        print(f"  reported, untyped   : {unlabeled}")
        return 0

    print(f"candidates            : {len(oids)}")
    print(f"found in TNS          : {matched} ({matched/len(oids):.1%})")
    print(f"  confirmed SN        : {positives}")
    print(f"  confirmed not-SN    : {negatives}")
    print(f"  reported, untyped   : {unlabeled}")
    print(f"not in TNS at all     : {len(oids) - matched}")
    print("\nNote: 'not in TNS' is NOT a negative label -- most ZTF alerts are")
    print("never reported. Only the confirmed rows above are ground truth.")

    if matched:
        print("\ntop TNS types among our candidates:")
        for name, count in labels.tns_type.value_counts().head(10).items():
            print(f"  {name:>14}: {count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
