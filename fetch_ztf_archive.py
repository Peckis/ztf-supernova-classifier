"""Build a negative set straight from ZTF's public alert archive.

One nightly tarball carries every alert with its cutouts inside the packet, so
a sequential download replaces tens of thousands of API requests, and nothing
has been through a broker's selection first.

    python fetch_ztf_archive.py --date 20260801 --n 20000

Tarballs are 5-15 GB (occasionally 33 GB) and reach back only to early 2025;
historical TNS supernovae still come from the ALeRCE API by object id.
"""

from __future__ import annotations

import argparse
import gzip
import io
import os
import random
import sys
import tarfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from astropy.io import fits
from tqdm import tqdm

warnings.filterwarnings("ignore", module="astropy.io.fits")

ARCHIVE = "https://ztf.uw.edu/alerts/public"
STAMP_SIZE = 63
SENTINEL_THRESHOLD = -1e30
# packet field -> our channel order, matching fetch_data.py exactly so that
# train.py cannot tell the two sources apart
CUTOUTS = ("cutoutScience", "cutoutTemplate", "cutoutDifference")


def download(date: str, dest: Path) -> Path:
    """Stream the nightly tarball to disk, atomically.

    A truncated tarball is still a readable gzip stream, so iter_packets would
    parse part of the night, stop cleanly, and ingest_night would record the
    night as done -- silent data loss. Hence .part until the byte count
    matches Content-Length.
    """
    url = f"{ARCHIVE}/ztf_public_{date}.tar.gz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    if dest.exists():
        # trust a cached file only at the right size, for the reason above
        have = dest.stat().st_size
        try:
            expect = int(requests.head(url, timeout=60).headers
                         .get("content-length", 0))
        except Exception:
            expect = 0
        if expect and have != expect:
            print(f"cached {dest.name} is {have:,} bytes but the archive says "
                  f"{expect:,} -- discarding and re-downloading")
            dest.unlink()
        else:
            print(f"{dest} already present ({have / 1e9:.1f} GB), reusing")
            return dest
    if part.exists():
        # no range requests on the archive, so there is nothing to resume from
        print(f"discarding incomplete {part.name} from an interrupted run")
        part.unlink()

    print(f"downloading {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        if r.status_code != 200:
            sys.exit(f"archive returned {r.status_code} for {url}\n"
                     "Check the date exists at https://ztf.uw.edu/alerts/public/")
        total = int(r.headers.get("content-length", 0))
        bar = tqdm(total=total, unit="B", unit_scale=True, desc="tarball", disable=None)
        written = 0
        with open(part, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                written += len(chunk)
                bar.update(len(chunk))
            f.flush()
            os.fsync(f.fileno())   # survive a power cut, not just a crash
        bar.close()

    if total and written != total:
        part.unlink(missing_ok=True)
        sys.exit(f"download incomplete: got {written:,} of {total:,} bytes. "
                 f"Nothing was ingested; the next run will retry this night.")

    part.replace(dest)             # atomic on the same filesystem
    print(f"wrote {dest} ({dest.stat().st_size / 1e9:.1f} GB)")
    return dest


def cutout_to_array(cutout) -> np.ndarray:
    """One packet cutout -> a 63x63 float32 plane (gzipped FITS in stampData)."""
    raw = gzip.decompress(cutout["stampData"])
    with fits.open(io.BytesIO(raw), memmap=False) as hdul:
        data = hdul[0].data
    plane = np.asarray(data, dtype=np.float32)
    # ZTF marks gaps as NaN in some planes and -FLT_MAX padding in others, and
    # a finite -3.4e38 would sail into the network and blow up the loss
    plane[~np.isfinite(plane) | (plane < SENTINEL_THRESHOLD)] = np.nan
    return plane


def packet_to_stamp(packet) -> np.ndarray:
    """Stack the three cutouts into [3, 63, 63], or raise if unusable."""
    planes = []
    for name in CUTOUTS:
        cutout = packet.get(name)
        if cutout is None or not cutout.get("stampData"):
            raise ValueError(f"missing {name}")
        plane = cutout_to_array(cutout)
        if plane.shape != (STAMP_SIZE, STAMP_SIZE):
            raise ValueError(f"{name} is {plane.shape}")
        planes.append(plane)
    return np.stack(planes, axis=0)


def iter_packets(tarball: Path):
    """Stream Avro packets out of the tarball without extracting it.

    Streaming mode ('r|gz'), so a 15 GB archive is never unpacked or held.
    """
    from fastavro import reader as avro_reader

    with tarfile.open(tarball, "r|gz") as tar:
        # a truncated archive (partial or interrupted download) raises when the
        # stream runs out mid-member; stop cleanly and keep what we already read
        try:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".avro"):
                    continue
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                try:
                    for packet in avro_reader(io.BytesIO(handle.read())):
                        yield packet
                except Exception:
                    continue
        except (tarfile.ReadError, EOFError, OSError) as exc:
            print(f"  archive ended early ({type(exc).__name__}) -- "
                  f"keeping what was read")


def build(tarball: Path, n: int, out_dir: Path, name: str,
          sample_rate: float, seed: int = 0) -> pd.DataFrame:
    stamp_dir = out_dir / "stamps"
    stamp_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    rows, failures, seen_oids = [], 0, set()
    bar = tqdm(total=n, desc=f"alerts[{name}]", unit="obj", disable=None)

    for packet in iter_packets(tarball):
        if len(rows) >= n:
            break
        # thin the stream: the first N alerts would all be one patch of sky
        if sample_rate < 1.0 and rng.random() > sample_rate:
            continue

        oid = packet.get("objectId")
        if not oid or oid in seen_oids:
            continue
        seen_oids.add(oid)

        try:
            arr = packet_to_stamp(packet)
        except Exception:
            failures += 1
            continue

        path = stamp_dir / f"{oid}.npy"
        np.save(path, arr)

        c = packet.get("candidate", {})
        rows.append({
            "oid": oid,
            "candid": packet.get("candid"),
            # no p_sn: raw stream, no broker has scored it
            "p_sn": float("nan"),
            "ra": c.get("ra"),
            "dec": c.get("dec"),
            "jd": c.get("jd"),
            "magpsf": c.get("magpsf"),
            "fid": c.get("fid"),
            # ZTF's own real/bogus score -- useful context, NOT a label
            "rb": c.get("rb"),
            "drb": c.get("drb"),
            "ndethist": c.get("ndethist"),
            "n_missing": int(np.isnan(arr).sum()),
            "missing_frac": round(float(np.isnan(arr).mean()), 4),
            "stamp_path": str(path.relative_to(out_dir)).replace("\\", "/"),
        })
        bar.update(1)
        bar.set_postfix(failed=failures, refresh=False)

    bar.close()

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        print("no usable alerts found")
        return manifest

    path = out_dir / f"manifest_{name}.csv"
    if path.exists():
        old = pd.read_csv(path)
        manifest = (pd.concat([old, manifest], ignore_index=True)
                      .drop_duplicates("oid", keep="first").reset_index(drop=True))
        print(f"merged with existing {path.name}: {len(old)} -> {len(manifest)} rows")
    manifest.to_csv(path, index=False)

    print(f"\nwrote {path} ({len(manifest)} rows, {failures} unusable)")
    print(f"unique objects: {manifest.oid.nunique()}")
    if "ndethist" in manifest:
        single = (manifest.ndethist <= 1).mean()
        print(f"single-detection: {single:.0%}  (the easy majority of the stream)")
    print("\nThese are UNLABELED raw-stream alerts. Treat them as negatives only")
    print("on the base-rate argument (SNe are <1% of the stream), and cross-check")
    print("against TNS with fetch_tns.py before trusting them.")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="night to fetch, YYYYMMDD")
    parser.add_argument("--tarball", type=Path,
                        help="use an already-downloaded tarball instead")
    parser.add_argument("--n", type=int, default=20000, help="alerts to keep")
    parser.add_argument("--sample-rate", type=float, default=0.3,
                        help="keep this fraction of packets, so the sample is "
                             "spread across the night rather than the first N "
                             "alerts (one patch of sky)")
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--name", default="stream")
    parser.add_argument("--keep-tarball", action="store_true",
                        help="do not delete the tarball afterwards")
    args = parser.parse_args(argv)

    if not args.date and not args.tarball:
        parser.error("give --date YYYYMMDD or --tarball PATH")

    tarball = args.tarball or download(
        args.date, args.out / "archive" / f"ztf_public_{args.date}.tar.gz")

    manifest = build(tarball, args.n, args.out, args.name, args.sample_rate)

    if args.date and not args.keep_tarball and not manifest.empty:
        size = tarball.stat().st_size / 1e9
        tarball.unlink()
        print(f"removed {tarball.name} ({size:.1f} GB freed) -- "
              f"pass --keep-tarball to keep it")
    return 0 if not manifest.empty else 1


if __name__ == "__main__":
    sys.exit(main())
