"""Phase 2, step 1: score one night of the ZTF alert stream and store it.

Runs the CNN over every alert in a nightly public archive tarball and writes the
predictions to SQLite. Ground truth is not available yet -- TNS confirms these
objects over the following weeks, and label_from_tns.py fills it in later. That
delay is the whole point of the project.

Two sizing facts drive the design, both measured rather than assumed:

  * A night holds ~55,000 alerts on a 2.6 GB file, up to ~340,000 on a 16 GB one.
  * Stamps are 47 KB each, so keeping all of them would be ~1 TB/year.

So predictions for every alert go to the database (~200 bytes each), and stamps
are kept only for the top --keep-stamps candidates by model score. With recall
~0.97, the handful TNS later confirms will almost always be inside that set,
which is what retraining needs.

    python ingest_night.py --date 20260110
    python ingest_night.py --tarball part.tar.gz --dry-run

The tarball is streamed, never unpacked, and deleted afterwards unless
--keep-tarball is given.
"""

from __future__ import annotations

import argparse
import heapq
import re
import sqlite3
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from fetch_ztf_archive import download, iter_packets, packet_to_stamp  # noqa: E402
from train import StampCNN, apply_normalisation  # noqa: E402

DATA = Path("data")
DB_PATH = DATA / "pipeline.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    candid        INTEGER PRIMARY KEY,   -- one row per alert, not per object
    oid           TEXT NOT NULL,
    night         TEXT NOT NULL,
    jd            REAL,
    ra            REAL,
    dec           REAL,
    magpsf        REAL,
    fid           INTEGER,
    rb            REAL,                  -- ZTF's own real/bogus score
    drb           REAL,
    ndethist      INTEGER,
    p_model       REAL,                  -- our CNN
    model_version TEXT,
    stamp_path    TEXT,                  -- NULL unless this alert made the cut
    -- everything below arrives later, from TNS and (on demand) from ALeRCE
    tns_name      TEXT,
    tns_type      TEXT,
    tns_label     INTEGER,
    labelled_at   TEXT,
    p_sn          REAL
);
CREATE INDEX IF NOT EXISTS idx_alerts_oid     ON alerts(oid);
CREATE INDEX IF NOT EXISTS idx_alerts_night   ON alerts(night);
CREATE INDEX IF NOT EXISTS idx_alerts_pmodel  ON alerts(p_model);
CREATE INDEX IF NOT EXISTS idx_alerts_label   ON alerts(tns_label);

CREATE TABLE IF NOT EXISTS runs (
    night         TEXT PRIMARY KEY,
    ingested_at   TEXT,
    n_alerts      INTEGER,
    n_unusable    INTEGER,
    n_stamps_kept INTEGER,
    model_version TEXT,
    seconds       REAL
);
"""


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    """Open the pipeline database, creating it on first use."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def load_model(checkpoint: Path, device: str):
    """Load weights AND the normalisation they were trained with.

    Refuses a bare state_dict: a model applied with the wrong statistics gives
    plausible-looking probabilities that are quietly meaningless, which is worse
    than an error.
    """
    blob = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(blob, dict) or "state_dict" not in blob:
        sys.exit(f"{checkpoint} predates normalisation-saving. Re-run train.py "
                 f"so the checkpoint carries norm_mean/norm_std.")
    model = StampCNN().to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    version = f"auc{blob.get('test_auc', float('nan')):.4f}"
    return model, blob["norm_mean"], blob["norm_std"], version


def score_batch(model, arrays, mean, std, device) -> np.ndarray:
    X = apply_normalisation(np.stack(arrays), mean, std)
    with torch.no_grad():
        logits = model(torch.from_numpy(X).float().to(device))
        return torch.softmax(logits, dim=1)[:, 1].cpu().numpy()


def ingest(tarball: Path, night: str, model, mean, std, version: str,
           device: str, keep_stamps: int, batch_size: int,
           dry_run: bool, max_ndethist: int = 2) -> dict:
    stamp_dir = DATA / "stream_stamps" / night
    conn = None if dry_run else connect()

    started = time.perf_counter()
    n_alerts = n_unusable = 0
    pending, rows = [], []
    # bounded min-heap of the best-scoring alerts, so memory stays flat no
    # matter how large the night is
    best: list = []

    def flush():
        nonlocal rows
        if not pending:
            return
        probs = score_batch(model, [a for _, a in pending], mean, std, device)
        for (meta, arr), p in zip(pending, probs):
            meta["p_model"] = float(p)
            rows.append(meta)
            if keep_stamps:
                item = (float(p), meta["candid"], arr)
                if len(best) < keep_stamps:
                    heapq.heappush(best, item)
                elif p > best[0][0]:
                    heapq.heapreplace(best, item)
        pending.clear()

    n_skipped = 0
    bar = tqdm(desc=f"scoring[{night}]", unit="alert", disable=None)
    for packet in iter_packets(tarball):
        c = packet.get("candidate", {})

        # Only score alerts from objects the model was trained to judge.
        # We trained on FIRST-detection stamps, where ndethist is ~1. Two thirds
        # of the stream is repeat detections of long-known objects (ndethist in
        # the hundreds), which is far outside that distribution -- and it shows:
        # the model gave 0.9988 to the 367th detection of a known AGN. An early
        # classifier's job is new objects, so anything already well-known is
        # skipped rather than scored badly.
        nd = c.get("ndethist")
        if max_ndethist and nd is not None and nd > max_ndethist:
            n_skipped += 1
            continue

        try:
            arr = packet_to_stamp(packet)
        except Exception:
            n_unusable += 1
            continue
        pending.append(({
            "candid": packet.get("candid"),
            "oid": packet.get("objectId"),
            "night": night,
            "jd": c.get("jd"), "ra": c.get("ra"), "dec": c.get("dec"),
            "magpsf": c.get("magpsf"), "fid": c.get("fid"),
            "rb": c.get("rb"), "drb": c.get("drb"),
            "ndethist": c.get("ndethist"),
        }, arr))
        n_alerts += 1
        if len(pending) >= batch_size:
            flush()
            bar.update(batch_size)
            bar.set_postfix(unusable=n_unusable, refresh=False)
    flush()
    bar.close()

    kept = {}
    if keep_stamps and not dry_run:
        stamp_dir.mkdir(parents=True, exist_ok=True)
        for _, candid, arr in best:
            path = stamp_dir / f"{candid}.npy"
            np.save(path, arr)
            kept[candid] = str(path.relative_to(DATA)).replace("\\", "/")

    for r in rows:
        r["model_version"] = version
        r["stamp_path"] = kept.get(r["candid"])

    if not dry_run:
        conn.executemany(
            "INSERT OR REPLACE INTO alerts "
            "(candid, oid, night, jd, ra, dec, magpsf, fid, rb, drb, "
            " ndethist, p_model, model_version, stamp_path) "
            "VALUES (:candid, :oid, :night, :jd, :ra, :dec, :magpsf, :fid, "
            " :rb, :drb, :ndethist, :p_model, :model_version, :stamp_path)",
            rows)
        conn.execute(
            "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?)",
            (night, datetime.now(timezone.utc).isoformat(timespec="seconds"),
             n_alerts, n_unusable, len(kept), version,
             time.perf_counter() - started))
        conn.commit()
        conn.close()

    scores = np.array([r["p_model"] for r in rows]) if rows else np.array([])
    return {
        "n_alerts": n_alerts, "n_unusable": n_unusable, "n_skipped": n_skipped,
        "n_stamps_kept": len(kept), "seconds": time.perf_counter() - started,
        "scores": scores,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="night to ingest, YYYYMMDD")
    parser.add_argument("--tarball", type=Path, help="use a local tarball instead")
    parser.add_argument("--checkpoint", type=Path, default=Path("model.pth"))
    parser.add_argument("--keep-stamps", type=int, default=1000,
                        help="how many top-scoring alerts keep their stamps "
                             "(0 disables). ~47 KB each.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-ndethist", type=int, default=2,
                        help="skip alerts whose object already has more than "
                             "this many prior detections (0 scores everything). "
                             "The model was trained on first detections; repeat "
                             "alerts for long-known objects are out of "
                             "distribution and score badly.")
    parser.add_argument("--keep-tarball", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="score but write nothing")
    args = parser.parse_args(argv)

    if not args.date and not args.tarball:
        parser.error("give --date YYYYMMDD or --tarball PATH")
    if args.date:
        night = args.date
    else:
        # pull the date out of the filename rather than slicing blindly -- a
        # wrong night silently files the alerts under a bogus date, and the
        # runs table would then show a gap that never gets re-ingested
        found = re.search(r"(\d{8})", args.tarball.name)
        if not found:
            parser.error(f"cannot find a YYYYMMDD date in {args.tarball.name}; "
                         f"pass --date explicitly")
        night = found.group(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, mean, std, version = load_model(args.checkpoint, device)
    print(f"device: {device}   model: {version}")

    tarball = args.tarball or download(
        args.date, DATA / "archive" / f"ztf_public_{args.date}.tar.gz")

    stats = ingest(tarball, night, model, mean, std, version, device,
                   args.keep_stamps, args.batch_size, args.dry_run,
                   args.max_ndethist)

    s = stats["scores"]
    print(f"\nnight {night}: {stats['n_alerts']:,} alerts scored, "
          f"{stats['n_skipped']:,} skipped as already-known "
          f"(ndethist>{args.max_ndethist}), {stats['n_unusable']} unusable, "
          f"in {stats['seconds']/60:.1f} min")
    if len(s):
        print(f"  p_model: median {np.median(s):.4f}   "
              f">=0.5: {(s >= 0.5).sum():,} ({(s >= 0.5).mean():.2%})   "
              f">=0.9: {(s >= 0.9).sum():,}")
        print(f"  stamps kept: {stats['n_stamps_kept']:,}")
    if args.dry_run:
        print("  (dry run -- nothing written)")

    if args.date and not args.keep_tarball and tarball.exists():
        size = tarball.stat().st_size / 1e9
        tarball.unlink()
        print(f"  removed {tarball.name} ({size:.1f} GB freed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
