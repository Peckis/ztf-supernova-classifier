"""Fetch labeled ZTF stamp cutouts from the ALeRCE broker.

For each object we pull the three 63x63 cutouts (science / template / difference)
and stack them into a single [3, 63, 63] float32 array -- the same layout an RGB
image has, which is why the MNIST CNN transfers over with only conv1 changing
from 1 input channel to 3.

Outputs
-------
data/stamps/<oid>.npy   one [3, 63, 63] float32 array per object
data/manifest.csv       one row per object: label, ALeRCE baseline probability,
                        sky position, and the path to the .npy

Run `python fetch_data.py --inspect ZTF17aaaazwi` to dump one object's stamps
without downloading a dataset.

Note on labels: the `label` column comes from ALeRCE's own stamp_classifier,
i.e. from the very baseline we intend to beat. It is a *proxy* label, good
enough to get the pipeline and the CNN working, but a model trained on it can
at best imitate the baseline. Real ground truth arrives later from TNS
(see tns_label, left empty here -- the ALeRCE client exposes no TNS field).
"""

from __future__ import annotations

import argparse
import gzip
import sys
import time
import warnings
from pathlib import Path

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from alerce.core import Alerce
from alerce.exceptions import (CandidError, FormatValidationError,
                               ObjectNotFoundError, ParseError)
from tqdm import tqdm

# Permanent failures: the object genuinely is not there, or the request is
# malformed. Retrying these is pure waste -- and with a long backoff it is
# expensive waste. A 404 once cost 85s of sleeping before we gave up on it.
# BadGzipFile means the API returned a JSON error where FITS bytes should be,
# i.e. no stamps are stored for that particular candid. Also permanent -- and
# expensive to get wrong, since these are common when asking for a specific
# epoch rather than the default one.
PERMANENT_ERRORS = (ObjectNotFoundError, CandidError, FormatValidationError,
                    ParseError, gzip.BadGzipFile)

# these fetches run for hours -- show rate and ETA, and keep it readable when
# stdout is a log file rather than a terminal
BAR = ("{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
       "{postfix}")

# The ALeRCE FITS cards use a >8 char keyword (STAMP_TYPE), which astropy warns
# about on every single download. The files are fine; silence the noise.
warnings.filterwarnings("ignore", module="astropy.io.fits")

# Channel order in the stacked array. We look these up by the STAMP_TYPE header
# rather than trusting HDU order, so a broker-side reordering can't silently
# scramble the channels.
CHANNELS = ("science", "template", "difference")
STAMP_SIZE = 63  # verified against live data; see --inspect

# ZTF marks "no data" pixels inconsistently: NaN in the science/difference
# planes but FITS null padding (-FLT_MAX, ~-3.4e38) in the template plane. Left
# alone, that sentinel is a finite float the CNN will happily ingest and blow
# up on. We normalise anything below this to NaN so missing data has exactly
# one representation downstream.
SENTINEL_THRESHOLD = -1e30

# ALeRCE's stamp_classifier (our baseline) emits exactly these classes.
STAMP_CLASSES = ("SN", "AGN", "VS", "asteroid", "bogus")

# Objects carry stamp_classifier results under several versions, and coverage is
# uneven -- most have 1.0.4, some only 1.0.0. Benchmarking has to pick one
# consistently, so we take the first version available in this order and record
# which one was used. Probabilities sum to 1 within a version.
VERSION_PREFERENCE = ("stamp_classifier_1.0.4", "stamp_classifier_1.0.0")

MANIFEST_COLUMNS = [
    "oid",
    "label",
    "alerce_class",
    # probability of alerce_class -- NOT of SN. For a bogus object this is
    # P(bogus). Use p_sn for anything that compares against our model.
    "alerce_class_probability",
    "p_sn",
    "p_sn_version",
    # no tns_label column here on purpose -- ground truth lives in
    # tns_labels.csv, written by fetch_tns.py. A placeholder here would collide
    # with the real column on merge and get silently suffixed to tns_label_x.
    "ra",
    "dec",
    "ndet",
    "firstmjd",
    "lastmjd",
    "n_missing",
    "missing_frac",
    "stamp_path",
]


class RateLimiter:
    """Global cap on requests per second, shared across worker threads.

    Learned the hard way: a 24-object burst at 8 threads scaled linearly with no
    errors, which looked like "not rate-limited". Running 14,574 objects the same
    way produced 10,453 APIErrors and got the query endpoint refusing us
    entirely. Short bursts never reveal a *sustained*-rate limit. A sequential
    run (~0.9 req/s) was demonstrably fine for 3 hours, so that is the anchor.
    """

    def __init__(self, per_second: float):
        self.min_interval = 1.0 / per_second if per_second > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self):
        if not self.min_interval:
            return
        with self._lock:
            now = time.monotonic()
            sleep_for = max(0.0, self._next - now)
            self._next = max(now, self._next) + self.min_interval
        if sleep_for:
            time.sleep(sleep_for)


# Default 2 req/s: about twice the rate a 3-hour sequential run sustained
# without a single throttle, and ~8x below what got us blocked.
LIMITER = RateLimiter(2.0)


def _retry(fn, *args, attempts: int = 4, **kwargs):
    """Call fn behind the rate limiter, backing off hard on failure.

    Backoff is deliberately long (5s, 20s, 60s). When the server is throttling,
    retrying after 1-2 seconds just adds to the load that caused the problem.
    """
    delays = [5.0, 20.0, 60.0]
    for attempt in range(attempts):
        LIMITER.wait()
        try:
            return fn(*args, **kwargs)
        except PERMANENT_ERRORS:
            # not transient -- no amount of waiting makes a 404 into data
            raise
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(delays[min(attempt, len(delays) - 1)])


def query_candidates(
    client: Alerce,
    class_name: str,
    probability: float,
    n: int,
    page_size: int = 100,
) -> pd.DataFrame:
    """Page through query_objects until we have n *distinct* objects of one class.

    ALeRCE returns one row per classifier *version*, so the same oid comes back
    more than once with different probabilities. Deduplicating matters: the same
    object landing in both the train and test split would leak, and quietly
    inflate every accuracy number we report against the baseline. We keep the
    highest probability per object -- that is the baseline's best call on it,
    which is the number benchmark.py should have to beat.
    """
    pages = []
    page = 1
    while True:
        df = _retry(
            client.query_objects,
            survey="ztf",
            classifier="stamp_classifier",
            class_name=class_name,
            probability=probability,
            page=page,
            page_size=page_size,
            format="pandas",
        )
        if df is None or len(df) == 0:
            break

        pages.append(df)
        combined = (
            pd.concat(pages, ignore_index=True)
            .sort_values("probability", ascending=False)
            .drop_duplicates(subset="oid", keep="first")
        )
        if len(combined) >= n:
            return combined.head(n).reset_index(drop=True)

        page += 1
        time.sleep(0.2)

    if not pages:
        return pd.DataFrame()
    return (
        pd.concat(pages, ignore_index=True)
        .sort_values("probability", ascending=False)
        .drop_duplicates(subset="oid", keep="first")
        .head(n)
        .reset_index(drop=True)
    )


def earliest_candids(client: Alerce, oid: str, limit: int = 5):
    """Candids of an object's earliest detections, oldest first.

    ALeRCE's stamp_classifier scores the FIRST alert -- that is what an early
    classifier is for. But get_stamps() without a candid returns an inconsistent
    epoch: measured across 8 objects, 4 gave the first detection, 2 the second,
    and 2 the LAST. For a supernova the last epoch can be near peak, far easier
    to classify than a marginal first detection, which would hand us an
    advantage ALeRCE never had.

    We return several because stamps are not stored for every alert -- asking
    for the true first candid fails ~40% of the time with a JSON error where the
    FITS should be. Walking forward from the earliest gives the closest epoch
    that actually exists.
    """
    det = _retry(client.query_detections, oid, survey="ztf", format="pandas")
    if det is None or len(det) == 0:
        raise ValueError("no detections returned")
    return list(det.sort_values("mjd").candid.iloc[:limit])


def fetch_stamp_array(client: Alerce, oid: str, candid=None) -> np.ndarray:
    """Download one object's cutouts as a [3, 63, 63] float32 array.

    Raises ValueError if the cutouts are the wrong size or a channel is missing
    -- both happen for real (objects near a CCD edge get truncated stamps), and
    a silently misshapen array would blow up much later inside the model.
    """
    if candid is None:
        hdul = _retry(client.get_stamps, oid, survey="ztf")
    else:
        hdul = _retry(client.get_stamps, oid, candid=candid, survey="ztf")

    by_type = {}
    for hdu in hdul:
        stamp_type = hdu.header.get("STAMP_TYPE")
        if stamp_type in CHANNELS and hdu.data is not None:
            by_type[stamp_type] = hdu.data

    missing = [c for c in CHANNELS if c not in by_type]
    if missing:
        raise ValueError(f"missing channel(s): {', '.join(missing)}")

    planes = []
    for channel in CHANNELS:
        plane = by_type[channel]
        if plane.shape != (STAMP_SIZE, STAMP_SIZE):
            raise ValueError(f"{channel} is {plane.shape}, expected "
                             f"({STAMP_SIZE}, {STAMP_SIZE})")
        # FITS is big-endian (>f4); torch.from_numpy rejects that, so convert
        # to native-endian float32 once here rather than in the training loop.
        plane = np.asarray(plane, dtype=np.float32)
        # Collapse null-padding sentinels and infinities into NaN so that
        # "missing" is a single thing train.py has to handle.
        plane[~np.isfinite(plane) | (plane < SENTINEL_THRESHOLD)] = np.nan
        planes.append(plane)

    return np.stack(planes, axis=0)


def merge_manifest(new: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Merge into any existing manifest instead of overwriting it.

    A throttled rerun once replaced a manifest that had 94% of its p_sn values
    with one that had 6%, losing hours of good data. A partial run must never be
    able to destroy a better earlier one: rows are unioned on oid, and for each
    object a real p_sn always beats a NaN.
    """
    if not path.exists() or new.empty:
        return new

    old = pd.read_csv(path)
    combined = pd.concat([old, new], ignore_index=True)
    # NaN p_sn sorts last, so keep='first' prefers a row that actually has one
    combined["_has_p"] = combined.p_sn.notna().astype(int)
    combined = (combined.sort_values("_has_p", ascending=False)
                        .drop_duplicates("oid", keep="first")
                        .drop(columns="_has_p")
                        .reset_index(drop=True))
    gained = len(combined) - len(old)
    recovered = int(combined.p_sn.notna().sum()) - int(old.p_sn.notna().sum())
    print(f"merged with existing {path.name}: {len(old)} -> {len(combined)} rows "
          f"(+{gained} new, {recovered:+d} with p_sn)")
    return combined


def fetch_one(oid: str, stamp_dir: Path, out_dir: Path, overwrite: bool,
              first_detection: bool = False,
              known_p_sn: dict | None = None) -> tuple:
    """Stamp + baseline p_sn for one object. Returns (row, failure).

    Each call builds its own Alerce client: these run on worker threads and the
    client holds a requests.Session, which is not safe to share.
    """
    client = Alerce()
    path = stamp_dir / f"{oid}.npy"
    candid, epoch_index = None, -1
    try:
        if path.exists() and not overwrite:
            arr = np.load(path)
        elif first_detection:
            # walk forward from the earliest detection to the first epoch that
            # actually has stamps stored
            arr, last_exc = None, None
            for i, cid in enumerate(earliest_candids(client, oid)):
                try:
                    arr = fetch_stamp_array(client, oid, candid=cid)
                    candid, epoch_index = cid, i
                    break
                except PERMANENT_ERRORS as exc:
                    last_exc = exc
                    continue
            if arr is None:
                raise last_exc or ValueError("no epoch had stamps")
            np.save(path, arr)
        else:
            arr = fetch_stamp_array(client, oid)
            np.save(path, arr)
    except Exception as exc:
        return None, (oid, f"{type(exc).__name__}: {exc}")

    failure = None
    # p_sn is a property of the object, not of which epoch's stamp we fetched,
    # so a value from an earlier run is still valid -- and skipping the call
    # removes a third of the requests on a re-fetch
    cached = known_p_sn.get(oid) if known_p_sn else None
    if cached is not None:
        p_sn, p_sn_version = cached
    else:
        try:
            p_sn, p_sn_version = fetch_p_sn(client, oid)
        except Exception as exc:
            failure = (oid, f"p_sn: {type(exc).__name__}: {exc}")
            p_sn, p_sn_version = float("nan"), ""

    return {
        "oid": oid,
        "p_sn": p_sn,
        "p_sn_version": p_sn_version,
        "candid": candid,
        "epoch_index": epoch_index,   # 0 = the true first detection
        "n_missing": int(np.isnan(arr).sum()),
        "missing_frac": round(float(np.isnan(arr).mean()), 4),
        "stamp_path": str(path.relative_to(out_dir)).replace("\\", "/"),
    }, failure


def fetch_many(oids, stamp_dir: Path, out_dir: Path, overwrite: bool,
               workers: int, desc: str, first_detection: bool = False,
               known_p_sn: dict | None = None) -> tuple[list, list]:
    """Fetch a list of oids concurrently.

    Measured: a single request takes ~2.3s regardless of load, and 8 threads
    give ~6.5x throughput with no extra failures -- so this is latency-bound,
    not server-throttled. 8 is a deliberate ceiling; ALeRCE is a free public
    service and there is no reason to lean on it harder.
    """
    oids = list(oids)
    rows, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_one, o, stamp_dir, out_dir, overwrite,
                               first_detection, known_p_sn): o
                   for o in oids}
        bar = tqdm(as_completed(futures), total=len(futures), desc=desc,
                   bar_format=BAR, mininterval=2.0, miniters=10, unit="obj")
        for future in bar:
            row, failure = future.result()
            if row is not None:
                rows.append(row)
            if failure is not None:
                failures.append(failure)
            bar.set_postfix(ok=len(rows), failed=len(failures), refresh=False)
    return rows, failures


def query_unfiltered(client: Alerce, mjd_start: float, mjd_end: float, n: int,
                     page_size: int = 100, windows: int = 24,
                     window_days: float = 15.0) -> pd.DataFrame:
    """Objects first detected across a date range, with NO classifier filter.

    This is the honest evaluation sample. Everything else in this file queries
    by `classifier="stamp_classifier", class_name=...`, which means we only ever
    see objects the baseline already committed to -- never the supernovae it
    missed, and only its confident rejects as negatives. A model cannot be
    fairly measured on a sample its competitor selected.

    The range is walked as several narrow windows rather than one wide query:
    a single multi-year span makes the API time out with a 504, and spreading
    the sample over time also stops the model learning detector-epoch quirks
    instead of what a transient looks like.
    """
    if windows < 1:
        windows = 1
    starts = np.linspace(mjd_start, max(mjd_start, mjd_end - window_days), windows)
    per_window = max(1, n // windows)

    collected, seen = [], set()
    for w, start in enumerate(starts, start=1):
        end = start + window_days
        got_here = 0
        page = 1
        while got_here < per_window:
            try:
                df = _retry(
                    client.query_objects,
                    survey="ztf",
                    firstmjd=[float(start), float(end)],
                    page=page,
                    page_size=page_size,
                    format="pandas",
                )
            except Exception as exc:
                print(f"  window {w}/{windows} MJD {start:.0f}: {type(exc).__name__}, skipping")
                break
            if df is None or len(df) == 0:
                break

            df = df[~df.oid.isin(seen)]
            if len(df):
                seen.update(df.oid)
                collected.append(df)
                got_here += len(df)
            page += 1
            time.sleep(0.2)

        print(f"  window {w}/{windows} MJD {start:.0f}-{end:.0f}: "
              f"{got_here} objects (total {len(seen)})")
        if len(seen) >= n:
            break

    if not collected:
        return pd.DataFrame()
    return (pd.concat(collected, ignore_index=True)
            .drop_duplicates(subset="oid").head(n).reset_index(drop=True))


def fetch_p_sn(client: Alerce, oid: str) -> tuple[float, str]:
    """The baseline's P(SN) for one object -- the number benchmark.py must beat.

    This needs its own call: query_objects only returns the probability of the
    class the object was matched on, so for a negative it reports P(bogus) or
    P(AGN), never P(SN).
    """
    probs = _retry(client.query_probabilities, oid, survey="ztf", format="pandas")
    stamp = probs[(probs.classifier_name == "stamp_classifier")
                  & (probs.class_name == "SN")]
    if stamp.empty:
        return float("nan"), ""

    for version in VERSION_PREFERENCE:
        row = stamp[stamp.classifier_version == version]
        if not row.empty:
            return float(row.probability.iloc[0]), version

    # some object carries a version we don't know about -- take it, but say which
    row = stamp.iloc[0]
    return float(row.probability), str(row.classifier_version)


def inspect(oid: str) -> None:
    """Print one object's stamps in detail. Use this before any bulk fetch."""
    client = Alerce()
    hdul = _retry(client.get_stamps, oid, survey="ztf")

    print(f"{oid}: {len(hdul)} HDUs")
    for i, hdu in enumerate(hdul):
        data = hdu.data
        print(f"  hdu{i}  STAMP_TYPE={hdu.header.get('STAMP_TYPE')!r:>14}  "
              f"shape={data.shape}  dtype={data.dtype}  "
              f"nan={int(np.isnan(data).sum())}  "
              f"min={np.nanmin(data):.4g}  max={np.nanmax(data):.4g}")

    arr = fetch_stamp_array(client, oid)
    print(f"\nstacked: shape={arr.shape} dtype={arr.dtype} "
          f"nan={int(np.isnan(arr).sum())}")
    print("per-channel mean/std:")
    for name, plane in zip(CHANNELS, arr):
        print(f"  {name:>10}: mean={np.nanmean(plane):9.3f} "
              f"std={np.nanstd(plane):9.3f}")


def build_dataset(
    classes: list[str],
    n_per_class: int,
    probability: float,
    out_dir: Path,
    positive_class: str = "SN",
    overwrite: bool = False,
) -> pd.DataFrame:
    client = Alerce()
    stamp_dir = out_dir / "stamps"
    stamp_dir.mkdir(parents=True, exist_ok=True)

    rows, failures = [], []
    seen: set[str] = set()  # an oid must not appear under two classes either

    for class_name in classes:
        print(f"\n[{class_name}] querying up to {n_per_class} objects "
              f"(p >= {probability}) ...")
        candidates = query_candidates(client, class_name, probability, n_per_class)
        if candidates.empty:
            print(f"[{class_name}] no objects returned, skipping")
            continue
        print(f"[{class_name}] got {len(candidates)}, downloading stamps")

        bar = tqdm(candidates.itertuples(index=False), total=len(candidates),
                   desc=f"stamps[{class_name}]", bar_format=BAR,
                   mininterval=2.0, miniters=10, unit="obj")
        for row in bar:
            oid = row.oid
            if oid in seen:
                continue
            seen.add(oid)
            path = stamp_dir / f"{oid}.npy"

            if path.exists() and not overwrite:
                arr = np.load(path)
            else:
                try:
                    arr = fetch_stamp_array(client, oid)
                except Exception as exc:
                    failures.append((oid, f"{type(exc).__name__}: {exc}"))
                    continue
                np.save(path, arr)
                time.sleep(0.1)  # be polite to a free public API

            try:
                p_sn, p_sn_version = fetch_p_sn(client, oid)
            except Exception as exc:
                failures.append((oid, f"p_sn: {type(exc).__name__}: {exc}"))
                p_sn, p_sn_version = float("nan"), ""

            rows.append({
                "oid": oid,
                "label": int(class_name == positive_class),
                "alerce_class": class_name,
                "alerce_class_probability": float(row.probability),
                "p_sn": p_sn,
                "p_sn_version": p_sn_version,
                "ra": float(row.meanra),
                "dec": float(row.meandec),
                "ndet": int(row.ndet),
                "firstmjd": float(row.firstmjd),
                "lastmjd": float(row.lastmjd),
                "n_missing": int(np.isnan(arr).sum()),
                "missing_frac": round(float(np.isnan(arr).mean()), 4),
                "stamp_path": str(path.relative_to(out_dir)).replace("\\", "/"),
            })

            bar.set_postfix(ok=len(rows), failed=len(failures), refresh=False)

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    if not manifest.empty:
        manifest_path = out_dir / "manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        print(f"\nwrote {manifest_path} ({len(manifest)} rows)")

    if failures:
        print(f"\n{len(failures)} object(s) failed:")
        for oid, err in failures[:10]:
            print(f"  {oid}: {err}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")

    return manifest


def query_hard_negatives(client: Alerce, n: int, min_span: float = 365.0,
                         recent_mjd: float = 60500.0, page_size: int = 100,
                         windows: int = 12, window_days: float = 30.0,
                         first_start: float = 58200.0,
                         first_end: float = 60000.0) -> pd.DataFrame:
    """Objects detected across more than a year -- confident hard negatives.

    A supernova fades in weeks to months. Anything ZTF has kept detecting for
    over a year is physically not one: it is a variable star, an AGN, or a
    recurring artifact. That verdict comes from ZTF's own detection history,
    NOT from any classifier, so unlike ALeRCE's class labels it is non-circular.

    These are the negatives our stream sample is short of -- 80% of that sample
    is single-detection junk, which is easy. AGN and variable stars are the
    genuinely hard cases, and they are what the model actually confuses with
    supernovae.
    """
    starts = np.linspace(first_start, first_end - window_days, windows)
    collected, seen = [], set()

    for w, start in enumerate(starts, start=1):
        page = 1
        got_here = 0
        while got_here < max(1, n // windows):
            try:
                df = _retry(
                    client.query_objects,
                    survey="ztf",
                    firstmjd=[float(start), float(start + window_days)],
                    lastmjd=[recent_mjd, 99999.0],
                    page=page,
                    page_size=page_size,
                    format="pandas",
                )
            except Exception as exc:
                print(f"  window {w}/{windows}: {type(exc).__name__}, skipping")
                break
            if df is None or len(df) == 0:
                break

            # verify the span client-side rather than trusting the query
            df = df[(df.lastmjd - df.firstmjd) > min_span]
            df = df[~df.oid.isin(seen)]
            if len(df):
                seen.update(df.oid)
                collected.append(df)
                got_here += len(df)
            page += 1
            time.sleep(0.2)

        print(f"  window {w}/{windows} first-detected MJD {start:.0f}: "
              f"{got_here} long-span objects (total {len(seen)})")
        if len(seen) >= n:
            break

    if not collected:
        return pd.DataFrame()
    return (pd.concat(collected, ignore_index=True)
            .drop_duplicates(subset="oid").head(n).reset_index(drop=True))


def build_hard_negatives(n: int, out_dir: Path, overwrite: bool = False,
                         workers: int = 8, min_span: float = 365.0) -> pd.DataFrame:
    """Fetch long-span objects as hard negatives, excluding any confirmed SN."""
    client = Alerce()
    stamp_dir = out_dir / "stamps"
    stamp_dir.mkdir(parents=True, exist_ok=True)

    print(f"querying up to {n} objects with detection span > {min_span:.0f}d ...")
    candidates = query_hard_negatives(client, n, min_span=min_span)
    if candidates.empty:
        print("no objects returned")
        return pd.DataFrame()

    # belt and braces: never let a TNS-confirmed supernova in as a negative
    tns_path = out_dir / "tns_ztf_all.csv"
    if tns_path.exists():
        tns = pd.read_csv(tns_path)
        confirmed = set(tns[tns.tns_label == 1].oid)
        before = len(candidates)
        candidates = candidates[~candidates.oid.isin(confirmed)]
        if before != len(candidates):
            print(f"dropped {before - len(candidates)} TNS-confirmed SNe from "
                  f"the hard-negative pool")

    print(f"got {len(candidates)}, downloading stamps")
    rows, failures = fetch_many(candidates.oid, stamp_dir, out_dir, overwrite,
                                workers, "stamps[hard-neg]")

    manifest = pd.DataFrame(rows)
    if not manifest.empty:
        meta = candidates[["oid", "meanra", "meandec", "ndet",
                           "firstmjd", "lastmjd"]].rename(
            columns={"meanra": "ra", "meandec": "dec"})
        manifest = manifest.merge(meta, on="oid", how="left")
        path = out_dir / "manifest_hard_negatives.csv"
        manifest = merge_manifest(manifest, path)
        manifest.to_csv(path, index=False)
        span = manifest.lastmjd - manifest.firstmjd
        print(f"\nwrote {path} ({len(manifest)} rows)")
        print(f"median detection span: {span.median():.0f} days, "
              f"median ndet: {manifest.ndet.median():.0f}")
        print(f"baseline p_sn >= 0.5 on: {int((manifest.p_sn >= 0.5).sum())} "
              f"({(manifest.p_sn >= 0.5).mean():.1%}) -- these are the hard ones")

    if failures:
        counts = {}
        for _, err in failures:
            counts[err.split(":")[0]] = counts.get(err.split(":")[0], 0) + 1
        print("failure types:", counts)

    return manifest


def build_unfiltered(mjd_start: float, mjd_end: float, n: int, out_dir: Path,
                     overwrite: bool = False, workers: int = 8) -> pd.DataFrame:
    """Build the honest evaluation set: date-sampled, unlabeled until TNS.

    Written to manifest_unfiltered.csv, kept apart from the training manifest --
    these rows have no alerce_class to label them with, on purpose.
    """
    client = Alerce()
    stamp_dir = out_dir / "stamps"
    stamp_dir.mkdir(parents=True, exist_ok=True)

    print(f"querying up to {n} objects first detected in MJD "
          f"[{mjd_start}, {mjd_end}], no class filter ...")
    candidates = query_unfiltered(client, mjd_start, mjd_end, n)
    if candidates.empty:
        print("no objects returned")
        return pd.DataFrame()
    print(f"got {len(candidates)}, downloading stamps")

    rows, failures = fetch_many(candidates.oid, stamp_dir, out_dir, overwrite,
                                workers, "stamps[unfiltered]")
    manifest = pd.DataFrame(rows)
    if not manifest.empty:
        # keep the query metadata: lastmjd-firstmjd is what identifies the
        # physically-confident hard negatives (an object detected across years
        # cannot be a supernova)
        meta = candidates[["oid", "meanra", "meandec", "ndet",
                           "firstmjd", "lastmjd"]].rename(
            columns={"meanra": "ra", "meandec": "dec"})
        manifest = manifest.merge(meta, on="oid", how="left")
    if not manifest.empty:
        path = out_dir / "manifest_unfiltered.csv"
        manifest = merge_manifest(manifest, path)
        manifest.to_csv(path, index=False)
        print(f"\nwrote {path} ({len(manifest)} rows)")
        print(f"baseline P(SN) >= 0.5 on: {int((manifest.p_sn >= 0.5).sum())} "
              f"of {len(manifest)} ({(manifest.p_sn >= 0.5).mean():.1%})")
        print("These rows have NO labels. Run fetch_tns.py to attach ground truth;")
        print("objects TNS never heard of stay unlabeled, they are not negatives.")

    if failures:
        print(f"\n{len(failures)} object(s) failed (first 5):")
        for oid, err in failures[:5]:
            print(f"  {oid}: {err}")

    return manifest


def build_from_oids(oid_file: Path, n: int, out_dir: Path, name: str,
                    only_label: float | None = None, overwrite: bool = False,
                    seed: int = 0, workers: int = 8,
                    first_detection: bool = False,
                    reuse_p_sn: Path | None = None) -> pd.DataFrame:
    """Fetch stamps for an explicit list of oids -- e.g. TNS-confirmed supernovae.

    This inverts the pipeline. Instead of asking ALeRCE which objects it thinks
    are supernovae (and inheriting its mistakes as labels), we start from
    spectroscopically confirmed truth and go fetch those images.
    """
    client = Alerce()
    # first-detection stamps go in their own directory: they are a different
    # epoch of the same object, so writing them over the existing ones would
    # destroy the ability to compare the two and re-run the old benchmark
    stamp_dir = out_dir / ("stamps_first" if first_detection else "stamps")
    stamp_dir.mkdir(parents=True, exist_ok=True)

    wanted = pd.read_csv(oid_file)
    if only_label is not None and "tns_label" in wanted.columns:
        wanted = wanted[wanted.tns_label == only_label]
    wanted = wanted.drop_duplicates("oid")
    # sample rather than take the head: the file is ordered by TNS name, so the
    # first N would all be from the earliest discovery years
    if len(wanted) > n:
        wanted = wanted.sample(n, random_state=seed)
    print(f"{len(wanted)} oids to fetch from {oid_file}")

    known = None
    if reuse_p_sn and reuse_p_sn.exists():
        prev = pd.read_csv(reuse_p_sn)
        prev = prev[prev.p_sn.notna()]
        known = {r.oid: (float(r.p_sn), str(r.p_sn_version))
                 for r in prev.itertuples()}
        print(f"reusing {len(known)} p_sn values from {reuse_p_sn.name} "
              f"(saves one request per object)")

    rows, failures = fetch_many(wanted.oid, stamp_dir, out_dir, overwrite,
                                workers, f"stamps[{name}]",
                                first_detection=first_detection,
                                known_p_sn=known)
    manifest = pd.DataFrame(rows)
    if not manifest.empty:
        path = out_dir / f"manifest_{name}.csv"
        manifest = merge_manifest(manifest, path)
        manifest.to_csv(path, index=False)
        print(f"\nwrote {path} ({len(manifest)} rows, {len(failures)} failed)")
        print("labels stay in tns_labels.csv / tns_ztf_all.csv -- join on oid")

    if failures:
        counts = {}
        for _, err in failures:
            counts[err.split(":")[0]] = counts.get(err.split(":")[0], 0) + 1
        print("failure types:", counts)

    return manifest


def summarise(manifest: pd.DataFrame) -> None:
    if manifest.empty:
        print("empty manifest -- nothing fetched")
        return
    print("\n--- summary ---")
    print(f"objects        : {len(manifest)}")
    print(f"positives (SN) : {int(manifest.label.sum())}")
    print(f"negatives      : {int((manifest.label == 0).sum())}")
    print("by ALeRCE class:")
    for name, count in manifest.alerce_class.value_counts().items():
        print(f"  {name:>10}: {count}")

    assert manifest.oid.is_unique, "duplicate oids in manifest -- split would leak"

    print("baseline P(SN) by class (this is what benchmark.py compares against):")
    for name, group in manifest.groupby("alerce_class"):
        print(f"  {name:>10}: median={group.p_sn.median():.3f}  "
              f"missing={int(group.p_sn.isna().sum())}")
    print("p_sn versions used:", manifest.p_sn_version.value_counts().to_dict())

    any_missing = int((manifest.n_missing > 0).sum())
    heavily_masked = manifest[manifest.missing_frac > 0.2]
    print(f"stamps with missing px: {any_missing}")
    if len(heavily_masked):
        print(f"  of which >20% missing: {len(heavily_masked)} "
              f"(consider dropping these in train.py)")
        for r in heavily_masked.itertuples():
            print(f"    {r.oid}: {r.missing_frac:.0%}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", metavar="OID",
                        help="dump one object's stamps and exit")
    parser.add_argument("--classes", nargs="+", default=["SN", "bogus"],
                        choices=list(STAMP_CLASSES),
                        help="stamp_classifier classes to fetch (default: SN bogus)")
    parser.add_argument("--n-per-class", type=int, default=50,
                        help="objects per class (default: 50)")
    parser.add_argument("--probability", type=float, default=0.7,
                        help="minimum baseline probability (default: 0.7)")
    parser.add_argument("--out", type=Path, default=Path("data"),
                        help="output directory (default: data)")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-download stamps that are already on disk")
    parser.add_argument("--unfiltered", action="store_true",
                        help="sample by date with NO classifier filter -- the "
                             "honest evaluation set (writes manifest_unfiltered.csv)")
    parser.add_argument("--mjd", type=float, nargs=2, metavar=("START", "END"),
                        default=[60800.0, 60830.0],
                        help="first-detection MJD window for --unfiltered")
    parser.add_argument("--n", type=int, default=500,
                        help="objects to fetch for --unfiltered / --oids-from")
    parser.add_argument("--oids-from", type=Path, metavar="CSV",
                        help="fetch stamps for an explicit oid list (e.g. "
                             "data/tns_ztf_all.csv) instead of querying by class")
    parser.add_argument("--name", default="from_oids",
                        help="suffix for the output manifest_<name>.csv")
    parser.add_argument("--only-label", type=float, metavar="N",
                        help="with --oids-from, keep only rows whose tns_label is N")
    parser.add_argument("--reuse-p-sn", type=Path, metavar="CSV",
                        help="take p_sn from an existing manifest instead of "
                             "re-querying it. The baseline probability belongs "
                             "to the object, not to the epoch we fetched, so an "
                             "earlier value is still valid -- and this removes "
                             "a third of the requests on a re-fetch.")
    parser.add_argument("--first-detection", action="store_true",
                        help="pin stamps to each object's FIRST detection, which "
                             "is the epoch ALeRCE's stamp_classifier scored. "
                             "Without this the epoch is inconsistent and we get "
                             "an unearned advantage. Writes to stamps_first/.")
    parser.add_argument("--hard-negatives", action="store_true",
                        help="fetch objects detected across >1 year -- variable "
                             "stars and AGN, which physically cannot be SNe. "
                             "Non-circular hard negatives (writes "
                             "manifest_hard_negatives.csv)")
    parser.add_argument("--min-span", type=float, default=365.0,
                        help="minimum detection span in days for --hard-negatives")
    parser.add_argument("--workers", type=int, default=3,
                        help="concurrent requests (default 3). 8 workers at "
                             "full tilt got us throttled off the API; the real "
                             "throughput cap is --rate, not this.")
    parser.add_argument("--rate", type=float, default=2.0,
                        help="max requests per second across all workers "
                             "(default 2.0). A 3-hour sequential run sustained "
                             "~0.9/s with zero throttling; 8 workers unthrottled "
                             "produced 10k APIErrors. Raise this only with "
                             "evidence from a long run, not a short burst.")
    args = parser.parse_args(argv)

    if args.inspect:
        inspect(args.inspect)
        return 0

    global LIMITER
    LIMITER = RateLimiter(args.rate)
    print(f"rate limit: {args.rate}/s across {args.workers} workers")

    if args.oids_from:
        manifest = build_from_oids(args.oids_from, args.n, args.out,
                                   args.name, args.only_label,
                                   overwrite=args.overwrite,
                                   workers=args.workers,
                                   first_detection=args.first_detection,
                                   reuse_p_sn=args.reuse_p_sn)
        return 0 if not manifest.empty else 1

    if args.hard_negatives:
        manifest = build_hard_negatives(args.n, args.out,
                                        overwrite=args.overwrite,
                                        workers=args.workers,
                                        min_span=args.min_span)
        return 0 if not manifest.empty else 1

    if args.unfiltered:
        manifest = build_unfiltered(args.mjd[0], args.mjd[1], args.n, args.out,
                                    overwrite=args.overwrite,
                                    workers=args.workers)
        return 0 if not manifest.empty else 1

    manifest = build_dataset(
        classes=args.classes,
        n_per_class=args.n_per_class,
        probability=args.probability,
        out_dir=args.out,
        overwrite=args.overwrite,
    )
    summarise(manifest)
    return 0 if not manifest.empty else 1


if __name__ == "__main__":
    sys.exit(main())
