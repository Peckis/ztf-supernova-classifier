"""Phase 2, step 3: export the money plot and a look inside the model.

Reads data/pipeline.db and writes static files -- no server, no live database
connection. A portfolio page just loads them, which is why this is an export
step rather than a web app: SQLite is not reachable from a browser, and neither
is Postgres, so something has to render either way.

    python export_dashboard.py --out dashboard/

Produces:
  accuracy.json    the numbers, for embedding anywhere
  accuracy.svg     performance over time, ours vs the ALeRCE baseline
  highlights.png   actual stamps: hits, misses, false alarms, disagreements
  index.html       a single page tying them together

On the plot: per-night accuracy is not worth plotting. TNS confirms a handful
of ZTF objects a night, so a nightly point is a coin flip on n=3. Instead this
shows a cumulative curve -- every object labelled up to that date -- which
starts noisy and tightens honestly as evidence accrues, and never implies more
precision than the sample supports. The label count is drawn underneath so a
reader can see exactly how much evidence each point rests on.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from ingest_night import DB_PATH  # noqa: E402

CHANNELS = ("science", "template", "difference")


def auc(prob, y):
    order = np.argsort(prob)
    ranks = np.empty(len(prob), dtype=float)
    ranks[order] = np.arange(1, len(prob) + 1)
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def labelled_frame(conn) -> pd.DataFrame:
    """One row per labelled object, earliest alert kept."""
    d = pd.read_sql("""
        SELECT oid, MIN(night) night, MAX(p_model) p_model, MAX(p_sn) p_sn,
               MAX(tns_label) tns_label, MAX(tns_type) tns_type,
               MAX(labelled_at) labelled_at,
               MAX(stamp_path) stamp_path
        FROM alerts WHERE tns_label IS NOT NULL
        GROUP BY oid ORDER BY night
    """, conn)
    # An empty result comes back as object dtype, which makes nlargest and the
    # numpy comparisons blow up rather than simply yield nothing. Everything
    # downstream is written to cope with zero rows, so force the types.
    for col in ("p_model", "p_sn", "tns_label"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    return d


def cumulative_history(d: pd.DataFrame) -> list[dict]:
    """Metrics over every object labelled up to each night."""
    out = []
    for night in sorted(d.night.unique()):
        upto = d[d.night <= night]
        y = upto.tns_label.to_numpy().astype(int)
        row = {"night": str(night), "n": int(len(upto)),
               "n_sn": int(y.sum()), "model_auc": None, "baseline_auc": None}
        if y.sum() and (y == 0).sum():
            row["model_auc"] = round(float(auc(upto.p_model.to_numpy(), y)), 4)
            has = upto.p_sn.notna()
            if has.sum() and y[has].sum() and (y[has] == 0).sum():
                row["baseline_auc"] = round(
                    float(auc(upto.p_sn[has].to_numpy(), y[has])), 4)
        out.append(row)
    return out


def write_plot(history, path: Path, min_points: int = 2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(9, 5.5), sharex=True, height_ratios=[3, 1])
    nights = [h["night"] for h in history]

    usable = [h for h in history if h["model_auc"] is not None]
    if len(usable) < min_points:
        ax.text(0.5, 0.5, "not enough labelled objects yet\n"
                          "(needs both classes confirmed by TNS)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="dimgray")
        ax.set_ylim(0, 1)
    else:
        ax.plot([h["night"] for h in usable], [h["model_auc"] for h in usable],
                marker="o", label="our CNN", color="#1f77b4")
        b = [h for h in usable if h["baseline_auc"] is not None]
        if b:
            ax.plot([h["night"] for h in b], [h["baseline_auc"] for h in b],
                    marker="s", label="ALeRCE stamp_classifier", color="#d62728")
        ax.legend(loc="lower right", fontsize=9)
        ax.set_ylim(0.4, 1.02)
        ax.axhline(0.5, ls=":", lw=1, color="gray")
    ax.set_ylabel("AUC (cumulative)")
    ax.set_title("Supernova identification vs the broker baseline\n"
                 "cumulative over every TNS-confirmed object to date",
                 fontsize=12)
    ax.grid(alpha=0.25)

    # the evidence behind each point, so nobody reads a curve built on n=3
    ax2.bar(nights, [h["n"] for h in history], color="0.6", width=0.6)
    ax2.set_ylabel("objects\nlabelled", fontsize=8)
    ax2.grid(alpha=0.25, axis="y")
    ax2.tick_params(axis="x", rotation=45, labelsize=7)

    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def stretch(plane):
    """Percentile-clip then asinh, as in show_stamps.py."""
    finite = plane[np.isfinite(plane)]
    if finite.size == 0:
        return np.zeros_like(plane)
    lo, hi = np.percentile(finite, [1.0, 99.9])
    if hi <= lo:
        hi = lo + 1.0
    out = np.arcsinh((np.clip(np.nan_to_num(plane, nan=lo), lo, hi) - lo)
                     / (hi - lo) * 10.0)
    return out / out.max() if out.max() > 0 else out


def pick_highlights(d: pd.DataFrame, per_row: int) -> list[tuple]:
    """The four cases worth looking at, most interesting first.

    False alarms and misses come before hits deliberately: a gallery of
    successes tells you nothing you did not already know from the AUC, whereas
    the errors are where the model's actual behaviour is legible.
    """
    have = d[d.stamp_path.notna()].copy()
    if have.empty:
        return []
    groups = []
    fp = have[(have.tns_label == 0)].nlargest(per_row, "p_model")
    fn = have[(have.tns_label == 1)].nsmallest(per_row, "p_model")
    hit = have[(have.tns_label == 1)].nlargest(per_row, "p_model")
    # Take the disagreements from BOTH directions, half each. Ranking by the
    # raw gap looks neutral but is not: our model is the more confident of the
    # two, so the largest gaps are nearly all cases where we scored high and
    # ALeRCE low -- a gallery of our own wins.
    dis = have[have.p_sn.notna()].copy()
    if len(dis):
        half = max(1, per_row // 2)
        we_higher = dis.assign(gap=dis.p_model - dis.p_sn).nlargest(half, "gap")
        they_higher = dis.assign(gap=dis.p_sn - dis.p_model).nlargest(
            per_row - half, "gap")
        dis = pd.concat([we_higher, they_higher]).drop_duplicates("oid")

    groups.append(("false alarms", "high score, TNS says not a supernova", fp))
    groups.append(("missed", "low score, TNS confirmed a supernova", fn))
    groups.append(("caught", "high score, TNS confirmed a supernova", hit))
    groups.append(("disagreements",
                   "biggest splits, both directions: we say SN / they say SN",
                   dis))
    return [g for g in groups if len(g[2])]


def write_highlights(d: pd.DataFrame, data_dir: Path, path: Path, per_row: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = pick_highlights(d, per_row)
    if not groups:
        return False

    rows, cols = len(groups), per_row * 3
    fig, axes = plt.subplots(rows, cols, figsize=(1.35 * cols, 1.75 * rows),
                             squeeze=False,
                             gridspec_kw={"wspace": 0.05, "hspace": 0.45})
    fig.subplots_adjust(left=0.15, top=0.90)
    fig.suptitle("What the model actually saw   —   "
                 "science / template / difference", fontsize=13)

    for r, (title, subtitle, sel) in enumerate(groups):
        for e in range(per_row):
            for c in range(3):
                axes[r][e * 3 + c].axis("off")
        for e, obj in enumerate(sel.itertuples()):
            arr = np.load(data_dir / obj.stamp_path)
            for c in range(3):
                ax = axes[r][e * 3 + c]
                ax.axis("on")
                ax.imshow(stretch(arr[c]), cmap="gray", origin="lower")
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_color("0.6")
                if r == 0:
                    ax.set_title(CHANNELS[c], fontsize=7, pad=2)
            baseline = "n/a" if pd.isna(obj.p_sn) else f"{obj.p_sn:.2f}"
            axes[r][e * 3].set_xlabel(
                f"{obj.oid}\nours {obj.p_model:.2f} | ALeRCE {baseline}"
                f" | {obj.tns_type}", fontsize=6, labelpad=2, loc="left")

        y = 1 - (r + 0.5) / rows * 0.86 - 0.03
        fig.text(0.135, y, title, fontsize=12, ha="right", va="center",
                 fontweight="bold")
        fig.text(0.135, y - 0.045, subtitle, fontsize=6.5, ha="right",
                 va="center", color="dimgray")

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


INDEX = """<!doctype html>
<meta charset="utf-8"><title>ZTF supernova classifier</title>
<style>
 body{{font:16px/1.6 system-ui,sans-serif;max-width:56rem;margin:3rem auto;
      padding:0 1.5rem;color:#1a1a1a}}
 h1{{font-size:1.6rem;margin-bottom:.2rem}} .sub{{color:#666;margin-top:0}}
 img{{max-width:100%;border:1px solid #ddd;border-radius:6px}}
 .stat{{display:inline-block;margin:0 2rem .5rem 0}}
 .stat b{{display:block;font-size:1.5rem}} .stat span{{color:#666;font-size:.85rem}}
 .caveat{{background:#fff8e6;border-left:3px solid #e0a800;padding:.8rem 1rem;
          font-size:.9rem}}
</style>
<h1>Continually-learning supernova classifier</h1>
<p class="sub">A CNN scores every ZTF alert as it arrives; TNS spectroscopy
confirms the truth weeks later; the model is measured against the ALeRCE
broker's own classifier on the same objects.</p>

<div>
 <div class="stat"><b>{n_alerts:,}</b><span>alerts scored</span></div>
 <div class="stat"><b>{n_nights}</b><span>nights ingested</span></div>
 <div class="stat"><b>{n_labelled:,}</b><span>TNS-confirmed</span></div>
 <div class="stat"><b>{auc_txt}</b><span>AUC vs baseline {base_txt}</span></div>
</div>

<h2>Performance over time</h2>
<img src="accuracy.svg" alt="AUC over time versus the ALeRCE baseline">
<p class="sub">Cumulative over every confirmed object to date. The bars show how
many labels each point rests on.</p>

<h2>Inside the model</h2>
<img src="highlights.png" alt="Example stamps the model classified">
<p class="sub">Errors first: a gallery of successes would say nothing the AUC
does not.</p>

<p class="caveat"><b>Read with care.</b> Objects are only labelled if someone
thought them worth a spectrum, so neither model can be measured on supernovae
the whole community missed. Alerts from objects with more than two prior
detections are not scored -- the model is an early classifier and they fall
outside what it was trained on.</p>
<p class="sub">Updated {updated}.</p>
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--out", type=Path, default=Path("dashboard"))
    parser.add_argument("--per-row", type=int, default=4,
                        help="example objects per highlight row")
    args = parser.parse_args()

    if not args.db.exists():
        sys.exit(f"{args.db} does not exist -- run ingest_night.py first")
    args.out.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(args.db)

    totals = pd.read_sql(
        "SELECT COUNT(*) n_alerts, COUNT(DISTINCT night) n_nights FROM alerts",
        conn).iloc[0]
    d = labelled_frame(conn)
    history = cumulative_history(d)

    y = d.tns_label.to_numpy().astype(int) if len(d) else np.array([])
    model_auc = auc(d.p_model.to_numpy(), y) if len(d) and y.sum() and (y == 0).sum() else float("nan")
    has = d.p_sn.notna() if len(d) else pd.Series(dtype=bool)
    base_auc = float("nan")
    if len(d) and has.sum() and y[has].sum() and (y[has] == 0).sum():
        base_auc = auc(d.p_sn[has].to_numpy(), y[has])

    payload = {
        "n_alerts": int(totals.n_alerts), "n_nights": int(totals.n_nights),
        "n_labelled": int(len(d)),
        "model_auc": None if np.isnan(model_auc) else round(float(model_auc), 4),
        "baseline_auc": None if np.isnan(base_auc) else round(float(base_auc), 4),
        "history": history,
        "updated": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    }
    (args.out / "accuracy.json").write_text(json.dumps(payload, indent=2))

    write_plot(history, args.out / "accuracy.svg")
    made = write_highlights(d, args.db.parent, args.out / "highlights.png",
                            args.per_row)

    (args.out / "index.html").write_text(INDEX.format(
        n_alerts=payload["n_alerts"], n_nights=payload["n_nights"],
        n_labelled=payload["n_labelled"],
        auc_txt="—" if payload["model_auc"] is None else f"{payload['model_auc']:.3f}",
        base_txt="—" if payload["baseline_auc"] is None else f"{payload['baseline_auc']:.3f}",
        updated=payload["updated"]), encoding="utf-8")

    print(f"wrote {args.out}/  (accuracy.json, accuracy.svg, index.html"
          f"{', highlights.png' if made else ''})")
    print(f"  {payload['n_alerts']:,} alerts over {payload['n_nights']} night(s), "
          f"{payload['n_labelled']} labelled")
    if payload["model_auc"] is None:
        print("  no AUC yet: needs TNS-confirmed objects of BOTH classes")
    else:
        print(f"  AUC ours {payload['model_auc']} vs baseline "
              f"{payload['baseline_auc']}")
    if not made:
        print("  no highlights yet: needs labelled objects with stored stamps")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
