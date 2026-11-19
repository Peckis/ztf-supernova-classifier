"""Export the accuracy plot, a stamp gallery and a static page from the db.

Writes files rather than serving them, because SQLite is not reachable from a
browser and a portfolio page just loads static assets.

    python export_dashboard.py --out dashboard/
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


def labelled_frame(conn, controls_per_night: int = 300) -> pd.DataFrame:
    """Confirmed supernovae plus presumed-negative stream controls.

    TNS-labelled objects alone are degenerate: TNS is a supernova registry, so
    its ZTF entries run ~14,500 confirmed SNe to ~610 confirmed non-SNe. On
    this pipeline that once gave 24 positives and ONE negative. Controls are
    therefore drawn from the stream on the base-rate argument, deterministically
    per night so the set is stable across rebuilds.
    """
    positives = pd.read_sql("""
        SELECT oid, MIN(night) night, MAX(p_model) p_model, MAX(p_sn) p_sn,
               MAX(tns_label) tns_label, MAX(tns_type) tns_type,
               MAX(labelled_at) labelled_at, MAX(stamp_path) stamp_path,
               NULL cat_class, NULL cat_source
        FROM alerts WHERE tns_label IS NOT NULL
        GROUP BY oid
    """, conn)

    has_cat = "cat_class" in {r[1] for r in conn.execute("PRAGMA table_info(alerts)")}
    cat_cols = ("cat_class, cat_source," if has_cat else
                "NULL cat_class, NULL cat_source,")
    # objects in TNS but untyped are excluded from both sides: they were
    # reported as candidate transients, so they are enriched in supernovae
    controls = pd.read_sql(f"""
        SELECT oid, night, p_model, p_sn, NULL tns_label, NULL tns_type,
               NULL labelled_at, stamp_path, cat_class, cat_source
        FROM (
            SELECT oid, night, p_model, p_sn, stamp_path, {cat_cols}
                   ROW_NUMBER() OVER (PARTITION BY night ORDER BY candid) rn
            FROM alerts
            WHERE tns_label IS NULL AND tns_name IS NULL
        ) WHERE rn <= ?
    """, conn, params=(controls_per_night,))
    controls["tns_label"] = 0
    controls["tns_type"] = controls.cat_class.map(
        {"star": "catalogued variable star / stellar source",
         "agn": "catalogued AGN / quasar"}).fillna("unlabelled (presumed not-SN)")

    d = pd.concat([positives, controls], ignore_index=True)
    d = d.drop_duplicates("oid").sort_values("night").reset_index(drop=True)
    # an empty result comes back as object dtype, which breaks nlargest
    for col in ("p_model", "p_sn", "tns_label"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    return d


def recall_at(prob, y, threshold=0.5) -> tuple[int, int]:
    """Needs no negatives, so it stays meaningful however lopsided the set is."""
    sn = y == 1
    return int((prob[sn] >= threshold).sum()), int(sn.sum())


def paired_interval(d: pd.DataFrame, n_boot: int = 2000) -> dict | None:
    """Error bars on the head-to-head, from a paired bootstrap.

    Only on objects BOTH models scored: the two headline AUCs rest on different
    populations, because the baseline is fetched for a subset.
    """
    sub = d[d.p_sn.notna()]
    y = sub.tns_label.to_numpy().astype(int)
    # refuse to quote an interval this pipeline has not earned -- it once
    # computed an AUC against a single negative
    if len(sub) < 20 or y.sum() < 5 or (y == 0).sum() < 5:
        return None

    from benchmark import bootstrap_auc
    ours, theirs = sub.p_model.to_numpy(), sub.p_sn.to_numpy()
    a, b, diff = bootstrap_auc(ours, theirs, y, n_boot=n_boot)
    if not len(diff):
        return None

    def ci(arr):
        return [round(float(v), 4) for v in np.percentile(arr, [2.5, 97.5])]

    return {
        "n": int(len(sub)), "n_sn": int(y.sum()),
        "model_auc": round(float(auc(ours, y)), 4), "model_ci": ci(a),
        "baseline_auc": round(float(auc(theirs, y)), 4), "baseline_ci": ci(b),
        "diff": round(float(np.mean(diff)), 4), "diff_ci": ci(diff),
        "excludes_zero": bool(np.percentile(diff, 2.5) > 0),
    }


def cumulative_history(d: pd.DataFrame) -> list[dict]:
    """Metrics over every object in the evaluation set up to each night.

    Cumulative rather than per-night: TNS confirms a handful of objects a
    night, so a nightly point is a coin flip on n=3.
    """
    out = []
    for night in sorted(d.night.unique()):
        upto = d[d.night <= night]
        y = upto.tns_label.to_numpy().astype(int)
        found, total = recall_at(upto.p_model.to_numpy(), y)
        row = {"night": str(night), "n": int(len(upto)),
               "n_sn": int(y.sum()), "n_control": int((y == 0).sum()),
               "recall": round(found / total, 4) if total else None,
               "model_auc": None, "baseline_auc": None}
        if y.sum() and (y == 0).sum():
            row["model_auc"] = round(float(auc(upto.p_model.to_numpy(), y)), 4)
            # score the baseline only where both have an opinion
            has = upto.p_sn.notna().to_numpy()
            if has.sum() and y[has].sum() and (y[has] == 0).sum():
                row["baseline_auc"] = round(
                    float(auc(upto.p_sn.to_numpy()[has], y[has])), 4)
                row["baseline_n"] = int(has.sum())
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
        ax.text(0.5, 0.5, "waiting for spectroscopic confirmations\n"
                          "(objects are scored on arrival; TNS publishes\n"
                          "what they were over the following weeks)",
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
                 "TNS-confirmed supernovae against unlabelled stream controls",
                 fontsize=12)
    ax.grid(alpha=0.25)

    # confirmed SNe, not total: they are the scarce ingredient and they
    # determine whether a point means anything
    ax2.bar(nights, [h["n_sn"] for h in history], color="0.6", width=0.6)
    ax2.set_ylabel("confirmed\nSNe", fontsize=8)
    ax2.grid(alpha=0.25, axis="y")
    ax2.tick_params(axis="x", rotation=45, labelsize=7)

    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def stretch(plane):
    """Percentile-clip then asinh, for display only."""
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
    """The four cases worth looking at, errors first.

    A gallery of successes tells you nothing the AUC did not.
    """
    have = d[d.stamp_path.notna()].copy()
    if have.empty:
        return []
    groups = []
    fp = have[(have.tns_label == 0)].nlargest(per_row, "p_model")
    fn = have[(have.tns_label == 1)].nsmallest(per_row, "p_model")
    hit = have[(have.tns_label == 1)].nlargest(per_row, "p_model")
    # take disagreements from BOTH directions, half each: ranking by the raw
    # gap looks neutral but our model is the more confident of the two, so the
    # largest gaps would all be our own wins
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
<html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ZTF supernova classifier — live</title>
<meta http-equiv="refresh" content="1800">
<style>
 :root{{--fg:#16181d;--muted:#6b7280;--line:#e5e7eb;--bg:#fff;--card:#f9fafb;
        --ok:#15803d;--okbg:#dcfce7;--warn:#b45309;--warnbg:#fef3c7;
        --bad:#b91c1c;--badbg:#fee2e2}}
 @media(prefers-color-scheme:dark){{
   :root{{--fg:#e5e7eb;--muted:#9ca3af;--line:#374151;--bg:#0f1115;--card:#171a21;
          --ok:#4ade80;--okbg:#14321f;--warn:#fbbf24;--warnbg:#3a2c0a;
          --bad:#f87171;--badbg:#3b1414}}
   img{{filter:brightness(.92)}}
 }}
 *{{box-sizing:border-box}}
 body{{font:16px/1.65 system-ui,-apple-system,Segoe UI,sans-serif;max-width:62rem;
      margin:0 auto;padding:2.5rem 1.5rem 4rem;color:var(--fg);background:var(--bg)}}
 h1{{font-size:1.75rem;margin:0 0 .3rem}}
 h2{{font-size:1.1rem;margin:2.5rem 0 .6rem;padding-bottom:.35rem;
     border-bottom:1px solid var(--line)}}
 .sub{{color:var(--muted);margin:.2rem 0 0}}
 .badge{{display:inline-block;padding:.15rem .6rem;border-radius:99px;
         font-size:.8rem;font-weight:600;vertical-align:middle}}
 .ok{{background:var(--okbg);color:var(--ok)}}
 .warn{{background:var(--warnbg);color:var(--warn)}}
 .bad{{background:var(--badbg);color:var(--bad)}}
 .stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(9rem,1fr));
         gap:.75rem;margin:1.5rem 0}}
 .stat{{background:var(--card);border:1px solid var(--line);border-radius:8px;
        padding:.8rem 1rem}}
 .stat b{{display:block;font-size:1.5rem;line-height:1.2}}
 .stat span{{color:var(--muted);font-size:.8rem}}
 img{{max-width:100%;border:1px solid var(--line);border-radius:8px;
      background:#fff}}
 .caveat{{background:var(--card);border-left:3px solid var(--warn);
          padding:.9rem 1.1rem;font-size:.9rem;border-radius:0 6px 6px 0}}
 footer{{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--line);
         color:var(--muted);font-size:.85rem}}
 a{{color:inherit}}
</style>

<h1>Continually-learning supernova classifier
  <span class="badge {status_class}">{status_text}</span></h1>
<p class="sub">A CNN scores every alert from the ZTF sky survey as it arrives.
Weeks later, spectroscopy from the Transient Name Server confirms what those
objects actually were — and the model is scored against the ALeRCE broker's own
classifier on exactly the same images.</p>

<div class="stats">
 <div class="stat"><b>{n_alerts:,}</b><span>alerts scored</span></div>
 <div class="stat"><b>{n_nights}</b><span>nights ingested</span></div>
 <div class="stat"><b>{n_labelled:,}</b><span>confirmed by spectroscopy</span></div>
 <div class="stat"><b>{recall_txt}</b><span>of those we flagged</span></div>
 <div class="stat"><b>{auc_txt} / {base_txt}</b><span>AUC, ours / ALeRCE</span></div>
</div>

<h2>Performance over time</h2>
<img src="accuracy.svg" alt="AUC over time versus the ALeRCE baseline">
<p class="sub">Cumulative over every confirmed object to date, so the curve
tightens as evidence accrues rather than jittering on three-object nights. The
bars show how many labels each point rests on.</p>

{ci_line}

{gallery}
<h2>How to read these numbers</h2>
<p class="caveat"><b>Three honest caveats.</b>
<b>Most negatives are catalogued, the rest presumed.</b> TNS is a supernova
registry — its ZTF entries run roughly 14,500 confirmed supernovae to 600
confirmed non-supernovae — so scoring against TNS negatives alone would rest on
a handful of objects. Instead the controls are ordinary stream alerts,
crossmatched against SIMBAD, Milliquas and the AAVSO variable star index:
{n_confirmed_negatives} of {n_controls} are catalogued stars or active galactic
nuclei, which positively cannot be supernovae. The rest are presumed negative
because supernovae are under 1% of the stream — true of the set, not proven of
any one object. An alert matching a catalogued <em>galaxy</em> is deliberately
never counted: supernovae explode inside galaxies, so that match is exactly what
a real one looks like.
<b>Confirmation is selective.</b> Objects only get a spectrum if someone judged
them worth one, so neither model can be measured on supernovae the whole
community missed.
<b>Only new objects are scored.</b> Alerts from objects with more than two prior
detections are skipped — this is an <em>early</em> classifier, and repeat
detections of long-known variables fall outside what it was trained on.</p>

<footer>
 Updated {updated} · pipeline {status_text_lower} · rebuilt nightly
 · <a href="{repo_url}">source and methodology on GitHub</a>
</footer>
</html>
"""


def pipeline_health() -> tuple[str, str]:
    """Reuse status.py's verdict rather than reimplementing freshness checks."""
    try:
        import status
        events = status.parse_events(14)
        verdict = max((c[0] for c in (status.check_scheduler(events),
                                      status.check_last_outcome(events),
                                      status.check_errors(events))),
                      key=lambda v: status.RANK[v])
    except Exception:
        return "unknown", "warn"
    return {"OK": ("running", "ok"), "DEGRADED": ("running", "ok"),
            "FAILING": ("degraded", "warn"),
            "STALLED": ("stalled", "bad")}.get(verdict, ("unknown", "warn"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--out", type=Path, default=Path("dashboard"))
    parser.add_argument("--repo-url",
                        default="https://github.com/Peckis/ztf-supernova-classifier",
                        help="linked from the page footer")
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

    found, total_sn = recall_at(d.p_model.to_numpy(), y) if len(d) else (0, 0)
    n_confirmed_neg = int(d.cat_class.notna().sum()) if "cat_class" in d else 0
    paired = paired_interval(d) if len(d) else None
    payload = {
        "paired": paired,
        "n_alerts": int(totals.n_alerts), "n_nights": int(totals.n_nights),
        "n_confirmed_sn": int(total_sn), "n_controls": int((y == 0).sum()) if len(y) else 0,
        "n_confirmed_negatives": n_confirmed_neg,
        "recall": round(found / total_sn, 4) if total_sn else None,
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

    health, health_class = pipeline_health()
    payload["status"] = health

    # only reference the gallery once it exists, or it renders as a broken
    # image icon on a public page
    gallery = ("""<h2>What the model actually saw</h2>
<img src="highlights.png" alt="Example stamps the model classified">
<p class="sub">Errors first — a gallery of successes would say nothing the AUC
does not. Each object shows the three channels the network receives: the new
image, the reference, and their difference.</p>""" if made else
        """<h2>What the model actually saw</h2>
<p class="sub">Waiting for the first spectroscopic confirmations. Objects are
scored the night they arrive; the Transient Name Server publishes what they
actually were over the following weeks, and the gallery appears once there is
truth to compare against.</p>""")

    if paired is None:
        ci_line = ('<p class="sub">Not enough confirmed objects yet to put an '
                   'error bar on the difference.</p>')
    else:
        verdict = ("the interval excludes zero" if paired["excludes_zero"]
                   else "the interval spans zero, so this is a tie")
        ci_line = (
            f'<p class="sub">Head-to-head on the {paired["n"]:,} objects both '
            f'models scored ({paired["n_sn"]} confirmed supernovae): ours '
            f'<b>{paired["model_auc"]:.3f}</b> '
            f'[{paired["model_ci"][0]:.3f}, {paired["model_ci"][1]:.3f}] '
            f'vs ALeRCE <b>{paired["baseline_auc"]:.3f}</b> '
            f'[{paired["baseline_ci"][0]:.3f}, {paired["baseline_ci"][1]:.3f}]. '
            f'Difference <b>{paired["diff"]:+.3f}</b> '
            f'[{paired["diff_ci"][0]:+.3f}, {paired["diff_ci"][1]:+.3f}] — '
            f'{verdict}. 95% CI from a paired bootstrap, resampling the same '
            f'objects for both models.</p>')

    (args.out / "index.html").write_text(INDEX.format(
        n_alerts=payload["n_alerts"], n_nights=payload["n_nights"],
        n_labelled=payload["n_confirmed_sn"],
        recall_txt=("—" if payload["recall"] is None
                    else f"{found}/{total_sn}"),
        auc_txt="—" if payload["model_auc"] is None else f"{payload['model_auc']:.3f}",
        base_txt="—" if payload["baseline_auc"] is None else f"{payload['baseline_auc']:.3f}",
        updated=payload["updated"], status_text=health,
        status_text_lower=health.lower(), status_class=health_class,
        repo_url=args.repo_url, gallery=gallery, ci_line=ci_line,
        n_confirmed_negatives=payload["n_confirmed_negatives"],
        n_controls=payload["n_controls"]), encoding="utf-8")

    print(f"wrote {args.out}/  (accuracy.json, accuracy.svg, index.html"
          f"{', highlights.png' if made else ''})")
    print(f"  {payload['n_alerts']:,} alerts over {payload['n_nights']} night(s), "
          f"{payload['n_labelled']} labelled")
    if payload["model_auc"] is None:
        print("  no AUC yet: needs TNS-confirmed objects of BOTH classes")
    else:
        print(f"  AUC ours {payload['model_auc']} vs baseline "
              f"{payload['baseline_auc']}")
    if paired:
        print(f"  paired head-to-head on {paired['n']:,} objects "
              f"({paired['n_sn']} SNe):")
        print(f"    ours     {paired['model_auc']:.4f}  "
              f"[{paired['model_ci'][0]:.4f}, {paired['model_ci'][1]:.4f}]")
        print(f"    ALeRCE   {paired['baseline_auc']:.4f}  "
              f"[{paired['baseline_ci'][0]:.4f}, {paired['baseline_ci'][1]:.4f}]")
        print(f"    diff    {paired['diff']:+.4f}  "
              f"[{paired['diff_ci'][0]:+.4f}, {paired['diff_ci'][1]:+.4f}]  "
              + ("<- excludes zero" if paired["excludes_zero"]
                 else "<- SPANS ZERO, do not claim a win"))
    print(f"  evaluation set: {payload['n_confirmed_sn']} confirmed SNe vs "
          f"{payload['n_controls']} controls "
          f"({payload['n_confirmed_negatives']} catalogue-confirmed, rest presumed)")
    if payload["recall"] is not None:
        print(f"  recall @0.5   : {found}/{total_sn} = {payload['recall']:.1%}")
    if not made:
        print("  no highlights yet: needs labelled objects with stored stamps")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
