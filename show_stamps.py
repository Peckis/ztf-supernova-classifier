"""Render example stamps per class, so you can see what the model sees.

Astronomical images need a stretch to be legible: raw values span orders of
magnitude, so a linear grey scale shows nothing. We clip to robust percentiles
and apply asinh -- the same compression train.py feeds the network.

    python show_stamps.py            # 3 examples per class
    python show_stamps.py --n 5
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path("data")
CHANNELS = ("science", "template", "difference")
CLASSES = ("SN", "AGN", "VS", "bogus")

HINT = {
    "SN": "new point source, absent from template,\noften offset from its host galaxy",
    "AGN": "point source sitting exactly on\na galaxy nucleus",
    "VS": "source ALREADY in the template --\na star that changed brightness",
    "bogus": "artifacts: dipoles, bad subtraction\nresiduals near bright stars",
}


def stretch(plane):
    """Percentile-clip then asinh.

    The upper cut is 99.9, not 99.5: a point source covers only ~15 of the 3969
    pixels, so clipping the top 0.5% clips away the very thing we want to see.
    """
    finite = plane[np.isfinite(plane)]
    if finite.size == 0:
        return np.zeros_like(plane)
    lo, hi = np.percentile(finite, [1.0, 99.9])
    if hi <= lo:
        hi = lo + 1.0
    clipped = np.clip(np.nan_to_num(plane, nan=lo), lo, hi)
    out = np.arcsinh((clipped - lo) / (hi - lo) * 10.0)
    return out / out.max() if out.max() > 0 else out


def central_snr(arr):
    """Peak of the difference image near the centre, in units of its own noise.

    Used to pick clear teaching examples rather than marginal detections.
    """
    diff = np.nan_to_num(arr[2])
    core = diff[26:37, 26:37]
    noise = 1.4826 * np.median(np.abs(diff - np.median(diff))) + 1e-6
    return float(np.abs(core).max() / noise)


def pick(manifest, class_name, n):
    """Clearest examples of one class.

    For SN we require a short detection span first: an object detected across
    years cannot be a supernova, and ~38% of the SN class is contaminated that
    way. Showing those as textbook examples would teach the wrong thing.
    """
    rows = manifest[manifest.alerce_class == class_name]
    if class_name == "SN":
        plausible = rows[(rows.lastmjd - rows.firstmjd) < 100]
        if len(plausible) >= n:
            rows = plausible
    rows = rows.nlargest(min(60, len(rows)), "alerce_class_probability").copy()
    rows["snr"] = [central_snr(np.load(DATA / p)) for p in rows.stamp_path]
    return rows.nlargest(n, "snr")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=3, help="examples per class")
    parser.add_argument("--out", type=Path, default=Path("stamp_examples.png"))
    args = parser.parse_args()

    manifest = pd.read_csv(DATA / "manifest.csv")
    rows, cols = len(CLASSES), args.n * 3

    fig, axes = plt.subplots(rows, cols, figsize=(1.45 * cols, 1.75 * rows),
                             gridspec_kw={"wspace": 0.06, "hspace": 0.30})
    fig.subplots_adjust(left=0.16, top=0.90, bottom=0.02, right=0.99)
    fig.suptitle("ALeRCE stamps by class   —   science / template / difference",
                 fontsize=14)

    for r, class_name in enumerate(CLASSES):
        chosen = pick(manifest, class_name, args.n)
        for e, obj in enumerate(chosen.itertuples()):
            arr = np.load(DATA / obj.stamp_path)
            for c, channel in enumerate(CHANNELS):
                ax = axes[r, e * 3 + c]
                ax.imshow(stretch(arr[c]), cmap="gray", origin="lower")
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color("0.6")
                if r == 0:
                    ax.set_title(channel, fontsize=8, pad=3)
                # # red cross marks the alert position -- where the transient is
                # ax.plot(31, 31, marker="+", color="red", markersize=8,
                #         markeredgewidth=0.7, alpha=0.75)
            axes[r, e * 3].set_xlabel(f"{obj.oid}  snr={obj.snr:.0f}",
                                      fontsize=6, labelpad=1.5)

        # row label in the left margin, outside the axes
        y = 1 - (r + 0.5) / rows * 0.88 - 0.02
        fig.text(0.145, y, class_name, fontsize=15, ha="right", va="center",
                 fontweight="bold")
        fig.text(0.145, y - 0.055, HINT[class_name], fontsize=6.8, ha="right",
                 va="center", color="dimgray")

    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
