"""Check whether the two classes differ in nuisance ways a CNN could exploit.

Positives and negatives come from different pipelines, so if they also differ
in sky background, brightness or noise, the network can score well by learning
"brighter = supernova" instead of what a supernova looks like. Measured off the
pixels, because that is what the model sees.

    python audit_shortcuts.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path("data")
C = 31  # stamp centre


def stamp_features(path):
    arr = np.load(DATA / path)
    sci, tmpl, diff = arr[0], arr[1], arr[2]

    def robust_noise(p):
        finite = p[np.isfinite(p)]
        if finite.size == 0:
            return np.nan
        return 1.4826 * np.median(np.abs(finite - np.median(finite))) + 1e-9

    core = diff[C - 3:C + 4, C - 3:C + 4]
    noise = robust_noise(diff)
    return {
        "sky": np.nanmedian(sci),                       # background level
        "sci_peak": np.nanmax(sci) - np.nanmedian(sci),  # source brightness
        "diff_snr": np.nanmax(np.abs(core)) / noise,     # detection strength
        "diff_noise": noise,                             # image depth
        "tmpl_core": np.nanmedian(tmpl[C - 2:C + 3, C - 2:C + 3])
                     - np.nanmedian(tmpl),               # pre-existing source
    }


def load(name, label):
    d = pd.read_csv(DATA / name)
    feats = pd.DataFrame([stamp_features(p) for p in d.stamp_path])
    feats["label"] = label
    feats["oid"] = d.oid.values
    return feats


def main():
    frames = [load("manifest_positives.csv", 1),
              load("manifest_unfiltered.csv", 0)]
    hard = DATA / "manifest_hard_negatives.csv"
    if hard.exists():
        frames.append(load(hard.name, 0))
    f = pd.concat(frames, ignore_index=True)

    cols = ["sky", "sci_peak", "diff_snr", "diff_noise", "tmpl_core"]
    print(f"{'feature':<12} {'positives(med)':>15} {'negatives(med)':>15} "
          f"{'ratio':>8} {'AUC':>7}")
    print("-" * 62)
    for c in cols:
        pos = f.loc[f.label == 1, c].dropna()
        neg = f.loc[f.label == 0, c].dropna()
        # AUC of this one feature: 1.0 would separate the classes on its own
        vals = np.concatenate([pos, neg])
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        order = vals.argsort()
        ranks = np.empty(len(vals)); ranks[order] = np.arange(1, len(vals) + 1)
        a = (ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
        print(f"{c:<12} {pos.median():>15.3f} {neg.median():>15.3f} "
              f"{pos.median()/neg.median() if neg.median() else float('nan'):>8.2f} "
              f"{max(a, 1-a):>7.3f}")

    print("\nAUC near 0.5 = harmless. Near 0.9+ = that single number separates")
    print("the classes by itself, and the CNN will happily learn it instead of")
    print("morphology. sky near 0.5 means both samples share observing")
    print("conditions, which is the failure mode that would matter most.")

    # the decisive test: if a tiny model on these five numbers alone reaches
    # the CNN's AUC, the CNN is mostly learning how the samples were collected
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score

    X = f[cols].to_numpy()
    y = f.label.to_numpy()
    auc = cross_val_score(HistGradientBoostingClassifier(max_iter=200),
                          X, y, cv=5, scoring="roc_auc")
    print(f"\nnuisance-only model (these 5 numbers, no image): "
          f"AUC = {auc.mean():.3f} +/- {auc.std():.3f}")
    print("Compare against the CNN's AUC. A large gap means the CNN is learning")
    print("real morphology; a small gap means it is mostly riding these.")


if __name__ == "__main__":
    main()
