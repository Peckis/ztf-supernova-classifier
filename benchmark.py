"""Head-to-head against the ALeRCE stamp_classifier, with error bars.

Reads data/predictions.csv (written by train.py) and answers one question
honestly: does our model actually beat the broker's, or is the gap noise?

Three things this does that a bare accuracy number does not:

  1. Bootstraps confidence intervals on both AUCs *and on their difference*,
     resampling the same objects for both models so the comparison is paired.
  2. Projects precision to the real ~0.2% base rate. The test split is heavily
     enriched in supernovae; precision measured on it is meaningless for an
     observer deciding where to point a telescope.
  3. Reports a nuisance floor -- how well simple photometry does with no image
     at all -- so a high AUC cannot be mistaken for morphological skill.

    python benchmark.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
BASE_RATE = 0.002   # measured: SNe are ~0.2% of raw stream alerts


def auc(prob, y):
    order = prob.argsort()
    ranks = np.empty(len(prob), dtype=np.float64)
    ranks[order] = np.arange(1, len(prob) + 1)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def bootstrap_auc(prob_a, prob_b, y, n_boot=2000, seed=0):
    """Paired bootstrap: resample objects, score both models on the same draw.

    Pairing matters. The two models are evaluated on identical objects, so their
    errors are correlated; comparing independent confidence intervals would
    overstate the uncertainty on the *difference*, which is the quantity we
    actually care about.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    a_s, b_s, d_s = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        a = auc(prob_a[idx], yy)
        b = auc(prob_b[idx], yy)
        a_s.append(a); b_s.append(b); d_s.append(a - b)
    return np.array(a_s), np.array(b_s), np.array(d_s)


def rates(prob, y, threshold):
    guess = (prob >= threshold).astype(int)
    tp = int(((guess == 1) & (y == 1)).sum())
    fp = int(((guess == 1) & (y == 0)).sum())
    fn = int(((guess == 0) & (y == 1)).sum())
    tn = int(((guess == 0) & (y == 0)).sum())
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return recall, fpr, precision


def best_threshold(prob, y):
    grid = np.linspace(0.01, 0.99, 197)
    def f1(t):
        r, _, p = rates(prob, y, t)
        return 2 * p * r / (p + r) if p + r else 0.0
    return float(grid[int(np.argmax([f1(t) for t in grid]))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DATA / "predictions.csv")
    parser.add_argument("--boot", type=int, default=2000)
    args = parser.parse_args()

    d = pd.read_csv(args.predictions)
    d = d[d.p_sn.notna()].reset_index(drop=True)
    y = d.label.to_numpy()
    ours = d.p_model.to_numpy()
    theirs = d.p_sn.to_numpy()

    print(f"test objects: {len(y)}   confirmed SNe: {int(y.sum())} "
          f"({y.mean():.1%})   negatives: {int((y == 0).sum())}")

    # --- AUC with paired bootstrap -------------------------------------
    a_s, b_s, d_s = bootstrap_auc(ours, theirs, y, n_boot=args.boot)
    lo_a, hi_a = np.percentile(a_s, [2.5, 97.5])
    lo_b, hi_b = np.percentile(b_s, [2.5, 97.5])
    lo_d, hi_d = np.percentile(d_s, [2.5, 97.5])

    print("\n--- AUC (95% CI from a paired bootstrap) ---")
    print(f"  our CNN                 {auc(ours, y):.4f}  [{lo_a:.4f}, {hi_a:.4f}]")
    print(f"  ALeRCE stamp_classifier {auc(theirs, y):.4f}  [{lo_b:.4f}, {hi_b:.4f}]")
    print(f"  difference (ours-theirs){np.mean(d_s):+.4f}  [{lo_d:+.4f}, {hi_d:+.4f}]")
    if lo_d > 0:
        print("  -> the interval excludes zero: our model is genuinely ahead")
    elif hi_d < 0:
        print("  -> the interval excludes zero: the baseline is genuinely ahead")
    else:
        print("  -> the interval spans zero: this is a TIE, do not claim a win")

    # --- operating points, both tuned ----------------------------------
    t_ours, t_theirs = best_threshold(ours, y), best_threshold(theirs, y)
    print("\n--- operating points (each model at its own best-F1 threshold) ---")
    print(f"{'model':<26} {'thr':>5} {'recall':>7} {'FPR':>8} {'prec':>7}  prec@0.2%")
    print("-" * 70)
    for name, p, t in [("our CNN", ours, t_ours),
                       ("ALeRCE stamp_classifier", theirs, t_theirs)]:
        r, fpr, prec = rates(p, y, t)
        # precision the same model would show on a stream that is 0.2% SNe
        real = (BASE_RATE * r / (BASE_RATE * r + (1 - BASE_RATE) * fpr)
                if fpr > 0 else 1.0)
        print(f"{name:<26} {t:>5.2f} {r:>7.3f} {fpr:>8.4f} {prec:>7.3f}  {real:>8.1%}")

    print("\n--- what this means on a real night (100,000 alerts, ~200 SNe) ---")
    for name, p, t in [("our CNN", ours, t_ours),
                       ("ALeRCE", theirs, t_theirs)]:
        r, fpr, _ = rates(p, y, t)
        found, false = r * 200, fpr * 99800
        print(f"  {name:<8} finds {found:>5.0f}/200 SNe, {false:>6.0f} false alarms "
              f"({false / max(found, 1):>5.1f} per real SN)")

    # --- breakdown by where the negatives came from --------------------
    if "source" in d.columns:
        print("\n--- false-positive rate by negative type ---")
        for src, g in d[d.label == 0].groupby("source"):
            fp_ours = (g.p_model >= t_ours).mean()
            fp_theirs = (g.p_sn >= t_theirs).mean()
            print(f"  {src:<8} n={len(g):>5}  ours={fp_ours:>6.1%}  "
                  f"ALeRCE={fp_theirs:>6.1%}")

    print("\nCaveats that belong in any write-up of these numbers:")
    print("  * Positives are TNS-CONFIRMED supernovae, so both models are scored")
    print("    on SNe someone already thought worth a spectrum. Neither can be")
    print("    measured on supernovae the whole community missed.")
    print("  * A nuisance-only model (5 photometric numbers, no image) reaches")
    print("    AUC ~0.92 here -- see audit_shortcuts.py. The classes differ in")
    print("    brightness and environment, not only in being supernovae, so this")
    print("    benchmark is easier than the real streaming task.")


if __name__ == "__main__":
    main()
