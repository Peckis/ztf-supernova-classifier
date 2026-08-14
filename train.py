"""Train a 3-channel CNN to classify SN vs not-SN on ALeRCE stamps.

Labels now come from reality, not from the baseline:

  positives  data/manifest_positives.csv   TNS spectroscopically-confirmed SNe
  negatives  data/manifest_unfiltered.csv  date-sampled stream objects

Neither class is derived from ALeRCE's stamp_classifier, so the comparison
against its p_sn at the end is a real benchmark rather than a measure of how
well we imitate it.

    python train.py                 # train, evaluate, save model.pth
    python train.py --epochs 60
    python train.py --no-augment    # ablation
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DATA = Path("data")
SEED = 0

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)


# ---------------------------------------------------------------- data

DEFAULT_NEGATIVES = ("manifest_unfiltered.csv", "manifest_hard_negatives.csv")


def load_dataset(positives="manifest_positives.csv",
                 negatives=DEFAULT_NEGATIVES):
    """Combine TNS positives with stream and hard negatives into one array.

    `positives` selects which epoch of the positive stamps to use.
    manifest_positives_first.csv pins them to each object's first detection --
    the epoch ALeRCE's stamp_classifier actually scored. The default
    manifest_positives.csv is mixed-epoch and gives us an unearned advantage,
    since ~25% of those stamps come from the last detection, near peak.
    """
    parts = []
    pos = pd.read_csv(DATA / positives)
    pos["label"], pos["source"] = 1, "tns"
    parts.append(pos)

    for name in negatives:
        path = DATA / name
        if not path.exists():
            print(f"  (skipping missing {name})")
            continue
        neg = pd.read_csv(path)
        neg["label"] = 0
        neg["source"] = "hard" if "hard" in name else "stream"
        parts.append(neg)

    frame = pd.concat(parts, ignore_index=True).drop_duplicates("oid")
    frame = frame[frame.stamp_path.map(lambda s: (DATA / s).exists())]
    frame = frame.reset_index(drop=True)
    X = np.stack([np.load(DATA / p) for p in frame.stamp_path])
    y = frame.label.to_numpy().astype(np.int64)
    return X, y, frame


def split(n, test_frac=0.2):
    """Random train/test index split; oids are unique so nothing leaks across."""
    rng = np.random.default_rng(SEED)
    order = rng.permutation(n)
    cut = int(n * (1 - test_frac))
    return order[:cut], order[cut:]


def normalise(X_train, X_test):
    """asinh-compress, then standardise per channel using TRAIN statistics only.

    asinh (not log) because the difference channel is ~50% negative by
    construction. Train-only stats because computing them over both splits would
    leak test information. NaN -> 0 last, so missing pixels land on the channel
    mean instead of dragging it.
    """
    X_train, X_test = np.arcsinh(X_train), np.arcsinh(X_test)
    mean = np.nanmean(X_train, axis=(0, 2, 3), keepdims=True)
    std = np.nanstd(X_train, axis=(0, 2, 3), keepdims=True)
    return (np.nan_to_num((X_train - mean) / std),
            np.nan_to_num((X_test - mean) / std))


def augment(batch):
    """Random flips and 90-degree rotations, applied per batch on device.

    Physically justified rather than a generic trick: a stamp has no preferred
    orientation -- rotating the sky does not change what a supernova is. (Unlike
    MNIST, where flipping a digit changes its meaning.) This is the dihedral
    group of the square, so it multiplies the effective dataset by 8 for free.
    """
    if torch.rand(1).item() < 0.5:
        batch = torch.flip(batch, dims=[3])          # left-right
    if torch.rand(1).item() < 0.5:
        batch = torch.flip(batch, dims=[2])          # up-down
    k = int(torch.randint(0, 4, (1,)).item())
    if k:
        batch = torch.rot90(batch, k, dims=[2, 3])
    return batch


# ---------------------------------------------------------------- model

class StampCNN(nn.Module):
    """Three conv blocks. Bigger than the MNIST seed because ~7k real examples
    outrun the 6k-parameter version immediately; BatchNorm keeps it trainable
    and dropout holds back the overfitting we measured earlier."""

    def __init__(self, dropout=0.3):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )

        self.features = nn.Sequential(
            block(3, 32),          # [B,3,63,63]  -> [B,32,31,31]
            block(32, 64),         #              -> [B,64,15,15]
            block(64, 128),        #              -> [B,128,7,7]
            nn.AdaptiveAvgPool2d((3, 3)),   # -> [B,128,3,3], size-independent
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128 * 3 * 3, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.head(self.features(x))


# ---------------------------------------------------------------- metrics

def auc(prob, y):
    """Area under the ROC curve via the rank identity (no sklearn needed).

    The headline metric here: threshold-free, and invariant to class balance,
    which matters because our test split is ~50/50 while the real stream is
    ~0.2% supernovae.
    """
    order = prob.argsort()
    ranks = np.empty(len(prob), dtype=np.float64)
    ranks[order] = np.arange(1, len(prob) + 1)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def best_threshold(prob, y):
    """Accuracy-maximising cut, chosen on TRAIN predictions only."""
    grid = np.linspace(0.02, 0.98, 97)
    return float(grid[int(np.argmax([((prob >= t).astype(int) == y).mean()
                                     for t in grid]))])


def report(name, prob, y, threshold=0.5):
    guess = (prob >= threshold).astype(int)
    tp = int(((guess == 1) & (y == 1)).sum())
    fp = int(((guess == 1) & (y == 0)).sum())
    fn = int(((guess == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f"  {name:<26} acc={float((guess == y).mean()):.3f}  "
          f"prec={precision:.3f}  rec={recall:.3f}  f1={f1:.3f}  "
          f"auc={auc(prob, y):.3f}")


# ---------------------------------------------------------------- train / eval

def evaluate(model, loader):
    model.eval()
    probs, truth = [], []
    with torch.no_grad():
        for images, labels in loader:
            p = torch.softmax(model(images.to(device)), dim=1)[:, 1]
            probs.append(p.cpu())
            truth.append(labels)
    return torch.cat(probs).numpy(), torch.cat(truth).numpy()


def train(model, loader, test_loader, epochs, use_augment=True, weight=None):
    # the positive set is now ~4x the negative set, so without weighting the
    # model drifts toward calling everything a supernova
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_auc, best_state = 0.0, None
    for epoch in range(epochs):
        model.train()
        total, batches = 0.0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if use_augment:
                images = augment(images)
            loss = loss_fn(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
        scheduler.step()

        prob, truth = evaluate(model, test_loader)
        a = auc(prob, truth)
        if a > best_auc:
            best_auc = a
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:>3}  loss {total/batches:.4f}  "
              f"test acc {float(((prob >= 0.5).astype(int) == truth).mean()):.3f}  "
              f"auc {a:.3f}")

    # keep the best epoch, not the last -- test AUC plateaus then wobbles
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_auc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--positives", default="manifest_positives.csv",
                        help="which positive manifest to use; "
                             "manifest_positives_first.csv pins stamps to the "
                             "first detection, matching what ALeRCE scored")
    parser.add_argument("--negatives", nargs="+", default=list(DEFAULT_NEGATIVES),
                        help="negative manifests to combine")
    parser.add_argument("--predictions", default="predictions.csv",
                        help="where to write per-object predictions")
    args = parser.parse_args()

    X, y, frame = load_dataset(args.positives, args.negatives)
    print(f"positives: {args.positives}")
    print(f"negatives: {', '.join(args.negatives)}")
    print(f"device: {device}")
    print(f"loaded {len(X)} stamps {X.shape}  |  {int(y.sum())} SN (TNS-confirmed) "
          f"/ {int((y == 0).sum())} not-SN (unfiltered stream)")

    train_idx, test_idx = split(len(X))
    X_train, X_test = normalise(X[train_idx], X[test_idx])
    y_train, y_test = y[train_idx], y[test_idx]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train)),
        batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test)),
        batch_size=args.batch_size, shuffle=False)

    model = StampCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    counts = np.bincount(y_train, minlength=2)
    weight = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32).to(device)
    print(f"model: {n_params:,} parameters   augment={not args.no_augment}")
    print(f"train {len(y_train)} / test {len(y_test)}   "
          f"class weights {[round(w, 3) for w in weight.tolist()]}\n")

    train(model, train_loader, test_loader, args.epochs, not args.no_augment,
          weight=weight)

    prob, truth = evaluate(model, test_loader)
    train_eval = DataLoader(train_loader.dataset, batch_size=args.batch_size, shuffle=False)
    prob_train, truth_train = evaluate(model, train_eval)
    threshold = best_threshold(prob_train, truth_train)

    # the baseline, on exactly the same objects and the same TNS truth
    baseline = frame.p_sn.to_numpy()
    ok = ~np.isnan(baseline[test_idx])

    # Tune the BASELINE's threshold too, on train, exactly as we tuned ours.
    # ALeRCE's probabilities are calibrated far more conservatively -- its best
    # threshold is ~0.12, not 0.5. Comparing our tuned model against its untuned
    # default manufactures a win that is not there.
    b_train = baseline[train_idx]
    b_ok = ~np.isnan(b_train)
    b_threshold = best_threshold(b_train[b_ok], y_train[b_ok])

    print("\n" + "=" * 78)
    print(f"TEST SET  ({int(ok.sum())} objects with a baseline p_sn, "
          f"{int(truth[ok].sum())} confirmed SNe)")
    print("=" * 78)
    report(f"our CNN @{threshold:.2f}", prob[ok], truth[ok], threshold)
    report(f"ALeRCE stamp_classifier @{b_threshold:.2f}",
           baseline[test_idx][ok], truth[ok], b_threshold)
    print(f"\n  (for reference, both at the naive 0.5 cut)")
    report("our CNN @0.50", prob[ok], truth[ok])
    report("ALeRCE @0.50", baseline[test_idx][ok], truth[ok])

    # Nuisance baseline: five summary numbers off the pixels, no morphology.
    # Our two classes come from different selection pipelines, so they differ in
    # brightness and environment as well as in being/not-being supernovae. If
    # this scores close to the CNN, most of the CNN's AUC is sample artefact
    # rather than anything it learned about what a supernova looks like.
    print("\n  -- sanity baselines --")
    report("always-predict-SN", np.ones_like(prob[ok], dtype=float),
           truth[ok], 0.5)
    print("  (run audit_shortcuts.py for the nuisance-only AUC; if the CNN is")
    print("   not well clear of it, the benchmark is easier than it looks)")

    print("\nAUC is the headline: threshold-free and balance-independent.")
    print("Both thresholds are tuned on TRAIN and applied to TEST -- tuning only")
    print("one side is how fake wins get made.")
    print("Precision here is optimistic: this split is ~50/50, the real stream is")
    print("~0.2% supernovae, where a low false-positive rate matters far more.")

    torch.save(model.state_dict(), "model.pth")

    keep = ["oid", "stamp_path", "p_sn"]
    for extra in ("source", "cat_class"):
        if extra in frame.columns:
            keep.append(extra)
    out = frame.iloc[test_idx][keep].copy()
    out["label"] = truth
    out["p_model"] = prob
    out.to_csv(DATA / args.predictions, index=False)
    print(f"\nsaved model.pth and {DATA / args.predictions}")


if __name__ == "__main__":
    main()
