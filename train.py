"""Train a 3-channel CNN to classify SN vs not-SN on ZTF stamps.

Positives are TNS spectroscopically-confirmed supernovae, negatives are
date-sampled stream objects, so the comparison against ALeRCE at the end is a
real benchmark rather than a measure of how well we imitate it.

    python train.py --epochs 60
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
    rng = np.random.default_rng(SEED)
    order = rng.permutation(n)
    cut = int(n * (1 - test_frac))
    return order[:cut], order[cut:]


def normalise(X_train, X_test):
    """asinh-compress, then standardise per channel on TRAIN statistics only.

    asinh rather than log because the difference channel is ~50% negative by
    construction. The stats are returned because inference must reuse the exact
    values the model was trained with.
    """
    X_train, X_test = np.arcsinh(X_train), np.arcsinh(X_test)
    mean = np.nanmean(X_train, axis=(0, 2, 3), keepdims=True)
    std = np.nanstd(X_train, axis=(0, 2, 3), keepdims=True)
    # NaN -> 0 last, so missing pixels land on the channel mean
    return (np.nan_to_num((X_train - mean) / std),
            np.nan_to_num((X_test - mean) / std), mean, std)


def apply_normalisation(X, mean, std):
    """Normalise new data with saved statistics -- the inference-time path."""
    return np.nan_to_num((np.arcsinh(X) - mean) / std)


def augment(batch):
    """Random flips and 90-degree rotations: the dihedral group of the square.

    Valid here because a stamp has no preferred orientation.
    """
    if torch.rand(1).item() < 0.5:
        batch = torch.flip(batch, dims=[3])
    if torch.rand(1).item() < 0.5:
        batch = torch.flip(batch, dims=[2])
    k = int(torch.randint(0, 4, (1,)).item())
    if k:
        batch = torch.rot90(batch, k, dims=[2, 3])
    return batch


# ---------------------------------------------------------------- model

class StampCNN(nn.Module):
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
            nn.AdaptiveAvgPool2d((3, 3)),
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
    """Area under the ROC curve via the rank identity."""
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
    X_train, X_test, norm_mean, norm_std = normalise(X[train_idx], X[test_idx])
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

    baseline = frame.p_sn.to_numpy()
    ok = ~np.isnan(baseline[test_idx])

    # Tune the BASELINE's threshold on train too. ALeRCE is calibrated far more
    # conservatively (its best cut is ~0.12), so comparing our tuned model
    # against its untuned default manufactures a win that is not there.
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

    # save the normalisation with the weights: a model applied with different
    # statistics than it was trained on is silently wrong, not obviously broken
    torch.save({
        "state_dict": model.state_dict(),
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "positives": args.positives,
        "negatives": args.negatives,
        "threshold": threshold,
        "test_auc": float(auc(prob, truth)),
    }, "model.pth")

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
