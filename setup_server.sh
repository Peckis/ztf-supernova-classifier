#!/usr/bin/env bash
# One-time setup for the Phase 2 pipeline on the Ubuntu server.
#
#   git clone git@github.com:Peckis/ztf-supernova-classifier.git ~/ztf
#   cd ~/ztf && bash setup_server.sh
#
# Installs CPU-only PyTorch on purpose. Inference measured at ~1,400 alerts/sec
# on 4 threads, so a whole night takes under 5 minutes and the download
# dominates by orders of magnitude. The K620s would not help even if they
# worked: modern PyTorch has dropped kernels for Maxwell (compute 5.0).
#
# Training stays on the laptop GPU. model.pth is tracked in git, so deploying a
# retrain is `git pull`.

set -euo pipefail
cd "$(dirname "$0")"

echo "=== python ==="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
[ -d .venv ] || uv venv --python 3.11

# CPU wheels: the CUDA build is ~2.5 GB and buys nothing here
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install alerce astropy fastavro matplotlib numpy pandas requests tqdm

PY=.venv/bin/python
mkdir -p data logs

echo
echo "=== checks ==="
$PY - <<'EOF'
import shutil, torch, pathlib
print(f"torch {torch.__version__} (cpu-only: {not torch.cuda.is_available()})")
free = shutil.disk_usage("data").free / 1e9
print(f"free disk on data/: {free:.0f} GB")
if free < 40:
    print("  WARNING: a backfill wants ~15 GB free for the largest single night,")
    print("  and more if you keep tarballs. Mount the HDD at ./data if it is tight.")
ck = pathlib.Path("model.pth")
if not ck.exists():
    raise SystemExit("model.pth missing -- did the clone include it? (it is tracked)")
blob = torch.load(ck, map_location="cpu", weights_only=False)
assert "norm_mean" in blob, "checkpoint has no normalisation stats; retrain"
print(f"model.pth ok, test AUC {blob.get('test_auc', float('nan')):.4f}")
EOF

echo
echo "=== TNS credentials ==="
missing=0
for v in TNS_BOT_ID TNS_BOT_NAME TNS_API_KEY; do
  if [ -z "${!v:-}" ]; then echo "  $v NOT set"; missing=1; else echo "  $v ok"; fi
done
if [ "$missing" = 1 ]; then
  echo "  Add them to ~/.profile (needed by label_from_tns.py, not by ingest):"
  echo "    export TNS_BOT_ID=... TNS_BOT_NAME=... TNS_API_KEY=..."
fi

cat <<'EOF'

=== ready ===
1. Credentials (cron does not read ~/.profile, so they go in a file):
     cp .env.example .env && $EDITOR .env

2. Backfill some past nights, so the dashboard has real numbers now instead of
   in several months. Downloads dominate; run it overnight:
     nohup .venv/bin/python -u backfill_nights.py --nights 8 > logs/backfill.log 2>&1 &
     tail -f logs/backfill.log

3. Turn on the nightly loop. One entry; nightly.sh sequences
   ingest -> label -> dashboard itself and recovers its own gaps:
     bash install_cron.sh

   Try it once by hand first:
     bash nightly.sh && tail -40 logs/nightly-$(date -u +%Y%m%d).log

After that nothing needs a human except retraining, which is a laptop job (the
server has no usable GPU). nightly.sh tells you in the log when enough new
labels have accumulated to make one worthwhile.
EOF
