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
Backfill a few nights (downloads dominate; run it overnight):
  nohup .venv/bin/python -u backfill_nights.py --nights 8 > logs/backfill.log 2>&1 &
  tail -f logs/backfill.log

Then attach ground truth and build the dashboard:
  .venv/bin/python label_from_tns.py --full
  .venv/bin/python export_dashboard.py

For the ongoing nightly loop, add to crontab -e (ZTF publishes ~07:10 UTC):
  30 8 * * * cd ~/ztf && .venv/bin/python -u ingest_night.py --date $(date -u -d yesterday +\%Y\%m\%d) >> logs/ingest.log 2>&1
  0  9 * * * cd ~/ztf && .venv/bin/python -u label_from_tns.py --delta $(date -u -d yesterday +\%Y\%m\%d) >> logs/label.log 2>&1
  30 9 * * * cd ~/ztf && .venv/bin/python -u export_dashboard.py >> logs/dash.log 2>&1
EOF
