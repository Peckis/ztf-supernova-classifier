#!/usr/bin/env bash
# Run the full data fetch on the Ubuntu server, unattended. The server is no
# faster per request (~2.3s either way); what it buys is being able to close
# the laptop on an ~8h job.
#
#   scp fetch_on_server.sh fetch_data.py fetch_tns.py user@server:~/stamps/
#   ssh user@server 'cd ~/stamps && ./fetch_on_server.sh'
#
# Export TNS_BOT_ID, TNS_BOT_NAME, TNS_API_KEY first (for the --all-ztf step).

set -euo pipefail

N_POSITIVES="${N_POSITIVES:-14574}"   # every confirmed SN available
N_NEGATIVES="${N_NEGATIVES:-12000}"   # stream sample; ~99.8% genuine negatives
N_HARD="${N_HARD:-6000}"              # long-span objects: AGN / variable stars
WORKERS="${WORKERS:-3}"
RATE="${RATE:-2}"                     # requests/sec across ALL workers.
# Do not raise RATE without evidence from a LONG run. 8 unthrottled workers
# produced 10,453 APIErrors and got the query endpoint refusing us; a 3-hour
# sequential run at ~0.9/s never hit a throttle. A burst test cannot detect a
# sustained-rate limit: it looks perfectly linear right up until it isn't.
MJD_START="${MJD_START:-58200}"
MJD_END="${MJD_END:-61000}"

cd "$(dirname "$0")"

# this script only orchestrates -- the real work is in the python modules
for f in fetch_data.py fetch_tns.py; do
  [ -f "$f" ] || { echo "ERROR: $f missing. Copy it here alongside this script."; exit 1; }
done

for v in TNS_BOT_ID TNS_BOT_NAME TNS_API_KEY; do
  if [ -z "${!v:-}" ] && [ ! -f data/tns_ztf_all.csv ]; then
    echo "ERROR: \$$v is not set, and data/tns_ztf_all.csv does not exist yet."
    echo "Either export the TNS credentials, or copy tns_ztf_all.csv over."
    exit 1
  fi
done

if ! command -v uv >/dev/null 2>&1; then
  echo "installing uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d .venv ]; then
  echo "creating venv ..."
  uv venv --python 3.11
fi
uv pip install alerce astropy numpy pandas requests tqdm

PY=.venv/bin/python
mkdir -p data logs

# 1. TNS catalogue -> the pool of confirmed supernovae (112 MB, so cached)
if [ ! -f data/tns_ztf_all.csv ]; then
  echo "=== TNS catalogue ==="
  $PY fetch_tns.py --all-ztf 2>&1 | tee logs/tns.log
else
  echo "=== TNS catalogue already present, skipping ==="
fi

# 2. Positives: spectroscopically confirmed SNe. Real ground truth.
echo "=== positives (${N_POSITIVES}) ==="
$PY -u fetch_data.py --oids-from data/tns_ztf_all.csv --only-label 1 \
    --n "$N_POSITIVES" --name positives --workers "$WORKERS" --rate "$RATE" \
    2>&1 | tee logs/positives.log

# 3. Negatives: unfiltered stream sample, spread over years so the model
#    cannot learn detector-epoch quirks instead of what a transient looks like
echo "=== negatives (${N_NEGATIVES}) ==="
$PY -u fetch_data.py --unfiltered --n "$N_NEGATIVES" \
    --mjd "$MJD_START" "$MJD_END" --workers "$WORKERS" --rate "$RATE" \
    2>&1 | tee logs/negatives.log

# 4. Hard negatives: objects detected for over a year, so physically not
#    supernovae. The verdict comes from the detection history rather than a
#    classifier, so it stays non-circular, and these are the cases the model
#    actually confuses -- the stream sample above is 80% easy junk.
echo "=== hard negatives (${N_HARD}) ==="
$PY -u fetch_data.py --hard-negatives --n "$N_HARD" --workers "$WORKERS" --rate "$RATE" \
    2>&1 | tee logs/hard_negatives.log

echo
echo "=== done ==="
du -sh data/stamps
echo "stamps: $(find data/stamps -name '*.npy' | wc -l)"
echo
echo "Transfer back to the laptop with:"
echo "  tar czf stamps.tar.gz data/"
echo "  # then, from the laptop:"
echo "  scp user@server:~/stamps/stamps.tar.gz ."
