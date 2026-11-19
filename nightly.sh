#!/usr/bin/env bash
# The whole loop, once a night, from cron: ingest last night, attach any TNS
# labels, rebuild the dashboard. Install with install_cron.sh.
#
# Not a three-line crontab because cron does not source ~/.profile (so .env),
# TNS keeps deltas only two weeks (so the LOOKBACK_DAYS replay), a 16 GB night
# can still be downloading when the next run fires (so flock), and a failed
# ingest must not stop labelling alerts from three weeks ago.

set -uo pipefail          # deliberately not -e: each stage reports and continues

cd "$(dirname "$(readlink -f "$0")")"
REPO="$PWD"
PY="$REPO/.venv/bin/python"
# the server venv is bin/, a Windows dev checkout is Scripts/
[ -x "$PY" ] || PY="$REPO/.venv/Scripts/python.exe"
LOG_DIR="$REPO/logs"
LOOKBACK_DAYS="${LOOKBACK_DAYS:-7}"   # ingest and label catch-up window
KEEP_LOGS_DAYS="${KEEP_LOGS_DAYS:-60}"

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/nightly-$(date -u +%Y%m%d).log"
EVENTS="$LOG_DIR/events.tsv"
exec >>"$LOG" 2>&1

# event stream for status.py: event LEVEL STAGE MESSAGE, LEVEL in OK|WARN|ERROR.
# Tab-separated, not JSON, so appending needs no quoting or subprocess.
event() {
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" \
    "$(printf '%s' "$3" | tr '\t\n' '  ')" >>"$EVENTS"
}

echo "================================================================"
echo "nightly run $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
event OK run "started"
echo "================================================================"

# --- only one instance at a time ---------------------------------------
# fail OPEN: treating a missing flock as "already locked" would make the job
# silently never run while the logs looked normal, which is worse than overlap
if command -v flock >/dev/null 2>&1; then
  exec 9>"$REPO/.nightly.lock"
  if ! flock -n 9; then
    echo "another nightly run is still going; exiting"
    event WARN run "skipped, previous run still going"
    exit 0
  fi
else
  echo "note: flock not available, running without a lock"
fi

# --- credentials -------------------------------------------------------
if [ -f "$REPO/.env" ]; then
  set -a; . "$REPO/.env"; set +a
fi
for v in TNS_BOT_ID TNS_BOT_NAME TNS_API_KEY; do
  if [ -z "${!v:-}" ]; then
    echo "WARNING: $v not set (copy .env.example to .env) -- "
    echo "         ingest will still run, labelling will be skipped"
    event WARN config "$v not set; labelling skipped"
    NO_TNS=1
  fi
done

status=0

# --- 1. ingest any night we are missing --------------------------------
# ask the database which nights it has rather than assuming last night
# succeeded, so a gap after an outage backfills itself
missing=$("$PY" - <<EOF
import sqlite3, datetime, pathlib
db = pathlib.Path("data/pipeline.db")
have = set()
if db.exists():
    c = sqlite3.connect(db)
    have = {r[0] for r in c.execute("SELECT night FROM runs")}
    c.close()
today = datetime.datetime.now(datetime.timezone.utc).date()
# skip today: ZTF publishes the previous night around 07:10 UTC
for back in range(1, ${LOOKBACK_DAYS} + 1):
    night = (today - datetime.timedelta(days=back)).strftime("%Y%m%d")
    if night not in have:
        print(night)
EOF
)

if [ -z "$missing" ]; then
  echo "-- ingest: nothing missing in the last ${LOOKBACK_DAYS} days"
else
  for night in $missing; do
    echo "-- ingest $night"
    if "$PY" -u ingest_night.py --date "$night"; then
      echo "   ok"
      event OK ingest "$night ingested"
    else
      # an empty night (weather) is a 74-byte placeholder and perfectly normal
      echo "   ingest failed or night was empty; continuing"
      event WARN ingest "$night failed or empty (weather nights are normal)"
    fi
  done
fi

# --- 2. attach TNS labels ----------------------------------------------
if [ "${NO_TNS:-0}" = 1 ]; then
  echo "-- labels: skipped, no credentials"
else
  for back in $(seq 1 "$LOOKBACK_DAYS"); do
    day=$(date -u -d "$back days ago" +%Y%m%d)
    echo "-- labels from TNS delta $day"
    if "$PY" -u label_from_tns.py --delta "$day"; then
      event OK labels "delta $day applied"
    else
      # a delta not yet published, or aged out of TNS's two-week window, is
      # expected rather than broken: the replay covers it
      echo "   delta $day unavailable; continuing"
      event WARN labels "delta $day unavailable"
    fi
  done
fi

# --- 2b. confirm negatives against external catalogues ------------------
# only the control objects the dashboard evaluates, and only ones without a
# verdict yet, so it stays small after the first run
echo "-- confirming negatives against SIMBAD / Milliquas / VSX"
if "$PY" -u confirm_negatives.py --limit 3000; then
  event OK confirm "catalogue crossmatch done"
else
  echo "   crossmatch failed"
  event WARN confirm "crossmatch failed (controls stay presumed-negative)"
fi

# --- 3. rebuild the static dashboard -----------------------------------
echo "-- dashboard"
if "$PY" -u export_dashboard.py; then
  echo "   ok"
  event OK dashboard "rebuilt"
else
  echo "   dashboard export FAILED"
  event ERROR dashboard "export failed"
  status=1
fi

# --- 3b. publish it, if a target repo is configured ---------------------
# only runs once publish_dashboard.sh --setup has been done. A publish failure
# is a warning: the pipeline is fine and the next run pushes the same content.
if [ -d "$REPO/.publish" ] || [ -n "${PUBLISH_REPO:-}" ]; then
  echo "-- publish"
  if bash publish_dashboard.sh; then
    event OK publish "pushed"
  else
    echo "   publish failed"
    event WARN publish "push failed (dashboard is still generated locally)"
  fi
fi

# --- 4. tell the human when a retrain is worth doing --------------------
# retraining is a laptop job: the server has no usable GPU
"$PY" - <<'EOF'
import sqlite3, pathlib
db = pathlib.Path("data/pipeline.db")
if db.exists():
    c = sqlite3.connect(db)
    n = c.execute("SELECT COUNT(DISTINCT oid) FROM alerts "
                  "WHERE tns_label IS NOT NULL").fetchone()[0]
    c.close()
    print(f"-- labelled objects in total: {n}")
    if n >= 200:
        print("   enough new truth to be worth a retrain on the laptop:")
        print("     git pull && python train.py --epochs 40 ... && git push")
EOF

find "$LOG_DIR" -name 'nightly-*.log' -mtime +"$KEEP_LOGS_DAYS" -delete 2>/dev/null

if [ "$status" = 0 ]; then event OK run "finished cleanly"; else event ERROR run "finished with status $status"; fi
echo "finished $(date -u +'%H:%M:%S UTC') with status $status"
exit $status
