#!/usr/bin/env bash
# The whole loop, once a night, from cron. Nothing here needs a human.
#
#   ingest last night  ->  attach any TNS labels  ->  rebuild the dashboard
#
# Install with:  bash install_cron.sh
# Run by hand :  bash nightly.sh
#
# Four things this handles that a naive three-line crontab does not:
#
#  1. cron does NOT source ~/.profile, so TNS_API_KEY would be unset and every
#     label run would fail silently. Credentials come from .env instead.
#  2. TNS keeps delta files for only two weeks. A single failed night would lose
#     those labels forever, so we replay the last LOOKBACK_DAYS every time.
#     apply_labels() skips rows whose type has not changed, so replaying is
#     cheap and self-healing.
#  3. A 16 GB night can still be downloading when the next run fires. flock
#     makes a second instance exit rather than fight over the database.
#  4. A failed ingest must not stop labelling and export -- yesterday's alerts
#     are not a prerequisite for labelling alerts from three weeks ago.

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

# Structured event stream for status.py. Tab-separated rather than JSON so the
# shell can append a line without spawning anything or escaping quotes; the
# message has tabs and newlines stripped so one event is always one line.
#   event LEVEL STAGE MESSAGE       LEVEL in OK|WARN|ERROR
event() {
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" \
    "$(printf '%s' "$3" | tr '\t\n' '  ')" >>"$EVENTS"
}

echo "================================================================"
echo "nightly run $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
event OK run "started"
echo "================================================================"

# --- only one instance at a time ---------------------------------------
# Fail OPEN, not closed. Treating a missing flock as "already locked" would make
# the job silently never run while the logs looked entirely normal -- a far
# worse outcome than the rare overlap locking is meant to prevent.
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
# Ask the database which nights it already has rather than assuming last night
# succeeded. This backfills a gap after an outage without any bookkeeping.
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
      # An empty night (weather) is published as a 74-byte placeholder and is a
      # perfectly normal outcome, so this is a warning rather than a failure.
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
      # A delta that is not published yet, or has aged out of TNS's two-week
      # window, is expected rather than broken -- the replay covers it.
      echo "   delta $day unavailable; continuing"
      event WARN labels "delta $day unavailable"
    fi
  done
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
# Optional: only runs once publish_dashboard.sh --setup has been done. A
# publish failure is a WARNING, not an error -- the pipeline itself is fine and
# the next run will push the same content again.
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
# Retraining is a laptop job (the server has no usable GPU) and is only worth
# doing when enough new truth has arrived to move the model.
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
