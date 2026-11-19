#!/usr/bin/env bash
# Install (or refresh) the nightly cron entry. Safe to run repeatedly.
#
#   bash install_cron.sh          # install
#   bash install_cron.sh --remove # uninstall
#
# One entry, not three: nightly.sh sequences the stages itself, so only one
# thing can hold the database at a time.

set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
REPO="$PWD"
MARKER="# ztf-supernova-classifier nightly"
# 08:30 UTC: ZTF publishes the previous night around 07:10 UTC.
#
# Deliberately NOT wrapped in flock -- nightly.sh takes the lock itself, and
# flock() treats separate file descriptors independently even inside one
# process, so an outer flock would make the script's own attempt fail: it
# would report a run already going, exit 0, and never run, every single day.
ENTRY="30 8 * * * cd $REPO && bash nightly.sh $MARKER"

current=$(crontab -l 2>/dev/null || true)
cleaned=$(printf '%s\n' "$current" | grep -vF "$MARKER" || true)

if [ "${1:-}" = "--remove" ]; then
  printf '%s\n' "$cleaned" | crontab -
  echo "removed. remaining entries:"; crontab -l 2>/dev/null || echo "  (none)"
  exit 0
fi

for f in nightly.sh ingest_night.py label_from_tns.py export_dashboard.py; do
  [ -f "$f" ] || { echo "ERROR: $f missing"; exit 1; }
done
[ -x .venv/bin/python ] || { echo "ERROR: no .venv -- run setup_server.sh first"; exit 1; }

if [ ! -f .env ]; then
  echo "WARNING: no .env file. cron does not read ~/.profile, so TNS labelling"
  echo "         will be skipped until you create one:"
  echo "           cp .env.example .env && \$EDITOR .env"
fi

printf '%s\n%s\n' "$cleaned" "$ENTRY" | grep -v '^$' | crontab -
echo "installed:"
crontab -l | grep -F "$MARKER"
echo
echo "logs:        $REPO/logs/nightly-YYYYMMDD.log"
echo "test it now: bash nightly.sh && tail -40 logs/nightly-\$(date -u +%Y%m%d).log"
