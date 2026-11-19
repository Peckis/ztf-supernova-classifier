#!/usr/bin/env bash
# Push the generated dashboard to a public repo that GitHub Pages serves.
#
#   bash publish_dashboard.sh            # publish if anything changed
#   bash publish_dashboard.sh --setup    # one-time instructions
#
# A separate repo so the code repo can stay private, and so ~500 KB of nightly
# binary churn (~180 MB a year) stays out of a history that is otherwise text.
#
# Kept at exactly ONE commit, amended and force-pushed, so the published repo
# never grows. Nothing is lost: accuracy.json carries the full time series.

set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
REPO="$PWD"
DASH="$REPO/dashboard"
PUBLISH_REPO="${PUBLISH_REPO:-git@github.com:Peckis/ztf-dashboard.git}"
WORK="$REPO/.publish"

if [ "${1:-}" = "--setup" ]; then
  cat <<'EOF'
One-time setup
==============

1. Create a PUBLIC repo on GitHub called  ztf-dashboard  (empty, no README).

2. Give the server push access with a deploy key scoped to just that repo --
   safer than a personal token, which would grant access to everything:

     ssh-keygen -t ed25519 -f ~/.ssh/ztf_dashboard -N ""
     cat ~/.ssh/ztf_dashboard.pub

   Paste that into: the ztf-dashboard repo -> Settings -> Deploy keys
   -> Add deploy key -> TICK "Allow write access".

3. Tell ssh to use it for github:

     cat >> ~/.ssh/config <<'CFG'
     Host github.com
       IdentityFile ~/.ssh/ztf_dashboard
       IdentitiesOnly yes
     CFG

4. Turn on Pages: ztf-dashboard -> Settings -> Pages
   -> Source: "Deploy from a branch", Branch: main, folder: / (root)

5. Publish once, then the nightly job keeps it current:

     bash publish_dashboard.sh

Your live URL will be:  https://peckis.github.io/ztf-dashboard/
EOF
  exit 0
fi

[ -d "$DASH" ] || { echo "no dashboard/ yet -- run export_dashboard.py first"; exit 1; }

if [ ! -d "$WORK/.git" ]; then
  echo "cloning $PUBLISH_REPO"
  rm -rf "$WORK"
  if ! git clone --depth 1 "$PUBLISH_REPO" "$WORK" 2>/dev/null; then
    # empty repo has no branch to clone
    mkdir -p "$WORK" && git -C "$WORK" init -q -b main
    git -C "$WORK" remote add origin "$PUBLISH_REPO"
  fi
fi

cp -f "$DASH"/* "$WORK"/ 2>/dev/null || true
# .nojekyll stops GitHub Pages ignoring files that begin with an underscore
touch "$WORK/.nojekyll"

cd "$WORK"
git add -A
if git diff --cached --quiet 2>/dev/null && [ -n "$(git log -1 2>/dev/null)" ]; then
  echo "dashboard unchanged; nothing to publish"
  exit 0
fi

STAMP=$(date -u +'%Y-%m-%d %H:%M UTC')
if git log -1 >/dev/null 2>&1; then
  git commit -q --amend -m "dashboard $STAMP"   # keep history at one commit
else
  git commit -q -m "dashboard $STAMP"
fi

if git push -q --force origin main 2>/dev/null; then
  echo "published at $STAMP"
else
  echo "push failed -- is the deploy key installed with write access?"
  echo "  run: bash publish_dashboard.sh --setup"
  exit 1
fi
