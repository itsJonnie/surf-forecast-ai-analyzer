#!/usr/bin/env bash
#
# Manual helper script to push the current directory to an existing remote
# repository.  Provide the remote URL (SSH or HTTPS) as the first argument.
# Usage:
#   ./push_manual.sh git@github.com:yourname/surf-forecast-ai.git

set -euo pipefail

REMOTE_URL=${1:-}

if [[ -z "$REMOTE_URL" ]]; then
  echo "Usage: $0 <remote-url>"
  exit 1
fi

git init
git add .
git commit -m "Initial commit: Surf Forecast AI Analyzer"
git branch -M main
git remote add origin "$REMOTE_URL"
git push -u origin main