#!/usr/bin/env bash
#
# Helper script to create and push a new GitHub repository using the
# GitHub CLI (`gh`).  This script expects that you have `gh` installed
# and authenticated on your system.  Usage:
#
#   ./push.sh [repo-name] [public|private]
#
# Example:
#   ./push.sh surf-forecast-ai public
#
set -euo pipefail

REPO_NAME=${1:-surf-forecast-ai}
VISIBILITY=${2:-private}

if ! command -v gh &> /dev/null; then
  echo "Error: GitHub CLI (gh) is not installed." >&2
  exit 1
fi

# Create a new repository from the current directory and push
gh repo create "$REPO_NAME" --source . --remote origin --${VISIBILITY} --push