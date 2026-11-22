#!/usr/bin/env bash
set -euo pipefail

# Requires GitHub CLI: https://cli.github.com/ and `gh auth login`.
REPO="${1:-manav4u/LAWLESS}"
VISIBILITY="${2:-public}"

if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI (gh) is required. Install it and run: gh auth login" >&2
  exit 1
fi

gh repo create "$REPO" --"$VISIBILITY" --source=. --remote=origin --push
