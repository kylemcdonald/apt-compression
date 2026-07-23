#!/bin/bash
set -u

cd "$(dirname "$0")/.."
D=/Users/kyle/Documents/GitHub/uap/uap-materials-article/src/data
PY=.venv/bin/python
mkdir -p experiments/artifacts

run() {
  local f="$1"
  local slug
  slug=$(basename "$f" | tr '[:upper:]' '[:lower:]' | sed -E 's/\.pos$//; s/[^a-z0-9]+/-/g; s/^-+|-+$//g')
  echo "START $slug $(date +%H:%M:%S)"
  "$PY" experiments/run_bench.py --pos "$f" --methods all \
    > "experiments/artifacts/log_$slug.log" 2>&1
  local status=$?
  echo "DONE $slug $(date +%H:%M:%S) exit=$status"
  return "$status"
}

export PY
export -f run

# Discover the directory rather than hard-coding filenames, so a newly added
# eighth dataset is picked up automatically. Null delimiters preserve spaces.
find "$D" -maxdepth 1 -type f \( -iname '*.pos' \) -print0 |
  xargs -0 -n 1 -P 4 bash -c 'run "$1"' _
status=$?
echo "ALL DONE $(date +%H:%M:%S) exit=$status"
exit "$status"
