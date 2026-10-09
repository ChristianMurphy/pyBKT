#!/bin/sh
# Download the example datasets the experiments use (from CAHLR/pyBKT-examples).
set -e
DATA=${BKT_DATA:-/tmp/claude-0/data}
mkdir -p "$DATA"
for f in as.csv ct.csv; do
  curl -sSfL -o "$DATA/$f" "https://raw.githubusercontent.com/CAHLR/pyBKT-examples/master/data/$f"
done
ls -l "$DATA"
