#!/bin/bash
# Driver for omp_controlled.py. SITE: a directory with a compiled pyBKT (pip install --target).
# PY: a Python with NumPy. Background load: k processes spinning on one core each.
set -u
SITE=${SITE:?set SITE}; PY=${PY:-python3}; LABEL=${LABEL:-build}
here=$(cd "$(dirname "$0")" && pwd)
echo "build,load_procs,answers,students,parallel,median_us,p90_us,min_us"
for k in 0 2 4; do
  pids=()
  for _ in $(seq 1 $k); do "$PY" -c "while True: pass" & pids+=($!); done
  sleep 2
  for size in "5 2" "1800 230" "22900 1115" "200000 5000"; do
    for par in 0 1; do
      echo "$LABEL,$k,$(PYTHONPATH=$SITE "$PY" "$here/omp_controlled.py" $size $par)"
    done
  done
  for p in "${pids[@]}"; do kill $p 2>/dev/null; done; wait 2>/dev/null
done
