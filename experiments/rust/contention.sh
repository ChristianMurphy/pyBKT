#!/bin/bash
# Driver for contention.py. PY: the Rust experiments venv (C++ pyBKT installed); ROOT: /tmp/claude-0/exp/rust.
set -u
PY=${PY:-/tmp/claude-0/exp/rust/venv/bin/python}; ROOT=${ROOT:-/tmp/claude-0/exp/rust}
here=$(cd "$(dirname "$0")" && pwd)
echo "load_procs,fixture,variant,median_ms,p90_ms,min_ms"
for k in 0 2 4; do
  pids=()
  for _ in $(seq 1 $k); do "$PY" -c "while True: pass" & pids+=($!); done
  sleep 2
  for fx in as_all synth5m; do
    for v in cpp_serial cpp_par4 rayon_t4 lean_t4; do
      echo "$k,$(PYTHONPATH=$ROOT/inst/default:$ROOT/inst/lean "$PY" "$here/contention.py" $fx $v)"
    done
  done
  for p in "${pids[@]}"; do kill $p 2>/dev/null; done; wait 2>/dev/null
done
