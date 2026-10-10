#!/bin/bash
# Driver for autovec.py: every lane implementation x build, L = 4 and 8, 1 and 4 threads, both fixtures.
# Needs inst/lean and inst/lean_simd (build_lean.sh) and inst/lean_native, built from the rust/ directory with:
#   (cd bkt_lean && RUSTFLAGS="-C target-cpu=native" CARGO_TARGET_DIR=../target/lean_native \
#      ../venv/bin/maturin build --release -i ../venv/bin/python -o ../wheels/lean_native -q)
#   mkdir -p inst/lean_native && ./venv/bin/python -m zipfile -e wheels/lean_native/*.whl inst/lean_native
# Idle machine preferred. BKT_RS_ISA=sse2|sse4_2|avx2 caps the runtime dispatch (results/autovec_isa_cap.txt).
set -u
PY=${PY:-/tmp/claude-0/exp/rust/venv/bin/python}
here=$(cd "$(dirname "$0")" && pwd)
echo "fixture,build,impl,lanes,threads,median_ns,min_ns,max_rel_diff_vs_exact"
for fx in as_all synth5m; do
  for th in 1 4; do
    "$PY" "$here/autovec.py" $fx lean plain 0 $th          # exact scalar kernel, one student at a time
    for L in 4 8; do
      for spec in "lean plain" "lean blend" "lean_native plain" "lean_native blend" \
                  "lean_simd plain_dispatch" "lean_simd blend_dispatch" "lean_simd fearless"; do
        set -- $spec
        "$PY" "$here/autovec.py" $fx $1 $2 $L $th
      done
    done
  done
done
