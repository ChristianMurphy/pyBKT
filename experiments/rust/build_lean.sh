#!/bin/sh
# Builds bkt_lean wheels: default (no fearless_simd) and --features simd, unpacked into inst/lean{,_simd}
set -e
cd "$(dirname "$0")"
export PATH=/root/.cargo/bin:/root/.local/bin:$PATH
unset RUSTFLAGS
for flavor in lean lean_simd; do
  rm -rf wheels/$flavor inst/$flavor
  feat=""; [ $flavor = lean_simd ] && feat="--features simd"
  (cd bkt_lean && CARGO_TARGET_DIR=../target/$flavor ../venv/bin/maturin build --release $feat -i ../venv/bin/python -o ../wheels/$flavor -q)
  mkdir -p inst/$flavor && ./venv/bin/python -m zipfile -e wheels/$flavor/*.whl inst/$flavor
done
ls -l inst/lean*/bkt_lean/*.so wheels/lean*/*.whl
