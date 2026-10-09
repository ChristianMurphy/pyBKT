#!/bin/sh
# Builds two wheels of bkt_rs (portable default x86-64, and -C target-cpu=native)
# and unpacks each into inst/<flavor>/ so benchmarks can select one via PYTHONPATH.
set -e
cd "$(dirname "$0")"
export PATH=/root/.cargo/bin:/root/.local/bin:$PATH
for flavor in default native; do
  rm -rf wheels/$flavor inst/$flavor
  if [ $flavor = native ]; then export RUSTFLAGS="-C target-cpu=native"; else unset RUSTFLAGS; fi
  (cd bkt_rs && CARGO_TARGET_DIR=../target/$flavor ../venv/bin/maturin build --release -i ../venv/bin/python -o ../wheels/$flavor -q)
  mkdir -p inst/$flavor && ./venv/bin/python -m zipfile -e wheels/$flavor/*.whl inst/$flavor
done
ls -l inst/*/bkt_rs/*.so
