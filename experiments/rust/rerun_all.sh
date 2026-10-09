#!/bin/sh
# Rebuild everything and rerun all checks + benchmarks. Run from anywhere.
set -e
cd "$(dirname "$0")"
./build.sh                                   # -> inst/default (portable) and inst/native
# archived v1 (pre-forbid(unsafe_code)) build, only used as the baseline for the cost-of-safety column
export PATH=/root/.cargo/bin:/root/.local/bin:$PATH
rm -rf wheels/v1 inst/v1_unsafe_default
(cd archive/bkt_rs_v1_unsafe && CARGO_TARGET_DIR=../../target/v1 ../../venv/bin/maturin build --release -i ../../venv/bin/python -o ../../wheels/v1 -q)
mkdir -p inst/v1_unsafe_default && ./venv/bin/python -m zipfile -e wheels/v1/*.whl inst/v1_unsafe_default
cd py
export PYTHONPATH=../inst/default
../venv/bin/python correctness.py   > ../results/correctness.txt
../venv/bin/python generic_test.py  | tee ../results/generic_test.txt
../venv/bin/python determinism.py   | tee ../results/determinism.txt
../venv/bin/python ll_accuracy.py   | tee ../results/ll_accuracy.txt
../venv/bin/python em_compat_check.py | tee ../results/em_compat.txt
for isa in sse2 sse4_2 avx2 auto; do BKT_RS_ISA=$isa ../venv/bin/python lanes_check.py; done > ../results/lanes_check.txt
unset PYTHONPATH
rm -f ../results/bench_v2.jsonl
../venv/bin/python bench_all.py ../results/bench_v2.jsonl 3 > ../results/bench_v2.log   # ~25 min
../venv/bin/python summarize.py ../results/bench_v2.jsonl > ../results/bench_v2.md
