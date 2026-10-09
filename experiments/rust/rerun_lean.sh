#!/bin/sh
# bkt_lean: build, clippy, checks, dependency/unsafe/advisory audit, benchmark (~45 min). Needs ./build.sh done for bkt_rs.
set -e
cd "$(dirname "$0")"
export PATH=/root/.cargo/bin:/root/.local/bin:$PATH
./build_lean.sh                                    # -> inst/lean (default features), inst/lean_simd (--features simd)
(cd bkt_rs   && CARGO_TARGET_DIR=../target/clippy     cargo clippy --release --all-targets -- -D warnings)
(cd bkt_lean && CARGO_TARGET_DIR=../target/lean_check cargo clippy --release --all-targets -- -D warnings)
(cd bkt_lean && CARGO_TARGET_DIR=../target/lean_check cargo clippy --release --all-targets --features simd -- -D warnings)
python3 audit/deps_unsafe.py > audit/deps_unsafe.md
python3 audit/rustsec_check.py > audit/rustsec.txt     # advisory-db clone at /tmp/claude-0/exp/depaudit/advisory-db
cd py
PYTHONPATH=../inst/default:../inst/lean      ../venv/bin/python lean_check.py > ../results/lean_check.txt
PYTHONPATH=../inst/default:../inst/lean_simd ../venv/bin/python lean_check.py > ../results/lean_simd_check.txt
PYTHONPATH=../inst/lean ../venv/bin/python fit_check.py > ../results/fit_check.txt
rm -f ../results/bench_lean.jsonl
../venv/bin/python bench_lean_all.py ../results/bench_lean.jsonl 3 > ../results/bench_lean.log
../venv/bin/python summarize_lean.py ../results/bench_lean.jsonl > ../results/bench_lean.md
