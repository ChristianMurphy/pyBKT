#!/bin/bash
# Merge each fork branch onto upstream CAHLR/pyBKT master in memory (git merge-tree, nothing is
# committed), then run that branch's test suite three ways: compiled + NumPy 2, pure Python + NumPy 2,
# pure Python + NumPy 1. Used for PLANNING.md section 5 (2026-10-09, upstream master cc1682e).
#
#   git remote add cahlr https://github.com/CAHLR/pyBKT && git fetch cahlr master
#   PY2=/path/to/python-with-numpy2 PY1=/path/to/python-with-numpy1 ./upstream_sync_check.sh
# Both interpreters need numpy, pandas, pytest; PY2 also needs setuptools and a C++ compiler.
set -u
PY2=${PY2:-python3}; PY1=${PY1:-python3}
cd "$(git rev-parse --show-toplevel)"
for ref in origin/test/regression-harness origin/claude/pybkt-performance-research-xqv9jv; do
  T=$(git merge-tree --write-tree cahlr/master "$ref") || { echo "### $ref: merge conflict"; continue; }
  D=$(mktemp -d)
  git archive "$T" | tar -x -C "$D"
  echo "### $ref merged onto cahlr/master (tree $T)"
  ( cd "$D" && "$PY2" -m pip install -q --no-build-isolation --no-deps --target "$D/site" . >"$D/build.log" 2>&1
    PYTHONPATH="$D/site" "$PY2" -c "import pyBKT.fit.E_step" && echo "compiled extension imports"
    echo "--- compiled, NumPy 2";    PYTHONPATH="$D/site"  "$PY2" -m pytest -q -rxXf tests -p no:cacheprovider 2>&1 | tail -25
    echo "--- pure Python, NumPy 2"; PYTHONPATH=source-py "$PY2" -m pytest -q -rxXf tests -p no:cacheprovider 2>&1 | tail -25
    echo "--- pure Python, NumPy 1"; PYTHONPATH=source-py "$PY1" -m pytest -q -rxXf tests -p no:cacheprovider 2>&1 | tail -25 )
done
