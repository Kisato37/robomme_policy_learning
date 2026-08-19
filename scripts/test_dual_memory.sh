#!/usr/bin/env bash
set -euo pipefail

python_bin=${PYTHON_BIN:-}
if [[ -z "$python_bin" && -x .venv/bin/python ]]; then
  python_bin=.venv/bin/python
elif [[ -z "$python_bin" ]]; then
  python_bin=python
fi

"$python_bin" -m pytest tests/dual_memory -q
