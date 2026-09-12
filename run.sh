#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
if [ -x ".venv/Scripts/python.exe" ]; then
  PYTHON_BIN=".venv/Scripts/python.exe"
elif [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
fi

if [ ! -f "data/splits/train.jsonl" ] || [ ! -f "data/splits/val.jsonl" ] || [ ! -f "data/splits/test.jsonl" ]; then
  "$PYTHON_BIN" data/prepare_data.py
fi
"$PYTHON_BIN" agents/structural_agent.py
"$PYTHON_BIN" agents/semantic_agent.py
"$PYTHON_BIN" agents/routine_agent.py
"$PYTHON_BIN" coordinator/fuse.py
"$PYTHON_BIN" scripts/smoke_test_models.py
