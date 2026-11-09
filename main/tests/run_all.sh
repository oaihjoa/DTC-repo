#!/usr/bin/env bash
# Full test suite on synthetic data (CPU, a few minutes; needs pyKT).
# Run from anywhere:  bash tests/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== step 1: property terms vs references =="
python3 tests/test_losses.py
echo "== step 2: end-to-end smoke (synthetic data) =="
python3 tests/test_train_smoke.py
echo "== step 3: preservation-mechanism gates =="
python3 tests/test_preservation.py
echo "== step 4: model adapter gates =="
python3 tests/test_model_adapters.py
echo "ALL TESTS PASSED"
