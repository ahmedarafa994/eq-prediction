#!/bin/bash
# AUDIT: LOW-12 — Hardened with set -euo pipefail and error handling
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/insight" || { echo "ERROR: cd insight failed"; exit 1; }

if [ ! -f train.py ]; then
    echo "ERROR: train.py not found in $(pwd)"
    exit 1
fi

LOG_FILE="training_$(date +%Y%m%d_%H%M%S).log"
echo "Starting training — log: ${LOG_FILE}"
PYTHONUNBUFFERED=1 python train.py 2>&1 | tee "${LOG_FILE}"
