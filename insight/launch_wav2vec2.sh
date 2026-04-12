#!/bin/bash
# AUDIT: LOW-12 — Hardened with set -euo pipefail, PID file, and error handling
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}" || { echo "ERROR: cd insight failed"; exit 1; }

if [ ! -f train.py ]; then
    echo "ERROR: train.py not found in $(pwd)"
    exit 1
fi

if [ ! -f conf/config_wav2vec2.yaml ]; then
    echo "ERROR: conf/config_wav2vec2.yaml not found"
    exit 1
fi

LOG_FILE="training_wav2vec2_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="training_wav2vec2.pid"

PYTHONUNBUFFERED=1 nohup python train.py --config conf/config_wav2vec2.yaml > "${LOG_FILE}" 2>&1 &
echo "$!" > "${PID_FILE}"
echo "Launched wav2vec2 training — PID: $(cat "${PID_FILE}"), Log: ${LOG_FILE}"
