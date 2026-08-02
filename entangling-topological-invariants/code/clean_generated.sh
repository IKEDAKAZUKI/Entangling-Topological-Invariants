#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
rm -rf data figures
mkdir -p data figures
printf 'Removed generated numerical data and figures.\n'
