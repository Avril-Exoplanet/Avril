#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="${0:A:h:h}"
BUILD_ROOT="${AVRIL_NOVA_BUILD_ROOT:-$PROJECT_ROOT/.build/avril-nova-app}"
APP="$BUILD_ROOT/Avril Exoplanet Workstation.app"
PYTHON="${AVRIL_NOVA_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"

if [[ ! -x "$APP/Contents/MacOS/AvrilNova" ]]; then
  AVRIL_NOVA_SKIP_RUNTIME=1 "$PROJECT_ROOT/scripts/build_app.sh"
fi
if [[ ! -x "$PYTHON" ]]; then
  print -u2 "No development Python environment found."
  print -u2 "Run: ./scripts/bootstrap_engine.sh cpu"
  exit 1
fi

export AVRIL_NOVA_PYTHON="$PYTHON"
export AVRIL_NOVA_ENGINE_DIR="$PROJECT_ROOT/Engine"
exec "$APP/Contents/MacOS/AvrilNova"
