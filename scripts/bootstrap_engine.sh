#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="${0:A:h:h}"
PROFILE="${1:-cpu}"
PYTHON="${PYTHON:-}"
VENV="$PROJECT_ROOT/.venv"

if [[ -z "$PYTHON" ]]; then
  for candidate in python3.12 python3.11 python3.13 python3; do
    RESOLVED="$(command -v "$candidate" 2>/dev/null || true)"
    if [[ -n "$RESOLVED" ]] && "$RESOLVED" -c \
      'import sys; raise SystemExit(not ((3, 11) <= sys.version_info[:2] < (3, 14)))'; then
      PYTHON="$RESOLVED"
      break
    fi
  done
fi
if [[ -z "$PYTHON" ]] || ! "$PYTHON" -c \
  'import sys; raise SystemExit(not ((3, 11) <= sys.version_info[:2] < (3, 14)))'; then
  print -u2 "Avril requires Python 3.11-3.13; Python 3.11 or 3.12 is recommended."
  exit 1
fi

# Building small Python wheels on ExFAT/network caches is unreliable: recent
# pip/setuptools can finish writing a wheel and then fail to discover it on the
# non-native filesystem.  Keep the installer cache on the host filesystem by
# default, even when the parent shell exports a shared PIP_CACHE_DIR.  Advanced
# users can still select an explicit native cache with AVRIL_PIP_CACHE_DIR.
if [[ -n "${AVRIL_PIP_CACHE_DIR:-}" ]]; then
  PIP_CACHE="$AVRIL_PIP_CACHE_DIR"
elif [[ "$(uname -s)" == "Darwin" ]]; then
  USER_CACHE_ROOT="$(getconf DARWIN_USER_CACHE_DIR 2>/dev/null || true)"
  if [[ -n "$USER_CACHE_ROOT" ]]; then
    PIP_CACHE="${USER_CACHE_ROOT%/}/avril/pip"
  else
    PIP_CACHE="$HOME/Library/Caches/Avril/pip"
  fi
else
  PIP_CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/avril/pip"
fi
mkdir -p "$PIP_CACHE"

case "$PROFILE" in
  cpu) EXTRA=""; BACKEND="cpu" ;;
  scout) EXTRA="[scout]"; BACKEND="cpu" ;;
  cuda12) EXTRA="[scout,cuda12]"; BACKEND="cuda" ;;
  *)
    print -u2 "Usage: $0 [cpu|scout|cuda12]"
    exit 2
    ;;
esac

"$PYTHON" -m venv "$VENV"
PIP_CACHE_DIR="$PIP_CACHE" "$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
PIP_CACHE_DIR="$PIP_CACHE" "$VENV/bin/python" -m pip install -e "$PROJECT_ROOT/Engine$EXTRA"
"$VENV/bin/avril" doctor --backend "$BACKEND"

print "Avril engine environment: $VENV"
print "Avril Python: $PYTHON"
print "Avril pip cache: $PIP_CACHE"
