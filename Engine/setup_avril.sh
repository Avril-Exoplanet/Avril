#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$project_dir"

# Avoid wheel-build failures on ExFAT and network-mounted pip caches.  Avril's
# dependency cache contains many small files and belongs on the native system
# disk.  AVRIL_PIP_CACHE_DIR remains available for an explicit override.
if [[ -n "${AVRIL_PIP_CACHE_DIR:-}" ]]; then
  pip_cache=$AVRIL_PIP_CACHE_DIR
elif [[ "$(uname -s)" == "Darwin" ]]; then
  user_cache_root=$(getconf DARWIN_USER_CACHE_DIR 2>/dev/null || true)
  if [[ -n "$user_cache_root" ]]; then
    pip_cache="${user_cache_root%/}/avril/pip"
  else
    pip_cache="$HOME/Library/Caches/Avril/pip"
  fi
else
  pip_cache="${XDG_CACHE_HOME:-$HOME/.cache}/avril/pip"
fi
mkdir -p "$pip_cache"

profile=${1:-cpu}
case "$profile" in
  cpu) requirement=requirements/cpu.txt ;;
  cuda) requirement=requirements/cuda12.txt ;;
  dev) requirement=requirements/dev.txt ;;
  *) echo "usage: ./setup_avril.sh [cpu|cuda|dev]" >&2; exit 2 ;;
esac

python_bin=${AVRIL_PYTHON:-}
if [[ -z "$python_bin" ]]; then
  for candidate in python3.12 python3.11 python3.13 python3; do
    resolved=$(command -v "$candidate" || true)
    if [[ -n "$resolved" ]] && "$resolved" -c \
      'import sys; raise SystemExit(not ((3, 11) <= sys.version_info[:2] < (3, 14)))'; then
      python_bin=$resolved
      break
    fi
  done
fi
if [[ -z "$python_bin" ]]; then
  echo "Python 3.11-3.13 is required; Python 3.11 or 3.12 is recommended." >&2
  exit 1
fi
if ! "$python_bin" -c 'import sys; raise SystemExit(not ((3, 11) <= sys.version_info[:2] < (3, 14)))'; then
  echo "Avril requires Python 3.11-3.13; Python 3.11 or 3.12 is recommended." >&2
  exit 1
fi

[[ -x .venv/bin/python ]] || "$python_bin" -m venv .venv
PIP_CACHE_DIR="$pip_cache" .venv/bin/python -m pip install --upgrade pip setuptools wheel
PIP_CACHE_DIR="$pip_cache" .venv/bin/python -m pip install -r "$requirement"
PIP_CACHE_DIR="$pip_cache" .venv/bin/python -m pip install . --no-deps --force-reinstall
.venv/bin/avril doctor --backend "$([[ "$profile" == cuda ]] && echo cuda || echo cpu)"

echo "Avril is ready. Activate it with: source .venv/bin/activate"
echo "Avril pip cache: $pip_cache"
