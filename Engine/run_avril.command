#!/bin/zsh
set -euo pipefail
project_dir=${0:A:h}
cd "$project_dir"
[[ -x .venv/bin/avril ]] || { echo 'Run ./setup_avril.sh cpu first.' >&2; exit 1; }
exec .venv/bin/avril run "$@"
