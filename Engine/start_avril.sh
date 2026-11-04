#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$project_dir"

if [[ ! -x .venv/bin/avril ]]; then
  "$project_dir/setup_avril.sh" cpu
fi

web_root=${AVRIL_WEB_ROOT:-}
if [[ -z "$web_root" ]]; then
  for candidate in \
    "$project_dir/../Avril-V3.0-Web/dist-pages" \
    "$project_dir/Avril-V3.0-Web/dist-pages" \
    "$project_dir/../Avril-V2.0-Web/dist-pages" \
    "$project_dir/Avril-V2.0-Web/dist-pages" \
    "$project_dir/../Avril-V1.0-Web/dist-pages" \
    "$project_dir/Avril-V1.0-Web/dist-pages" \
    "$project_dir/web/dist-pages"; do
    if [[ -f "$candidate/index.html" ]]; then
      web_root=$candidate
      break
    fi
  done
fi

if [[ -z "$web_root" || ! -f "$web_root/index.html" ]]; then
  echo "Extract the matching Avril Algorithm and Web packages side by side." >&2
  exit 1
fi

if [[ ${AVRIL_DRY_RUN:-0} == 1 ]]; then
  echo ".venv/bin/avril serve --web-root $web_root --open-browser"
  exit 0
fi

exec .venv/bin/avril serve --web-root "$web_root" --open-browser
