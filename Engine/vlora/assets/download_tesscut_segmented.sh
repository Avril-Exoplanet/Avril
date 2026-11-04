#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "usage: $0 OUTPUT_DIR RA DEC SIZE_PX CONNECTIONS_PER_FILE SECTOR..." >&2
  exit 2
fi

output_dir=$1
ra=$2
dec=$3
size_px=$4
connections_per_file=$5
shift 5
sectors=("$@")
file_workers=${VLORA_TESSCUT_FILE_WORKERS:-8}

if ! [[ "$connections_per_file" =~ ^[1-9][0-9]*$ ]]; then
  echo "CONNECTIONS_PER_FILE must be a positive integer" >&2
  exit 2
fi
if ! [[ "$file_workers" =~ ^[1-9][0-9]*$ ]]; then
  echo "VLORA_TESSCUT_FILE_WORKERS must be a positive integer" >&2
  exit 2
fi

mkdir -p "$output_dir"

download_sector() (
  set -euo pipefail
  sector=$1
  padded=$(printf '%04d' "$sector")
  final="$output_dir/sector-${padded}.zip"
  header="$output_dir/sector-${padded}.headers"
  api="https://mast.stsci.edu/tesscut/api/v0.1/astrocut?x=${size_px}&y=${size_px}&units=px&sector=${sector}&ra=${ra}&dec=${dec}"

  if [[ -f "$final" ]] && unzip -tq "$final" >/dev/null 2>&1; then
    echo "READY sector=$sector bytes=$(stat -f%z "$final")"
    exit 0
  fi

  direct=$(
    curl --http1.1 --location --fail --silent --show-error \
      --retry 20 --retry-all-errors --retry-delay 1 \
      --connect-timeout 20 --max-time 900 \
      --range 0-0 --dump-header "$header" --output /dev/null \
      --write-out '%{url_effective}' "$api"
  )
  total=$(awk 'tolower($1) == "content-range:" {sub("\r$", ""); sub(".*/", ""); value=$0} END{print value}' "$header")
  if [[ ! "$total" =~ ^[0-9]+$ ]] || (( total < 1 )); then
    echo "could not determine staged S3 object length for sector $sector" >&2
    exit 1
  fi

  chunk=$(( (total + connections_per_file - 1) / connections_per_file ))
  part_paths=()
  expected_sizes=()
  for (( index=0; index<connections_per_file; index++ )); do
    start=$(( index * chunk ))
    (( start < total )) || break
    end=$(( start + chunk - 1 ))
    (( end < total )) || end=$(( total - 1 ))
    part="$output_dir/sector-${padded}.part-${index}"
    expected=$(( end - start + 1 ))
    part_paths+=("$part")
    expected_sizes+=("$expected")
    if [[ -f "$part" ]] && [[ $(stat -f%z "$part") -eq $expected ]]; then
      continue
    fi
    (
      curl --http1.1 --location --fail --silent --show-error \
        --retry 30 --retry-all-errors --retry-delay 1 \
        --connect-timeout 20 --max-time 900 \
        --range "${start}-${end}" --output "${part}.download" "$direct"
      [[ $(stat -f%z "${part}.download") -eq $expected ]]
      mv "${part}.download" "$part"
    ) &
  done
  wait

  for (( index=0; index<${#part_paths[@]}; index++ )); do
    [[ -f "${part_paths[$index]}" ]]
    [[ $(stat -f%z "${part_paths[$index]}") -eq ${expected_sizes[$index]} ]]
  done

  assembled="$output_dir/sector-${padded}.assembled"
  : > "$assembled"
  for part in "${part_paths[@]}"; do
    cat "$part" >> "$assembled"
  done
  [[ $(stat -f%z "$assembled") -eq $total ]]
  unzip -tq "$assembled" >/dev/null
  mv "$assembled" "$final"
  rm -f -- "$header" "${part_paths[@]}"
  echo "DONE sector=$sector bytes=$total chunks=${#part_paths[@]}"
)

pids=()
status=0
for sector in "${sectors[@]}"; do
  download_sector "$sector" &
  pids+=("$!")
  if (( ${#pids[@]} >= file_workers )); then
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        status=1
      fi
    done
    pids=()
  fi
done

for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
