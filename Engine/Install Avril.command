#!/bin/zsh
set -u

project_dir=${0:A:h}
cd "$project_dir" || exit 1

if /bin/bash "$project_dir/setup_avril.sh" cpu; then
  status=0
  echo
  echo "Avril is installed. Double-click 'Start Avril.command' to open it."
else
  status=$?
  echo
  echo "Installation failed. Keep this window open and share the messages above."
fi

echo "Press Return to close this window."
read -r
exit "$status"
