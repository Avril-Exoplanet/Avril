#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="${0:A:h:h}"
if [[ -n "${AVRIL_NOVA_BUILD_ROOT:-}" ]]; then
  BUILD_ROOT="$AVRIL_NOVA_BUILD_ROOT"
else
  # A source checkout may live in iCloud Drive or another File Provider folder.
  # Those folders can reattach FinderInfo after signing, which strict codesign
  # correctly rejects. Assemble the application in the native user cache by
  # default; callers may still provide an explicit release destination.
  USER_CACHE_ROOT="$(getconf DARWIN_USER_CACHE_DIR 2>/dev/null || true)"
  if [[ -n "$USER_CACHE_ROOT" ]]; then
    BUILD_ROOT="${USER_CACHE_ROOT%/}/avril-nova-app"
  else
    BUILD_ROOT="$PROJECT_ROOT/.build/avril-nova-app"
  fi
fi
SCRATCH_ROOT="$BUILD_ROOT/swift-build"
APP="$BUILD_ROOT/Avril Nova.app"
CONTENTS="$APP/Contents"
RESOURCES="$CONTENTS/Resources"
ALGORITHM_SOURCE="$PROJECT_ROOT/Engine"

mkdir -p "$BUILD_ROOT" "$CONTENTS/MacOS" "$CONTENTS/Frameworks" "$RESOURCES"

swift build \
  --package-path "$PROJECT_ROOT" \
  --scratch-path "$SCRATCH_ROOT" \
  -c release \
  -Xswiftc -gnone

cp "$SCRATCH_ROOT/release/AvrilNova" "$CONTENTS/MacOS/AvrilNova"
rsync -a --delete \
  "$PROJECT_ROOT/Vendor/CorePlot.xcframework/macos-arm64/CorePlot.framework/" \
  "$CONTENTS/Frameworks/CorePlot.framework/"
cp "$PROJECT_ROOT/Info.plist" "$CONTENTS/Info.plist"
rsync -a --delete \
  --exclude 'python-runtime/' \
  --exclude 'Avril-V3.0-Algorithm/' \
  "$PROJECT_ROOT/Resources/" "$RESOURCES/"
swift "$PROJECT_ROOT/scripts/make_icon.swift" "$BUILD_ROOT"
cp "$BUILD_ROOT/AvrilNova.icns" "$RESOURCES/AvrilNova.icns"

if [[ "${AVRIL_NOVA_SKIP_RUNTIME:-0}" != "1" ]]; then
  if [[ -n "${AVRIL_NOVA_PYTHON_RUNTIME:-}" ]]; then
    if [[ ! -x "$AVRIL_NOVA_PYTHON_RUNTIME/bin/python3" && ! -x "$AVRIL_NOVA_PYTHON_RUNTIME/bin/python3.13" ]]; then
      print -u2 "AVRIL_NOVA_PYTHON_RUNTIME must contain bin/python3 or bin/python3.13"
      exit 1
    fi
    rsync -a --delete \
      "$AVRIL_NOVA_PYTHON_RUNTIME/" \
      "$RESOURCES/python-runtime/"
  else
    print -u2 "Building without an embedded Python runtime."
    print -u2 "For a distributable app, set AVRIL_NOVA_PYTHON_RUNTIME to a relocatable runtime."
  fi
fi

rsync -a --delete \
  --delete-excluded \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '.pytest_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude 'outputs/' \
  --exclude 'results/' \
  "$ALGORITHM_SOURCE/" \
  "$RESOURCES/Avril-V3.0-Algorithm/"

# Finder metadata and quarantine/resource-fork xattrs are not valid inside a
# signed app bundle.  Clear them only from the freshly assembled build output.
xattr -cr "$APP"
codesign --force --deep --sign - "$APP"
xattr -cr "$APP"
codesign --verify --deep --strict "$APP"
print "$APP"
