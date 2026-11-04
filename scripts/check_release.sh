#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="${0:A:h:h}"
cd "$PROJECT_ROOT"

print "[1/5] Checking for machine-specific paths and common credentials"
if rg -n --hidden \
  -g '!.git/**' -g '!.build/**' -g '!.venv/**' -g '!docs/assets/*.svg' \
  -g '!scripts/check_release.sh' \
  '(/Users/[^/]+/|/Volumes/|connect\.westd|seetacloud|root@|BEGIN (RSA |OPENSSH )?PRIVATE KEY|password\s*[:=])' .; then
  print -u2 "Release check failed: machine-specific or sensitive text remains."
  exit 1
fi

print "[2/5] Checking generated artifacts"
if find . -type f \( -name '*.fits' -o -name '*.fits.part' -o -name '*.pyc' -o -name '*.log' \) -not -path './.build/*' -not -path './.venv/*' | grep -q .; then
  print -u2 "Release check failed: generated science or cache files remain."
  exit 1
fi

print "[3/5] Compiling the Python source tree"
PYTHONPYCACHEPREFIX="$PROJECT_ROOT/.build/pycache" python3 -m compileall -q Engine

print "[4/5] Building the native macOS target"
swift build -c release

print "[5/5] Verifying required release documents"
for path in \
  README.md README_zh.md LICENSE CITATION.cff CHANGELOG.md CONTRIBUTING.md SECURITY.md \
  CODE_OF_CONDUCT.md THIRD_PARTY_NOTICES.md \
  docs/RESEARCH_OVERVIEW.md docs/SCIENTIFIC_CONTRACT.md docs/ARCHITECTURE.md \
  docs/TARGET_PRIOR.md docs/VALIDATION.md docs/DATA_AND_PRIVACY.md docs/RELEASING.md \
  docs/assets/README.md \
  docs/assets/avril-hero.png docs/assets/avril-hero-scene.png \
  docs/assets/avril-discovery-architecture.png docs/assets/planetary-census-gap-ai.png \
  docs/assets/avril-core-vetting.png \
  docs/assets/planetary-census-gap-validation.png docs/assets/planetary-census-gap-validation.svg \
  docs/assets/tess-discovery-queue-atlas.png docs/assets/tess-discovery-queue-atlas.svg \
  docs/assets/system-architecture.svg docs/assets/target-prior-model.svg \
  docs/assets/github-social-preview.png scripts/render_research_figures.py; do
  [[ -s "$path" ]] || { print -u2 "Missing $path"; exit 1; }
done

print "Open-source release checks passed."
