#!/usr/bin/env bash
# Ensure libomp is available on macOS (required for some ML libs).
# Usage: ./scripts/ensure_libomp.sh

set -e

if [[ "$(uname)" != "Darwin" ]]; then
  exit 0
fi

for dir in /opt/homebrew/opt/libomp /usr/local/opt/libomp; do
  if [[ -f "$dir/lib/libomp.dylib" ]]; then
    exit 0
  fi
done

echo "Run: brew install libomp"
exit 1
