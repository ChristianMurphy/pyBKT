#!/usr/bin/env bash
# Render the generated SVGs to PNG with headless Chromium and ImageMagick.
# Usage: CHROME=/path/to/chrome docs/logo/render.sh
set -euo pipefail
cd "$(dirname "$0")"
CHROME="${CHROME:-chromium}"

shot() { # svg png width height
  "$CHROME" --headless --no-sandbox --disable-gpu --hide-scrollbars \
    --default-background-color=FFFFFFFF --window-size="$3,$(($4 + 300))" \
    --screenshot="$2" "file://$PWD/$1" >/dev/null 2>&1
  convert "$2" -crop "$3x$4+0+0" +repage "$2"
}

mkdir -p png
for f in option-*.svg; do
  shot "$f" "png/${f%.svg}.png" 512 512
done
shot overview.svg png/overview.png 1760 440
for f in mockups/*-light-dark.svg; do
  shot "$f" "png/$(basename "${f%.svg}").png" 1024 512
done
for f in mockups/*-contexts.svg; do
  shot "$f" "png/$(basename "${f%.svg}").png" 1600 1000
done
