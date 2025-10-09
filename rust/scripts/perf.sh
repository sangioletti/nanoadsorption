#!/bin/bash

# See: https://sipb.mit.edu/doc/safe-shell/
set -euf -o pipefail

# Get the directory of this script in a relatively robust fashion (pun intended). See:
# http://www.binaryphile.com/bash/2020/01/12/determining-the-location-of-your-script-in-bash.html
here=$(cd "$(dirname "$BASH_SOURCE")"; cd -P "$(dirname "$(readlink "$BASH_SOURCE" || echo .)")"; pwd)

cargo build -p nanoadsorption --release

FILE="$here/../perf.data"

if [ -f "$FILE" ]; then
  rm "$FILE"
fi

# Create flamegraphs directory if it doesn't exist
FLAMEGRAPH_DIR="$here/../flamegraphs"
mkdir -p "$FLAMEGRAPH_DIR"

# Find the highest numbered flamegraph file and increment
NEXT_NUM=1
# Use find to get all existing flamegraph files
while IFS= read -r file; do
  # Extract the number from the filename
  NUM=$(basename "$file" .svg | sed 's/flamegraph_//')
  # Only process if NUM is a valid number
  if [[ "$NUM" =~ ^[0-9]+$ ]] && [ "$NUM" -ge "$NEXT_NUM" ]; then
    NEXT_NUM=$((NUM + 1))
  fi
done < <(find "$FLAMEGRAPH_DIR" -maxdepth 1 -name "flamegraph_*.svg" 2>/dev/null || true)

OUTPUT_FILE="$FLAMEGRAPH_DIR/flamegraph_$NEXT_NUM.svg"

cargo flamegraph -o "$OUTPUT_FILE" -p nanoadsorption --release 
echo "Flamegraph saved to: $OUTPUT_FILE"