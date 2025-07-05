#!/bin/bash

VERSION=""
SED_ARGS=()
MAX_RETRIES=10

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)
      VERSION="$2"
      shift 2
      ;;
    --sed)
      SED_ARGS+=("$2")
      shift 2
      ;;
    --retries)
      MAX_RETRIES="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done


SRC_DIR=".genai"
if [[ ! -d "$SRC_DIR" ]]; then
    git clone https://github.com/microsoft/onnxruntime-genai.git "$SRC_DIR"
fi

cd "$SRC_DIR"

if [[ -n "$VERSION" ]]; then
    git fetch --tags
    git checkout "v$VERSION"
fi

for sed_expr in "${SED_ARGS[@]}"; do
    sed -i "$sed_expr" cmake/deps.txt
done

count=0

while [ $count -lt $MAX_RETRIES ]; do
    echo "Attempt $((count+1))..."
    python build.py --config Release
    if [ $? -eq 0 ]; then
        echo "Build succeeded."
        exit 0
    fi
    count=$((count+1))
    echo "Build failed. Retry with `MAKEFLAGS=-j1` and sed `build/Linux/Release/_deps` replacements: ${SED_ARGS[@]}"

    export MAKEFLAGS=-j1
    for sed_expr in "${SED_ARGS[@]}"; do
        eval "find build/Linux/Release/_deps -type f -name '*.cmake' -exec sed -i \"$sed_expr\" {} +"
    done
done
