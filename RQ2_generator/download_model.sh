#!/bin/bash
REPO_URL="https://huggingface.co/code-philia/TRACE"
TARGET_DIR="generator/model_6"

git clone --depth 1 --filter=blob:none --sparse "$REPO_URL" TRACE_temp
cd TRACE_temp
git sparse-checkout set "$TARGET_DIR"
cd ..
cp -r "TRACE_temp/$TARGET_DIR" ./
rm -rf TRACE_temp
echo "✅ Downloaded: $TARGET_DIR"