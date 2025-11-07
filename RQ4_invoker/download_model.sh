#!/bin/bash
# Download specified subdirectory contents from a Hugging Face repository (invoker/)
# Target: https://huggingface.co/code-philia/TRACE/tree/main/invoker

REPO_URL="https://huggingface.co/code-philia/TRACE"
TARGET_DIR="invoker/model"

echo "Cloning repository with sparse-checkout ..."
git lfs install
git clone --depth 1 --filter=blob:none --sparse "$REPO_URL" TRACE_temp
cd TRACE_temp
git sparse-checkout set "$TARGET_DIR"
cd ..

# Copy invoker subdirectory to current directory
if [ -d "TRACE_temp/$TARGET_DIR" ]; then
    echo "Copying $TARGET_DIR to current directory ..."
    cp -r "TRACE_temp/$TARGET_DIR" ./
    echo "✅ Download completed: $(pwd)/$TARGET_DIR"
else
    echo "❌ Error: $TARGET_DIR not found in repository."
fi

# Clean up temporary directory
rm -rf TRACE_temp

TARGET_DIR="invoker/dataset"

echo "Cloning repository with sparse-checkout ..."
git lfs install
git clone --depth 1 --filter=blob:none --sparse "$REPO_URL" TRACE_temp
cd TRACE_temp
git sparse-checkout set "$TARGET_DIR"
cd ..

# Copy invoker subdirectory to current directory
if [ -d "TRACE_temp/$TARGET_DIR" ]; then
    echo "Copying $TARGET_DIR to current directory ..."
    cp -r "TRACE_temp/$TARGET_DIR" ./
    echo "✅ Download completed: $(pwd)/$TARGET_DIR"
else
    echo "❌ Error: $TARGET_DIR not found in repository."
fi

# Clean up temporary directory
rm -rf TRACE_temp