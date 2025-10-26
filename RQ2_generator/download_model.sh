#!/bin/bash
# Download specified subdirectory contents from a Hugging Face repository (locator/)
# Target: https://huggingface.co/code-philia/TRACE/tree/main/locator

REPO_URL="https://huggingface.co/code-philia/TRACE"
TARGET_DIR="generator/model_6"

# Create temporary directory
TMP_DIR=$(mktemp -d)
echo "Cloning repository into $TMP_DIR ..."
git lfs install
git clone --depth 1 "$REPO_URL" "$TMP_DIR"

# Copy locator subdirectory to current directory
if [ -d "$TMP_DIR/$TARGET_DIR" ]; then
    echo "Copying $TARGET_DIR to current directory ..."
    cp -r "$TMP_DIR/$TARGET_DIR" ./
    echo "✅ Download completed: $(pwd)/$TARGET_DIR"
else
    echo "❌ Error: $TARGET_DIR not found in repository."
fi

# Clean up temporary directory
rm -rf "$TMP_DIR"