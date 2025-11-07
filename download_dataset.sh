#!/usr/bin/env bash
set -e

TARGET_DIR="./dataset"

echo "📦 Downloading dataset: code-philia/TRACE ..."
echo "📁 Target directory: $TARGET_DIR"

# Create target directory
mkdir -p "$TARGET_DIR"

# Check whether hf CLI is installed
if ! command -v hf &> /dev/null
then
    echo "⚙️  hf CLI not found. Installing..."
    pip install -U "huggingface_hub[cli]"
fi

# Download the dataset into the target directory
hf download code-philia/TRACE \
    --repo-type dataset \
    --local-dir "$TARGET_DIR"

echo "✅ Download complete. Files saved to $TARGET_DIR"