#!/usr/bin/env bash
set -e

TARGET_DIR="./dataset"

echo "📦 Downloading dataset: code-philia/TRACE ..."
echo "📁 Target directory: $TARGET_DIR"

# 创建目标目录
mkdir -p "$TARGET_DIR"

# 检查 huggingface CLI 是否安装
if ! command -v huggingface &> /dev/null
then
    echo "⚙️  huggingface CLI not found. Installing..."
    pip install -U "huggingface_hub[cli]"
fi

# 如果需要登录（私有数据集），可取消下一行注释
# huggingface-cli login

# 下载数据集到指定目录
huggingface-cli download code-philia/TRACE \
    --repo-type dataset \
    --local-dir "$TARGET_DIR"

echo "✅ Download complete. Files saved to $TARGET_DIR"