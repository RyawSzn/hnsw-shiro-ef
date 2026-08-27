#!/bin/bash

# ==============================================
# 超简同步脚本 - 利用本地 ~/.ssh/config
# 用法: ./sync.sh [源路径] [目标路径]
#   示例: ./sync.sh ./my_project/
# ==============================================

# 1. 处理源路径（默认当前目录）
SOURCE_DIR="${1:-.}"
SOURCE_DIR="${SOURCE_DIR%/}"  # 去掉末尾斜杠

# 2. 自动生成远程路径（取文件夹名）
FOLDER_NAME=$(basename "$(cd "$SOURCE_DIR" && pwd)")
REMOTE_PATH="yc.shea@ssd11:~/dev/${FOLDER_NAME}/"

# 如果用户指定了第二个参数，则覆盖默认目标
if [ -n "$2" ]; then
    REMOTE_PATH="yc.shea@ssd11:$2"
fi

# 3. 确认信息
echo "From: $SOURCE_DIR"
echo "To: $REMOTE_PATH"
echo "----------------------------------------"

# 4. 执行同步（包含所有忽略项）
rsync -avnz --progress --delete \
    -e 'ssh -J yc.shea@projgw.cse.cuhk.edu.hk:2241' \
    --exclude='hnswlib.egg-info/' \
    --exclude='build/' \
    --exclude='dist/' \
    --exclude='tmp/' \
    --exclude='graphify-out/cache/' \
    --exclude='python_bindings/tests/__pycache__/' \
    --exclude='research/__pycache__/' \
    --exclude='*.pyd' \
    --exclude='hnswlib.cpython*.so' \
    --exclude='var/' \
    --exclude='.idea/' \
    --exclude='.vs/' \
    --exclude='.cache/' \
    --exclude='.opencode/' \
    --exclude='.sisyphus/' \
    --exclude='.omo/' \
    --exclude='.slim/deepwork' \
    --exclude='**.DS_Store' \
    --exclude='*.pyc' \
    --exclude='.venv' \
    --exclude='graphify-out/cost.json' \
    --exclude='index' \
    "$SOURCE_DIR/" "$REMOTE_PATH"

# 5. 检查结果
if [ $? -eq 0 ]; then
    echo "Sync Complete！"
else
    echo "Sync failed, please check your connection and credentials。"
fi
