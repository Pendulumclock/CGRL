#!/bin/bash
commit_msg=${1:-"Update project files"}

echo "📂 当前目录：$(pwd)"
echo "🔍 检查更改状态..."
git status

echo "➕ 添加所有更改文件..."
git add .

echo "💬 提交更改：$commit_msg"
git commit -m "$commit_msg"

echo "🚀 推送到远程仓库..."
git push

echo "✅ 上传完成！"
