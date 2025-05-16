#!/bin/bash

# 构建 Astro 项目
npm run build

# 复制 dist 内容到当前目录（包含隐藏文件，如 .nojekyll）
cp -r dist/* .
cp -r dist/.* . 2>/dev/null

# 提交并推送到 GitHub main 分支
git add .
git commit -m "Deploy $(date +'%Y-%m-%d %H:%M:%S')"
git push origin main
