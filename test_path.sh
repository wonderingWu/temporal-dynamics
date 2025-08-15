#!/bin/bash
# 此脚本用于测试Git Bash中的路径是否正确

echo "测试导航到regen_all目录..."
cd /c/log_time_model/regen_all

if [ $? -eq 0 ]; then
    echo "成功导航到目录！当前目录："
    pwd
    echo "
路径测试成功！您可以继续按照github_upload_guide.md中的步骤操作。"
else
    echo "导航失败，请检查路径是否正确。"
    echo "当前目录："
    pwd
    echo "
请确保您的文件夹结构为c:\log_time_model\regen_all"
fi