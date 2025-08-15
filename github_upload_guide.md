# 将代码上传到GitHub的详细指南

## 步骤1：安装Git

1. 访问Git官方下载页面：<mcurl name="Git Downloads" url="https://git-scm.com/downloads"></mcurl>
2. 在Windows部分点击下载最新版本的Git安装程序
3. 运行下载的.exe安装文件
4. 在安装向导中，使用默认选项即可，点击"Next"直到完成安装
5. 安装完成后，按下`Win + R`，输入`cmd`打开命令提示符，运行`git --version`验证安装是否成功
   - 如果成功，会显示Git版本号；如果失败，请检查安装路径是否已添加到系统环境变量

## 步骤2：创建GitHub账号

1. 访问GitHub官网：<mcurl name="GitHub" url="https://github.com/"></mcurl>
2. 点击右上角的"Sign up"按钮
3. 输入您的邮箱地址，创建密码，选择用户名
4. 完成 CAPTCHA 验证
5. 按照提示完成账号创建流程，包括邮箱验证

## 步骤3：配置Git

1. 打开Git Bash（在开始菜单中搜索Git Bash）
2. 设置您的用户名：
   ```bash
   git config --global user.name "您的GitHub用户名"
   ```
3. 设置您的邮箱（与GitHub注册邮箱一致）：
   ```bash
   git config --global user.email "您的邮箱地址"
   ```
4. 验证配置：
   ```bash
   git config --list
   ```
   应该能看到您刚刚设置的用户名和邮箱

## 步骤4：在GitHub上创建新仓库

1. 登录GitHub账号
2. 点击右上角的"+"图标，选择"New repository"
3. 填写仓库名称（例如"log_time_model"）
4. 选择仓库类型（公开或私有）
5. 可选：添加README文件
6. 点击"Create repository"
7. 复制仓库的HTTPS或SSH地址（稍后会用到）
https://github.com/wonderingWu/log_time_model

## 步骤5：准备本地代码

1. 打开文件资源管理器，导航到 <mcfile name="regen_all" path="C:\log_time_model\regen_all"></mcfile> 目录
2. 确保该目录下包含您要上传的代码文件和out文件夹

## 步骤6：初始化本地仓库并提交代码

1. 打开Git Bash，导航到regen_all目录：
   ```bash
   cd /c/log_time_model/regen_all
   ```
2. 初始化本地Git仓库：
   ```bash
   git init
   ```
3. 将所有文件添加到暂存区：
   ```bash
   git add .
   ```
   （注意末尾的点表示添加当前目录下的所有文件）
4. 提交文件到本地仓库：
   ```bash
   git commit -m "首次提交代码和输出结果"
   ```
   （将引号内的文本替换为您的提交信息）

## 步骤7：连接到GitHub远程仓库并推送代码

1. 添加远程仓库（将下面的URL替换为您在步骤4中复制的仓库地址）：
   ```bash
   git remote add origin https://github.com/您的用户名/您的仓库名.git
   ```
2. 推送代码到GitHub：
   ```bash
   git push -u origin main
   ```
3. 首次推送时，会提示您输入GitHub的用户名和密码（或个人访问令牌）
   - 注意：GitHub现在推荐使用个人访问令牌而不是密码
   - 如何创建个人访问令牌：<mcurl name="创建GitHub个人访问令牌" url="https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token"></mcurl>

## 步骤8：验证上传结果

1. 返回GitHub仓库页面，刷新页面
2. 您应该能看到刚刚上传的代码文件和out文件夹

## 常见问题解决

1. **Git命令不可用**：检查Git是否已正确安装并添加到系统环境变量
2. **推送失败**：检查网络连接，确保仓库地址正确，验证GitHub凭证
3. **文件太大无法上传**：对于大型文件，考虑使用Git LFS（Large File Storage）
4. **中文乱码**：确保Git配置了正确的字符编码
   ```bash
   git config --global core.quotepath false
   git config --global i18n.commitencoding utf-8
   git config --global i18n.logoutputencoding utf-8
   ```

如果您遇到其他问题，可以参考GitHub帮助文档或搜索相关错误信息。