# Contributing to Temporal Dynamics Project

首先，感谢您对本项目的兴趣！我们欢迎所有形式的贡献，包括代码改进、错误修复、文档改进和新的功能实现。

## 快速开始

1. Fork 这个仓库
2. 克隆您的fork到本地：`git clone https://github.com/[your-username]/temporal-dynamics.git`
3. 创建开发分支：`git checkout -b feature/your-feature-name`
4. 安装开发依赖：`pip install -r requirements.txt`
5. 运行测试确保一切正常

## 开发环境设置

### 环境要求
- Python 3.8+
- LaTeX (用于编译文档)
- Java 8+ (用于JIDT工具包)

### 依赖安装
```bash
# 安装所有依赖
pip install -r requirements.txt

# 下载JIDT工具包
python download_jar.py
```

## 代码规范

### Python代码风格
- 遵循 [PEP 8](https://pep8.org/) 标准
- 使用 [Black](https://black.readthedocs.io/) 进行代码格式化
- 使用 [Flake8](https://flake8.pycqa.org/) 进行代码检查
- 函数和类使用docstring注释

### 提交消息格式
使用清晰的提交消息：
```
type(scope): short description

Longer description if needed.

Fixes #issue-number
```

类型 (type) 包括：
- `feat`: 新功能
- `fix`: 错误修复
- `docs`: 文档更改
- `style`: 代码格式调整
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建或工具更改

## 测试

运行完整测试套件：
```bash
pytest tests/
```

运行特定测试：
```bash
pytest tests/test_specific_module.py
```

## 新功能开发

如果您想添加新功能：

1. 在 `CONTRIBUTING.md` 中描述您的计划
2. 创建详细的功能描述
3. 添加相应的测试用例
4. 更新文档

### 新系统集成

要添加新的物理系统：

1. 在 `src/` 目录创建新模块
2. 实现基础类接口
3. 添加相应的分析和可视化函数
4. 在主文件中集成
5. 更新文档和示例

## 错误报告

如果您发现bug，请创建Issue并包含：

- 操作系统和Python版本
- 完整的错误消息
- 重现问题的步骤
- 期望的行为
- 相关的代码片段或配置文件

## 文档改进

文档对于项目至关重要，您可以：

- 改进README和注释
- 添加代码示例
- 修复拼写和语法错误
- 翻译文档到其他语言

## Pull Request流程

1. 确保代码通过所有测试
2. 更新相关文档
3. 添加测试覆盖新功能
4. 确保代码符合项目规范
5. 提交PR并等待审查

## 审查准则

审查将关注：
- 代码质量和可读性
- 测试覆盖
- 文档完整性
- 性能影响
- 与项目目标的一致性

## 许可证

通过贡献，您同意您的贡献将在MIT许可证下发布。

## 行为准则

请保持友善和专业的态度。我们致力于为所有人提供一个包容性的环境。

## 联系方式

如果您有疑问或需要帮助：
- 创建GitHub Issue
- 发送邮件到项目维护者
- 参与现有讨论

感谢您的贡献！