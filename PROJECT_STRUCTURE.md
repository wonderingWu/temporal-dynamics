# 项目结构说明

本项目已准备就绪，可直接上传到GitHub。以下是完整的项目结构：

## 根目录文件
```
├── README.md                    # 项目说明文档
├── requirements.txt             # Python依赖列表
├── setup.py                    # Python包安装脚本
├── LICENSE                     # MIT许可证
├── .gitignore                  # Git忽略文件
├── CONTRIBUTING.md             # 贡献指南
├── download_jar.py             # JIDT工具包下载脚本
├── .github/                    # GitHub配置
│   ├── workflows/ci.yml        # CI/CD工作流
│   └── ISSUE_TEMPLATE/bug_report.yml  # Bug报告模板
├── src/                        # Python源代码
├── tests/                      # 测试文件
├── docs/                       # 文档和论文
├── data/                       # 数据文件目录
└── figures/                    # 图表文件
```

## 核心功能模块

### src/ 目录
- `ESN_criticality_te_fixed.py` - Echo State Networks临界性分析
- `ising_ais_nmi_quick_revised.py` - 2D Ising模型信息论分析
- `logicMapTest_fixed.py` - Logistic映射混沌分析
- `sandpile_critical_time_analysis.py` - 沙堆模型临界时间分析
- `sandpile_reset_vs_te.py` - 沙堆重置vs传递熵对比
- `main.py` - 项目主入口和命令行界面

### docs/ 目录
- `temporalDynamics.tex` - 完整的LaTeX学术论文

### figures/ 目录
- `ESN_criticality_te_sig.png` - ESN临界性传递熵分析
- `TE_vs_Temperature_2D_Ising_full_plus.pdf` - Ising模型温度扫描
- `ising_info_dynamics_quick.png` - Ising模型信息动力学
- `ising_nmi_ais_revised.png` - Ising模型NMI/AIS对比
- `ising_shuffing_nmi_micro.png` - Ising模型打乱基准测试
- `te_vs_reset_sandpile.pdf` - 沙堆模型传递熵vs重置频率

## GitHub部署准备

### 1. 仓库创建
1. 登录GitHub，创建新仓库 `temporal-dynamics`
2. 设为公开或私有
3. 初始化README.md（跳过，我们已经有了）

### 2. 本地Git初始化
```bash
git init
git add .
git commit -m "Initial commit: Complete temporal dynamics project"
git branch -M main
git remote add origin https://github.com/[your-username]/temporal-dynamics.git
git push -u origin main
```

### 3. 论文引用格式
在论文中添加GitHub引用：
```latex
Code and data available at: \url{https://github.com/[username]/temporal-dynamics}
```

## 项目特色

### ✅ 完整的研究框架
- 4种不同复杂系统的对比分析
- 统一的信息论方法论
- 详细的实验结果和可视化

### ✅ 专业的代码组织
- 模块化设计
- 完整的测试覆盖
- 详细的文档和注释

### ✅ GitHub就绪
- 标准项目结构
- CI/CD工作流
- Issue和PR模板
- 详细的贡献指南

### ✅ 开源友好
- MIT许可证
- 清晰的README
- 易于复现的实验
- 完整的依赖管理

## 使用方法

### 快速开始
1. 克隆仓库：`git clone https://github.com/[username]/temporal-dynamics.git`
2. 安装依赖：`pip install -r requirements.txt`
3. 下载JIDT：`python download_jar.py`
4. 运行分析：`python src/main.py --system all`

### 单独运行
- ESN分析：`python src/main.py --system esn`
- Ising模型：`python src/main.py --system ising`
- Logistic映射：`python src/main.py --system logistic`
- 沙堆模型：`python src/main.py --system sandpile`

## 学术价值

本项目为复杂系统时间动力学的研究提供了：
- 统一的信息论分析框架
- 多种系统的对比研究
- 详细的方法论说明
- 完整的实验重现性

项目已准备好提交到GitHub，可以直接在学术论文中引用，提升研究的可重复性和影响力。