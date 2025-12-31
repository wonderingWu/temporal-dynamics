# log_time_model

## 项目概述

本项目实现了一个基于对数时间（logarithmic time）的分析框架，用于研究复杂系统的演化行为。通过将传统线性时间 `t` 转换为对数时间 `τ = ln(t/t₀)`，可以更有效地观察系统在不同时间尺度上的动力学特性。

项目主要包含以下几个部分：
- 对数时间变换和等τ重采样方法
- 多种物理系统的模拟与分析（Ising模型、量子Ising模型、元胞自动机等）
- 宇宙微波背景（CMB）数据的处理与分析
- 统一的数据可视化和结果输出系统

## 安装指南

### 依赖环境
- Python 3.8+（推荐3.10+）
- 所需Python库：
  ```
  requests>=2.31.0
  pandas>=2.2.2
  numpy>=1.26.4
  matplotlib>=3.8.4
  astropy>=6.0.0
  ```
  可选依赖：
  ```
  numba>=0.59.0  # 用于加速计算
  qutip>=4.7.5   # 用于量子系统模拟
  ```

### 安装步骤
1. 克隆本仓库：
   ```bash
   git clone https://github.com/wonderingWu/log_time_model.git
   cd log_time_model/regen_all
   ```
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 使用方法

### 核心框架
```python
python logtime_minimal_framework.py --all  # 运行所有示例
```

### 特定系统分析
```python
# Ising模型分析
python logtime_minimal_framework.py --ising --ising-L 64 --ising-steps 200000 \
       --ising-burn 100000 --ising-thin 200 --n-real 10 --seed 123

# 元胞自动机分析
python logtime_minimal_framework.py --ca --ca-rule 30 110 --ca-size 256 \
       --ca-steps 4000 --ca-window-rows 256 --n-real 8 --seed 42

# 量子Ising模型分析
python logtime_minimal_framework.py --tfim --tfim-N 8 --tfim-tmax 20 --n-real 10

# 宇宙学数据分析
python cosmology_analysis.py --experiment Planck
```

### 数据可视化
```python
# 生成Ising模型能量演化对比图
python ising_comparison.py
```

## 文件结构

```
regen_all/
├── cosmology_analysis.py   # 宇宙学数据分析脚本
├── ising_comparison.py     # Ising模型对比分析
├── logtime_minimal_framework.py  # 对数时间框架核心脚本
├── out/                    # 输出文件夹
│   ├── ca/                 # 元胞自动机结果
│   ├── cosmology/          # 宇宙学分析结果
│   ├── ising/              # Ising模型结果
│   └── tfim*/              # 量子Ising模型结果
├── README.md               # 项目说明文档
└── requirements.txt        # 依赖库列表
```

## 功能详情

### 对数时间框架
- 实现了一致的对数时间变换 `τ = ln(t/t₀)`
- 支持等τ重采样方法，确保在对数时间尺度上均匀采样
- 提供鲁棒的参数化指标：总变差(TV)、平面弧长、曲率积分等
- 支持多次实现、随机种子控制和置信区间聚合

### Ising模型分析
- 2D Ising模型的蒙特卡洛模拟
- 支持Burn-in、 thinning和多次实现平均
- 能量和方差演化的线性时间与对数时间对比
- 自动生成对比图表和数据文件

### 宇宙学数据分析
- 支持Planck卫星观测数据加载
- 支持LiteBIRD和CMB-S4等未来实验的模拟数据
- 当astropy库不可用时，提供模拟数据作为后备
- 生成CMB功率谱比较图表

### 元胞自动机分析
- 支持Rule 30和Rule 110元胞自动机
- 位压缩窗口实现，提高计算效率
- 集成LZ78复杂度估计器

## 示例结果

### Ising模型能量演化对比
![Ising模型能量演化](out/ising/comparison_plot.png)

### CMB功率谱比较
![CMB功率谱](out/cosmology/cmb_Planck_comparison.png)

## 注意事项
- 部分功能需要可选依赖库（如qutip用于量子系统模拟）
- 大数据集处理可能需要较长计算时间，请合理设置参数
- 如需更改输出路径或其他高级设置，请修改源码中的相应部分

## 未来工作
- 添加更多物理系统的支持（如XY模型、 Heisenberg模型等）
- 优化大数据集的处理效率
- 实现更复杂的对数时间变换方法
- 增加机器学习辅助分析功能