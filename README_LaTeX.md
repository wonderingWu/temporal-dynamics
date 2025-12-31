# Temporal Dynamics LaTeX Document

## 文档概述
本项目包含关于复杂系统时间动力学的LaTeX学术论文，重点研究临界性、混沌和计算系统中的结构化状态传播机制。

## 编译要求
- **LaTeX引擎**: pdflatex
- **参考文献**: bibtex
- **图形包**: graphicx (用于插入图表)

## 图表文件
所有图表文件应放置在 `./figures/` 目录下，支持格式：
- .pdf (推荐)
- .png
- .jpg
- .eps

## 编译命令
```bash
pdflatex temporalDynamics.tex
bibtex temporalDynamics
pdflatex temporalDynamics.tex  
pdflatex temporalDynamics.tex
```

## 文档结构
- **摘要**: 研究背景和主要发现
- **方法**: 信息论测量方法(NMI, TE, AIS)
- **结果**: 
  - 2D Ising模型临界性分析
  - Echo State Networks边缘混沌
  - Logistic Map混沌动力学
  - BTW沙堆模型对比
- **讨论**: 统一框架和结论

## 主要改进
1. 添加了完整的图表引用和详细结果描述
2. 清理了重复内容，结构更加清晰
3. 补充了作者信息和参考文献部分
4. 增强了内容的学术严谨性