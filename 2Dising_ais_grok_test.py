#!/usr/bin/env python3
"""
2D Ising模型AIS - 快速测试版本
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from math import log2
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

def ising_metropolis(L, T, num_sweeps=1000, burn_in=200):
    """简化的Ising模型"""
    N = L * L
    lattice = np.random.choice([-1, 1], size=(L, L))
    exponentials = {dE: np.exp(-dE / T) for dE in [-8, -4, 0, 4, 8]}
    
    magnetization_series = []
    for sweep in range(num_sweeps + burn_in):
        for _ in range(N):
            i, j = np.random.randint(0, L, 2)
            s = lattice[i, j]
            nb = (lattice[(i+1)%L, j] + lattice[(i-1)%L, j] + 
                  lattice[i, (j+1)%L] + lattice[i, (j-1)%L])
            dE = 2 * s * nb
            if dE <= 0 or np.random.rand() < exponentials[dE]:
                lattice[i, j] *= -1
        
        if sweep >= burn_in:
            magnetization_series.append(np.abs(np.mean(lattice)))
    return np.array(magnetization_series)

def calculate_ais(series, tau=5, bins=8):
    """修复后的AIS计算"""
    try:
        S = pd.qcut(series, bins, labels=False, duplicates='drop')
        if S is None:
            S = pd.cut(series, bins, labels=False, include_lowest=True)
            if S is None:
                S = pd.cut(series, bins, labels=False)
        S = S.fillna(0).astype(int)
    except Exception as e:
        percentiles = np.linspace(0, 100, bins + 1)
        boundaries = np.percentile(series, percentiles)
        S = np.digitize(series, boundaries) - 1
        S = np.clip(S, 0, bins - 1)
    
    if len(S) < 2*tau + 10: return 0.0
    
    analysis_len = len(S) - 2*tau - 1
    X_t = S[tau+1:tau+1+analysis_len]
    X_t1 = S[tau:tau+analysis_len]
    X_t_tau = S[1:1+analysis_len]
    
    def joint_entropy(*arrays):
        data = np.column_stack(arrays)
        unique, counts = np.unique(data, axis=0, return_counts=True)
        ps = counts / len(data)
        return -np.sum(ps[ps > 0] * np.log2(ps[ps > 0]))
    
    H_xt_x_t1 = joint_entropy(X_t, X_t1) - joint_entropy(X_t1)
    H_xt_x_t1_x_t_tau = joint_entropy(X_t, X_t1, X_t_tau) - joint_entropy(X_t1, X_t_tau)
    ais = H_xt_x_t1 - H_xt_x_t1_x_t_tau
    return max(0.0, ais)

# 快速测试
print("🧪 2D Ising AIS 快速测试")
temps = [2.0, 2.2, 2.269, 2.4, 2.6]
for T in temps:
    print(f"温度 T={T}: ", end="")
    try:
        series = ising_metropolis(32, T, num_sweeps=500, burn_in=100)
        ais = calculate_ais(series)
        print(f"AIS={ais:.3f}")
    except Exception as e:
        print(f"错误: {e}")

print("✅ 2D Ising AIS 测试完成！")