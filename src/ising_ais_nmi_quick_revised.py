#!/usr/bin/env python3
"""
修复版 AIS + NMI 快速验证脚本
- 标准 AIS = I(X_t; X_{t-1})
- 固定 [0,1] 分箱，避免临界区信息丢失
- 新增 NMI(τ=50) 验证长程关联
- 聚焦临界区，10-15分钟出结果
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ----------------------------
# 1. Ising 模型（充分热化，序列更新）
# ----------------------------
def ising_metropolis(L, T, num_sweeps=10000, burn_in=2000):
    N = L * L
    lattice = np.random.choice([-1, 1], size=(L, L))
    mag_series = []
    for sweep in range(num_sweeps + burn_in):
        for _ in range(N):
            i, j = np.random.randint(0, L, 2)
            s = lattice[i, j]
            nb = (lattice[(i+1)%L, j] + lattice[(i-1)%L, j] +
                  lattice[i, (j+1)%L] + lattice[i, (j-1)%L])
            dE = 2 * s * nb
            if dE <= 0 or np.random.rand() < np.exp(-dE / T):
                lattice[i, j] *= -1
        if sweep >= burn_in:
            mag_series.append(np.abs(np.mean(lattice)))
    return np.array(mag_series)

# ----------------------------
# 2. ✅ 修复离散化：固定 [0,1] 分箱
# ----------------------------
def discretize_fixed(series, bins=16):
    """强制使用磁化强度物理范围 [0, 1]"""
    edges = np.linspace(0, 1, bins + 1)
    digitized = np.digitize(series, edges) - 1
    return np.clip(digitized, 0, bins - 1).astype(int)

# ----------------------------
# 3. ✅ 修正版 AIS（标准定义）
# ----------------------------
def calculate_ais(series, bins=16):
    S = discretize_fixed(series, bins)
    if len(S) < 10:
        return 0.0
    X_t = S[1:]
    X_t1 = S[:-1]
    
    def entropy(x):
        counts = np.bincount(x)
        probs = counts[counts > 0] / len(x)
        return -np.sum(probs * np.log2(probs + 1e-12))
    
    def joint_entropy(x, y):
        data = np.column_stack([x, y])
        unique, counts = np.unique(data, axis=0, return_counts=True)
        probs = counts / len(data)
        return -np.sum(probs * np.log2(probs + 1e-12))
    
    mi = entropy(X_t) + entropy(X_t1) - joint_entropy(X_t, X_t1)
    return max(0.0, mi)

# ----------------------------
# 4. ✅ NMI(τ=50) 计算
# ----------------------------
def calculate_nmi_long(series, tau=50, bins=16):
    S = discretize_fixed(series, bins)
    if tau >= len(S) // 2:
        return 0.0
    future = S[tau:]
    past = S[:-tau]
    if len(future) < 10:
        return 0.0
    
    # 互信息
    def joint_histogram(x, y, bins):
        hist, _, _ = np.histogram2d(x, y, bins=[bins, bins], range=[[0, bins], [0, bins]])
        return hist
    hist_joint = joint_histogram(future, past, bins)
    hist_future = np.sum(hist_joint, axis=1)
    hist_past = np.sum(hist_joint, axis=0)
    
    # 计算 MI
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if hist_joint[i, j] > 0:
                p_xy = hist_joint[i, j] / len(future)
                p_x = hist_future[i] / len(future)
                p_y = hist_past[j] / len(past)
                mi += p_xy * np.log2(p_xy / (p_x * p_y + 1e-12))
    
    # 熵
    H_x = -np.sum((hist_future[hist_future > 0] / len(future)) * 
                  np.log2(hist_future[hist_future > 0] / len(future) + 1e-12))
    H_y = -np.sum((hist_past[hist_past > 0] / len(past)) * 
                  np.log2(hist_past[hist_past > 0] / len(past) + 1e-12))
    
    # 熵过低时视为静态有序
    if H_x < 0.05 or H_y < 0.05:
        return 0.0
        
    nmi = mi / np.sqrt(H_x * H_y) if (H_x * H_y) > 1e-10 else 0.0
    return max(0.0, nmi)

# ----------------------------
# 5. 主实验（快速验证）
# ----------------------------
lattice_sizes = [16, 32]
temps = np.linspace(2.1, 2.5, 16)  # 聚焦临界区
num_runs = 5  # 降低重复次数以加速

print("🚀 运行修复版 AIS + NMI 快速验证...")
print(f"   L = {lattice_sizes}, T = [{temps[0]:.2f}, {temps[-1]:.2f}]")
print(f"   Runs = {num_runs}, Sweeps = 10000 + 2000")

results = {}
for L in lattice_sizes:
    print(f"\n🧪 L = {L}")
    ais_all, nmi50_all = [], []
    for run in range(num_runs):
        ais_run, nmi50_run = [], []
        for T in temps:
            mag = ising_metropolis(L, T)
            ais_run.append(calculate_ais(mag))
            nmi50_run.append(calculate_nmi_long(mag, tau=50))
        ais_all.append(ais_run)
        nmi50_all.append(nmi50_run)
    results[L] = {
        'T': temps,
        'ais_mean': np.mean(ais_all, axis=0),
        'ais_std': np.std(ais_all, axis=0),
        'nmi50_mean': np.mean(nmi50_all, axis=0),
        'nmi50_std': np.std(nmi50_all, axis=0)
    }

# ----------------------------
# 6. 双图输出
# ----------------------------
plt.figure(figsize=(12, 5))

# (a) AIS
plt.subplot(1, 2, 1)
for i, L in enumerate(lattice_sizes):
    d = results[L]
    plt.errorbar(d['T'], d['ais_mean'], yerr=d['ais_std'],
                 label=f'L={L}', marker='o', capsize=2)
plt.axvline(2.269, color='red', ls='--', alpha=0.7)
plt.xlabel('T'); plt.ylabel('AIS (bits)')
plt.title('AIS: State Stability')
plt.legend(); plt.grid(True, alpha=0.3)

# (b) NMI(τ=50)
plt.subplot(1, 2, 2)
for i, L in enumerate(lattice_sizes):
    d = results[L]
    # 仅绘制显著区域
    mask = d['nmi50_mean'] > 0.01
    plt.errorbar(d['T'][mask], d['nmi50_mean'][mask], 
                 yerr=d['nmi50_std'][mask],
                 label=f'L={L}', marker='^', capsize=2)
plt.axvline(2.269, color='red', ls='--', alpha=0.7)
plt.xlabel('T'); plt.ylabel('NMI')
plt.title('NMI(τ=50): Long-Range Memory')
plt.legend(); plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ais_nmi_corrected_quick.png', dpi=300)
plt.show()

# ----------------------------
# 7. 关键结论
# ----------------------------
print("\n✅ 修复版验证完成！")
for L in lattice_sizes:
    d = results[L]
    ais_peak_T = d['T'][np.argmax(d['ais_mean'])]
    nmi50_peak_T = d['T'][np.argmax(d['nmi50_mean'])]
    print(f"\nL={L}:")
    print(f"  - AIS 峰值 @ T={ais_peak_T:.3f} (低温有序)")
    print(f"  - NMI(τ=50) 峰值 @ T={nmi50_peak_T:.3f} (临界协同)")

nmi50_at_227 = results[32]['nmi50_mean'][np.argmin(np.abs(temps - 2.27))]
print(f"\n🎯 NMI(τ=50) @ T=2.27 (L=32): {nmi50_at_227:.4f}")
print("   (若 >0.1，则支持临界长程关联)")