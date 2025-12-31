import numpy as np
import matplotlib.pyplot as plt
# 修复了导入错误 - sklearn.neighbors 中没有 MutualInformationEstimator
# 实际代码中使用的是 sklearn.feature_selection.mutual_info_regression

def logistic_map(r, n_steps=10000, x0=0.5):
    x = np.zeros(n_steps)
    x[0] = x0
    for i in range(1, n_steps):
        x[i] = r * x[i-1] * (1 - x[i-1])
    return x[1000:] # 去除瞬态

def calc_ais_continuous(series, k=1):
    # AIS(X) = I(X_t; X_{t-1, ..., t-k})
    # 使用简单的连续变量互信息估计
    n = len(series)
    X_past = series[:-1].reshape(-1, 1) # t-1
    X_future = series[1:]               # t
    
    # 使用 sklearn 的 k-NN 互信息估计
    # 注意：这只是近似，严谨需用 JIDT 的 Kraskov 估计器
    from sklearn.feature_selection import mutual_info_regression
    mi = mutual_info_regression(X_past, X_future, discrete_features=False, n_neighbors=3)
    return mi[0]

# --- 实验 ---
r_values = np.linspace(3.5, 4.0, 100)
ais_values = []
lyapunov_values = []

print("Scanning Logistic Map...")
for r in r_values:
    ts = logistic_map(r)
    
    # 1. 计算 AIS (信息存储)
    ais = calc_ais_continuous(ts)
    ais_values.append(ais)
    
    # 2. 计算 Lyapunov 指数 (混沌程度)
    # lambda = mean(ln|f'(x)|) = mean(ln|r(1-2x)|)
    lyap = np.mean(np.log(np.abs(r * (1 - 2 * ts))))
    lyapunov_values.append(lyap)

# --- 绘图 ---
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Parameter r')
ax1.set_ylabel('Active Info Storage (AIS)', color=color)
ax1.plot(r_values, ais_values, color=color, label='AIS (Memory)')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Lyapunov Exponent', color=color)
ax2.plot(r_values, lyapunov_values, color=color, linestyle='--', label='Lyapunov (Chaos)', alpha=0.5)
ax2.axhline(0, color='grey', linewidth=0.5)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Falsification Test: Does Chaos Destroy Memory?')
plt.show()
plt.savefig('logistic_map_test.png')
print("Done!")