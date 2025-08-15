import numpy as np
import matplotlib.pyplot as plt
import os

# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 数据准备
# 导入out/ising文件夹下csv数据，使用绝对路径
t_data_path = os.path.join(current_dir, 'out', 'ising', 'ising_t_series.csv')
# 使用utf-8编码读取文件，并跳过可能的错误行
# 跳过第一行标题行
t_data = np.loadtxt(t_data_path, delimiter=',', encoding='utf-8', skiprows=1)
print(f"成功读取ising_t_series.csv，数据形状: {t_data.shape}")
t = t_data[:, 0]  # t_data第一列
t = np.atleast_2d(t).T  # 确保是列向量
E_t = t_data[:, 1]  # t_data第二列
var_t = t_data[:, 2]  # t_data第三列

tau_data_path = os.path.join(current_dir, 'out', 'ising', 'ising_tau_series.csv')
# 使用utf-8编码读取文件，并跳过可能的错误行
# 跳过第一行标题行
tau_data = np.loadtxt(tau_data_path, delimiter=',', encoding='utf-8', skiprows=1)
print(f"成功读取ising_tau_series.csv，数据形状: {tau_data.shape}")
tau = tau_data[:, 0]  # tau_data第一列
tau = np.atleast_2d(tau).T  # 确保是列向量
E_tau = tau_data[:, 1]  # tau_data第二列
var_tau = tau_data[:, 2]  # tau_data第三列
t_tau = 200 * np.exp(tau)  # 转换为实际步数

# 从数据中计算最终能量和收敛阈值
final_energy = E_t[-1]
convergence_threshold_t = var_t[-1]  # 假设使用最后一个方差值作为收敛阈值

# 创建图表
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.subplots_adjust(hspace=0.3)

# 能量演化对比
ax1.plot(t, E_t, 'b-', label='Linear Time (t)')
ax1.plot(t_tau, E_tau, 'r--', label='Log Time (τ)')
ax1.axhline(y=final_energy, color='k', linestyle=':', alpha=0.5, label='Final Energy')
ax1.set_ylabel('Energy')
ax1.set_title('Energy Evolution Comparison')
ax1.legend()
ax1.grid(True)

# 方差演化对比
ax2.semilogy(t, var_t, 'b-', label='Var(t)')
ax2.semilogy(t_tau, var_tau, 'r--', label='Var(τ)')
ax2.axhline(y=convergence_threshold_t, color='g', linestyle='--', label='Convergence Threshold')
ax2.set_xlabel('Simulation Steps')
ax2.set_ylabel('Variance (log scale)')
ax2.set_title('Variance Evolution Comparison')
ax2.legend()
ax2.grid(True)

# 保存图表
plot_save_path = os.path.join(current_dir, 'out', 'ising', 'comparison_plot.png')
plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
plt.close()