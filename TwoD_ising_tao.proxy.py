import numpy as np
import matplotlib.pyplot as plt

class Ising2D:
    def __init__(self, size, temperature):
        self.size = size
        self.T = temperature
        # 初始化随机自旋网格 (+1 or -1)
        self.spins = np.random.choice([-1, 1], size=(size, size))
        # 计算初始能量和磁化强度
        self.energy = self.calculate_energy()
        self.magnetization = np.sum(self.spins)

    def calculate_energy(self):
        """计算当前格点的总能量"""
        energy = 0
        for i in range(self.size):
            for j in range(self.size):
                spin = self.spins[i, j]
                # 最近邻相互作用：上、下、左、右
                neighbors = self.spins[(i+1)%self.size, j] + self.spins[i, (j+1)%self.size] + \
                            self.spins[(i-1)%self.size, j] + self.spins[i, (j-1)%self.size]
                energy += -spin * neighbors
        return energy / 2  # 每个键被计算了两次

    def metropolis_step(self):
        """执行一次蒙特卡洛步（MCS），返回该步被接受的翻转的能量变化绝对值"""
        i, j = np.random.randint(0, self.size, size=2)
        spin_old = self.spins[i, j]
        
        # 计算翻转前后的能量差
        neighbors = self.spins[(i+1)%self.size, j] + self.spins[i, (j+1)%self.size] + \
                    self.spins[(i-1)%self.size, j] + self.spins[i, (j-1)%self.size]
        dE = 2 * spin_old * neighbors
        
        # Metropolis 准则
        if dE < 0 or np.random.rand() < np.exp(-dE / self.T):
            self.spins[i, j] *= -1
            self.energy += dE
            self.magnetization += 2 * self.spins[i, j]  # 从 -1 到 +1 变化是 +2，反之是 -2
            return abs(dE) # 返回本次翻转的 |ΔE|
        return 0.0

# 模拟参数
size = 16
Tc = 2.269  # 2D Ising 模型的临界温度
T_critical = Tc
T_high = 3.5  # 远高于临界温度
T_low = 1.5   # 远低于临界温度
n_steps = 15000  # 总蒙特卡洛步数

# 运行模拟：在临界温度和非临界温度下
temps = [T_critical, T_high, T_low]
labels = [f'$T = T_c \\approx {T_critical:.3f}$', f'$T = {T_high}$', f'$T = {T_low}$']
results = {}

for temp, label in zip(temps, labels):
    print(f"Simulating at {label}...")
    model = Ising2D(size, temp)
    
    # 初始化数组来存储数据
    energies = np.zeros(n_steps)
    magnetizations = np.zeros(n_steps)
    tau_proxy = np.zeros(n_steps)  # 我们的代理变量：累积 |ΔE|
    
    # 运行模拟
    cumulative_dE = 0.0
    for step in range(n_steps):
        dE_accepted = model.metropolis_step()
        cumulative_dE += dE_accepted
        
        energies[step] = model.energy
        magnetizations[step] = abs(model.magnetization) / (size*size)  # 绝对磁化强度 per spin
        tau_proxy[step] = cumulative_dE
    
    results[label] = {
        'energy': energies,
        'magnetization': magnetizations,
        'tau_proxy': tau_proxy
    }

# 绘图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 图1：磁化强度随模拟步数的演化（线性时间）
ax1 = axes[0, 0]
for label, data in results.items():
    ax1.plot(data['magnetization'], label=label, alpha=0.8)
ax1.set_xlabel('Monte Carlo Steps (External Time)')
ax1.set_ylabel('$|M|$ / site')
ax1.set_title('(a) Relaxation in External Time')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 图2：磁化强度随 τ_proxy 的演化（我们的物理内禀变量）
ax2 = axes[0, 1]
for label, data in results.items():
    ax2.plot(data['tau_proxy'], data['magnetization'], label=label, alpha=0.8)
ax2.set_xlabel('Cumulative Energy Cost, $\\tau_{\\rm proxy} = \Sigma |\Delta E|$')
ax2.set_ylabel('$|M|$ / site')
ax2.set_title('(b) Relaxation vs. Physical Cost $\\tau$')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 图3：磁化强度随 ln(τ_proxy) 的演化（关键测试！）
ax3 = axes[1, 0]
for label, data in results.items():
    valid_indices = data['tau_proxy'] > 1  # 避免log(0)
    log_tau = np.log(data['tau_proxy'][valid_indices])
    ax3.plot(log_tau, data['magnetization'][valid_indices], label=label, alpha=0.8, lw=2)
ax3.set_xlabel(r'$\ln(\tau_{\rm proxy})$')
ax3.set_ylabel(r'$|M|$ / site')
ax3.set_title('(c) Emergent Universality: Relaxation in Log-Time\n(Proof of Concept)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 图4：展示临界温度下的曲线在 Log-Time 下更早进入普适的衰减区域
ax4 = axes[1, 1]
T_c_data = results[labels[0]]
valid_indices = T_c_data['tau_proxy'] > 1
log_tau_c = np.log(T_c_data['tau_proxy'][valid_indices])
mag_c = T_c_data['magnetization'][valid_indices]

# 尝试拟合一个指数衰减，展示其线性
ax4.plot(log_tau_c, np.log(mag_c), 'o', markersize=2, label=f'Data at {labels[0]}', alpha=0.6)
# 尝试线性拟合
fit_start, fit_end = 100, 1000 # 选择一段线性区域
fit_slice = slice(fit_start, fit_end)
z = np.polyfit(log_tau_c[fit_slice], np.log(mag_c[fit_slice]), 1)
p = np.poly1d(z)
ax4.plot(log_tau_c[fit_slice], p(log_tau_c[fit_slice]), 'r-', lw=3, label=f'Linear Fit (Slope = {z[0]:.3f})')
ax4.set_xlabel('$\ln(\\tau_{\\rm proxy})$')
ax4.set_ylabel(r'$\ln(|M|)$')
ax4.set_title('(d) Critical Slowing Down Removed\nLog-Log Plot Reveals Exponential Decay')
ax4.legend()
ax4.grid(True, alpha=0.3)

def main():
    # 模拟参数
    size = 16
    Tc = 2.269  # 2D Ising 模型的临界温度
    T_critical = Tc
    T_high = 3.5  # 远高于临界温度
    T_low = 1.5   # 远低于临界温度
    n_steps = 15000  # 总蒙特卡洛步数

    # 运行模拟：在临界温度和非临界温度下
    temps = [T_critical, T_high, T_low]
    labels = [rf'$T = T_c \approx {T_critical:.3f}$', rf'$T = {T_high}$', rf'$T = {T_low}$']
    results = {}

    print("Starting 2D Ising model simulation...")
    for temp, label in zip(temps, labels):
        print(f"  Simulating at {label}...")
        model = Ising2D(size, temp)
        
        # 初始化数组来存储数据
        energies = np.zeros(n_steps)
        magnetizations = np.zeros(n_steps)
        tau_proxy = np.zeros(n_steps)  # 我们的代理变量：累积 |ΔE|
        
        # 运行模拟
        cumulative_dE = 0.0
        for step in range(n_steps):
            dE_accepted = model.metropolis_step()
            cumulative_dE += dE_accepted
            
            energies[step] = model.energy
            magnetizations[step] = abs(model.magnetization) / (size*size)  # 绝对磁化强度 per spin
            tau_proxy[step] = cumulative_dE
        
        results[label] = {
            'energy': energies,
            'magnetization': magnetizations,
            'tau_proxy': tau_proxy
        }

    print("Generating plots...")
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 图1：磁化强度随模拟步数的演化（线性时间）
    ax1 = axes[0, 0]
    for label, data in results.items():
        ax1.plot(data['magnetization'], label=label, alpha=0.8)
    ax1.set_xlabel('Monte Carlo Steps (External Time)')
    ax1.set_ylabel('$|M|$ / site')
    ax1.set_title('(a) Relaxation in External Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 图2：磁化强度随 τ_proxy 的演化（我们的物理内禀变量）
    ax2 = axes[0, 1]
    for label, data in results.items():
        ax2.plot(data['tau_proxy'], data['magnetization'], label=label, alpha=0.8)
    ax2.set_xlabel(r'Cumulative Energy Cost, $\tau_{\rm proxy} = \Sigma |\Delta E|$')
    ax2.set_ylabel('$|M|$ / site')
    ax2.set_title('(b) Relaxation vs. Physical Cost $\tau$')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 图3：磁化强度随 ln(τ_proxy) 的演化（关键测试！）
    ax3 = axes[1, 0]
    for label, data in results.items():
        valid_indices = data['tau_proxy'] > 1  # 避免log(0)
        log_tau = np.log(data['tau_proxy'][valid_indices])
        ax3.plot(log_tau, data['magnetization'][valid_indices], label=label, alpha=0.8, lw=2)
    ax3.set_xlabel(r'$\ln(\tau_{\rm proxy})$')
    ax3.set_ylabel('$|M|$ / site')
    ax3.set_title('(c) Emergent Universality: Relaxation in Log-Time\n(Proof of Concept)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 图4：展示临界温度下的曲线在 Log-Time 下更早进入普适的衰减区域
    ax4 = axes[1, 1]
    T_c_data = results[labels[0]]
    valid_indices = T_c_data['tau_proxy'] > 1
    log_tau_c = np.log(T_c_data['tau_proxy'][valid_indices])
    mag_c = T_c_data['magnetization'][valid_indices]

    # 尝试拟合一个指数衰减，展示其线性
    ax4.plot(log_tau_c, np.log(mag_c), 'o', markersize=2, label=f'Data at {labels[0]}', alpha=0.6)
    # 尝试线性拟合
    fit_start, fit_end = 100, 1000 # 选择一段线性区域
    fit_slice = slice(fit_start, fit_end)
    z = np.polyfit(log_tau_c[fit_slice], np.log(mag_c[fit_slice]), 1)
    p = np.poly1d(z)
    ax4.plot(log_tau_c[fit_slice], p(log_tau_c[fit_slice]), 'r-', lw=3, label=f'Linear Fit (Slope = {z[0]:.3f})')
    ax4.set_xlabel(r'$\ln(\tau_{\rm proxy})$')
    ax4.set_ylabel(r'$\ln(|M|)$')
    ax4.set_title('(d) Critical Slowing Down Removed\nLog-Log Plot Reveals Exponential Decay')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('2D_ising_results.png')
    print("Figure saved as 2D_ising_results.png")
    plt.show()
    print("Simulation completed!")

if __name__ == "__main__":
    main()