import numpy as np
import matplotlib.pyplot as plt

class Ising2D_Wolff:
    def __init__(self, size, temperature):
        self.size = size
        self.T = temperature
        self.spins = np.random.choice([-1, 1], size=(size, size))
        self.energy = self.calculate_energy()
        self.magnetization = np.sum(self.spins)

    def calculate_energy(self):
        """计算总能量"""
        energy = 0
        for i in range(self.size):
            for j in range(self.size):
                S = self.spins[i, j]
                nb = self.spins[(i+1)%self.size, j] + self.spins[i, (j+1)%self.size] + \
                     self.spins[(i-1)%self.size, j] + self.spins[i, (j-1)%self.size]
                energy += -S * nb
        return energy/2

    def wolff_step(self):
        """执行一次 Wolff cluster 更新，并返回 ΔE 作为 τ proxy 成分"""
        L = self.size
        # 随机选择种子
        i, j = np.random.randint(0, L), np.random.randint(0, L)
        cluster_spin = self.spins[i, j]
        p_add = 1 - np.exp(-2.0 / self.T)

        # BFS/DFS 扩展簇
        cluster = [(i, j)]
        to_check = [(i, j)]
        in_cluster = np.zeros((L, L), dtype=bool)
        in_cluster[i, j] = True

        while to_check:
            x, y = to_check.pop()
            neighbors = [((x+1)%L, y), ((x-1)%L, y),
                         (x, (y+1)%L), (x, (y-1)%L)]
            for nx, ny in neighbors:
                if not in_cluster[nx, ny] and self.spins[nx, ny] == cluster_spin:
                    if np.random.rand() < p_add:
                        in_cluster[nx, ny] = True
                        to_check.append((nx, ny))
                        cluster.append((nx, ny))

        # 翻转簇
        old_energy = self.energy
        for (x, y) in cluster:
            self.spins[x, y] *= -1
        self.energy = self.calculate_energy()
        self.magnetization = np.sum(self.spins)

        dE = self.energy - old_energy
        return abs(dE), len(cluster)   # 返回能量变化和簇大小

# ---------------------- 模拟参数 ----------------------
L = 32
Tc = 2.269
temps = [Tc, 1.5, 3.5]  # 临界、低温、高温
labels = [rf"$T \approx T_c$", rf"$T = 1.5$", rf"$T = 3.5$"]
n_steps = 5000

results = {}

for temp, label in zip(temps, labels):
    print(f"Simulating Wolff at {label}...")
    model = Ising2D_Wolff(L, temp)

    magnetizations = np.zeros(n_steps)
    tau_proxy = np.zeros(n_steps)
    cumulative_dE = 0.0

    for step in range(n_steps):
        dE, cl_size = model.wolff_step()
        cumulative_dE += dE
        magnetizations[step] = abs(model.magnetization) / (L*L)
        tau_proxy[step] = cumulative_dE

    results[label] = {
        'magnetization': magnetizations,
        'tau_proxy': tau_proxy
    }

# ---------------------- 绘图 ----------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# (a) Magnetization vs steps
ax1 = axes[0, 0]
for label, data in results.items():
    ax1.plot(data['magnetization'], label=label, alpha=0.8)
ax1.set_xlabel("Cluster Updates (External Time)")
ax1.set_ylabel(r"$|M|$ / site")
ax1.set_title("(a) Relaxation in External Time")
ax1.legend(); ax1.grid(True, alpha=0.3)

# (b) vs τ proxy
ax2 = axes[0, 1]
for label, data in results.items():
    ax2.plot(data['tau_proxy'], data['magnetization'], label=label, alpha=0.8)
ax2.set_xlabel(r"Cumulative Energy Cost $\tau_{\rm proxy}$")
ax2.set_ylabel(r"$|M|$ / site")
ax2.set_title("(b) Relaxation vs Physical Cost τ")
ax2.legend(); ax2.grid(True, alpha=0.3)

# (c) vs log τ proxy
ax3 = axes[1, 0]
for label, data in results.items():
    valid = data['tau_proxy'] > 1
    ax3.plot(np.log(data['tau_proxy'][valid]), data['magnetization'][valid], label=label, lw=2)
ax3.set_xlabel(r"$\ln(\tau_{\rm proxy})$")
ax3.set_ylabel(r"$|M|$ / site")
ax3.set_title("(c) Emergent Universality in Log-Time")
ax3.legend(); ax3.grid(True, alpha=0.3)

# (d) Critical scaling check
ax4 = axes[1, 1]
crit = results[labels[0]]
valid = crit['tau_proxy'] > 1
log_tau = np.log(crit['tau_proxy'][valid])
mag = crit['magnetization'][valid]
# 添加一个小的偏移量来避免log(0)的问题
epsilon = 1e-12
mag_positive = mag + epsilon
fit_slice = slice(100, 1000)
z = np.polyfit(log_tau[fit_slice], np.log(mag_positive[fit_slice]), 1)
p = np.poly1d(z)
ax4.plot(log_tau, np.log(mag_positive), 'o', markersize=2, alpha=0.5, label="Critical Data")
ax4.plot(log_tau[fit_slice], p(log_tau[fit_slice]), 'r-', lw=2, 
         label=f"Linear Fit (Slope={z[0]:.3f})")
ax4.set_xlabel(r"$\ln(\tau_{\rm proxy})$")
ax4.set_ylabel(r"$\ln(|M|)$")
ax4.set_title("(d) Critical Decay in Log-Time (Wolff)")
ax4.legend(); ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"2D_ising_Wolff_L{L}.png", dpi=300)
plt.show()
