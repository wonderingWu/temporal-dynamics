import numpy as np
import matplotlib.pyplot as plt

class Ising2D:
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
                spin = self.spins[i, j]
                neighbors = (self.spins[(i+1)%self.size, j] + 
                             self.spins[i, (j+1)%self.size] +
                             self.spins[(i-1)%self.size, j] + 
                             self.spins[i, (j-1)%self.size])
                energy += -spin * neighbors
        return energy / 2

    def metropolis_step(self):
        """Metropolis 更新一次"""
        i, j = np.random.randint(0, self.size, size=2)
        spin_old = self.spins[i, j]
        neighbors = (self.spins[(i+1)%self.size, j] +
                     self.spins[i, (j+1)%self.size] +
                     self.spins[(i-1)%self.size, j] +
                     self.spins[i, (j-1)%self.size])
        dE = 2 * spin_old * neighbors
        if dE < 0 or np.random.rand() < np.exp(-dE / self.T):
            self.spins[i, j] *= -1
            self.energy += dE
            self.magnetization += 2 * self.spins[i, j]
            return abs(dE)
        return 0.0


def run_simulation(size, T, n_steps, n_runs=5, seed=42):
    """多次重复实验，输出平均和方差"""
    np.random.seed(seed)
    magnetizations_all, tau_all = [], []
    for run in range(n_runs):
        model = Ising2D(size, T)
        mag = np.zeros(n_steps)
        tau = np.zeros(n_steps)
        cumulative_dE = 0.0
        for step in range(n_steps):
            dE_acc = model.metropolis_step()
            cumulative_dE += dE_acc
            mag[step] = abs(model.magnetization) / (size*size)
            tau[step] = cumulative_dE
        magnetizations_all.append(mag)
        tau_all.append(tau)
    return (np.mean(magnetizations_all, axis=0),
            np.std(magnetizations_all, axis=0),
            np.mean(tau_all, axis=0))


def main():
    # 参数
    sizes = [16, 32]
    Tc = 2.269
    temps = [Tc, 3.5, 1.5]
    labels = [rf"$T=T_c\approx{Tc:.3f}$", rf"$T=3.5$", rf"$T=1.5$"]
    n_steps = 10000
    n_runs = 10  # 平均样本数

    results = {}
    print("Running simulations...")
    for L in sizes:
        for temp, label in zip(temps, labels):
            print(f"  L={L}, {label}")
            mag_mean, mag_std, tau_mean = run_simulation(L, temp, n_steps, n_runs=n_runs)
            results[(L, label)] = {
                'mag_mean': mag_mean,
                'mag_std': mag_std,
                'tau_mean': tau_mean
            }

    print("Generating plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 图 (a): 外部时间 vs magnetization
    ax1 = axes[0]
    for L in sizes:
        for temp, label in zip(temps, labels):
            data = results[(L, label)]
            ax1.plot(data['mag_mean'], label=f"L={L}, {label}")
    ax1.set_xlabel("Monte Carlo steps (external time)")
    ax1.set_ylabel("|M| per site")
    ax1.set_title("(a) Relaxation in external time")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 图 (b): log(τ) vs magnetization + 阴影误差带
    ax2 = axes[1]
    for L in sizes:
        for temp, label in zip(temps, labels):
            data = results[(L, label)]
            valid = data['tau_mean'] > 1
            log_tau = np.log(data['tau_mean'][valid])
            mag = data['mag_mean'][valid]
            mag_err = data['mag_std'][valid]
            ax2.plot(log_tau, mag, lw=2, label=f"L={L}, {label}")
            ax2.fill_between(log_tau, mag-mag_err, mag+mag_err, alpha=0.2)
    ax2.set_xlabel(r"$\ln(\tau_{\rm proxy})$")
    ax2.set_ylabel("|M| per site")
    ax2.set_title("(b) Relaxation in log-time (universality test)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.suptitle("2D Ising relaxation test\n"
                 "Purpose: Check universality of log-time scaling across sizes and temperatures",
                 fontsize=14, y=1.05)
    plt.savefig("ising_universality_test.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("Figure saved as ising_universality_test.png")

if __name__ == "__main__":
    main()
