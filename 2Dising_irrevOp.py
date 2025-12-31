import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Ising2D:
    def __init__(self, size, temperature):
        self.size = size
        self.T = temperature
        self.spins = np.random.choice([-1, 1], size=(size, size))
        self.energy = self.calculate_energy()
        self.magnetization = np.sum(self.spins)

    def calculate_energy(self):
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


def fit_slope(log_tau, mag, fit_start=200, fit_end=2000):
    """在 log-log 空间做线性拟合，返回 slope ± error"""
    valid = (log_tau > 0) & (mag > 0)
    log_tau = log_tau[valid]
    log_mag = np.log(mag[valid])

    if len(log_tau) < fit_end:
        fit_end = len(log_tau)

    x_fit = log_tau[fit_start:fit_end]
    y_fit = log_mag[fit_start:fit_end]

    coeffs, cov = np.polyfit(x_fit, y_fit, 1, cov=True)
    slope, intercept = coeffs
    slope_err = np.sqrt(cov[0, 0])
    return slope, slope_err, (x_fit, y_fit, coeffs)


def main():
    # 参数
    sizes = [16, 32, 64]
    Tc = 2.269
    temps = [Tc, 3.5, 1.5]
    labels = [rf"$T=T_c\approx{Tc:.3f}$", rf"$T=3.5$", rf"$T=1.5$"]
    n_steps = 10000
    n_runs = 10

    results = {}
    slope_results = []

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

            # 在临界温度下做 slope 拟合
            if np.isclose(temp, Tc, atol=1e-3):
                valid = tau_mean > 1
                log_tau = np.log(tau_mean[valid])
                mag = mag_mean[valid]
                slope, slope_err, _ = fit_slope(log_tau, mag)
                slope_results.append({
                    "System Size L": L,
                    "Temperature": temp,
                    "Slope": slope,
                    "Slope Error": slope_err
                })
                print(f"  >> Fit result L={L}: slope={slope:.3f} ± {slope_err:.3f}")

    # 保存斜率结果到 CSV
    df = pd.DataFrame(slope_results)
    df.to_csv("ising_slope_results.csv", index=False)
    print("Saved slope results to ising_slope_results.csv")

    # 绘制 log-time scaling 图
    fig, ax = plt.subplots(figsize=(8, 6))
    for L in sizes:
        for temp, label in zip(temps, labels):
            data = results[(L, label)]
            valid = data['tau_mean'] > 1
            log_tau = np.log(data['tau_mean'][valid])
            mag = data['mag_mean'][valid]
            mag_err = data['mag_std'][valid]
            ax.plot(log_tau, mag, lw=2, label=f"L={L}, {label}")
            ax.fill_between(log_tau, mag-mag_err, mag+mag_err, alpha=0.2)
    ax.set_xlabel(r"$\ln(\tau_{\rm proxy})$")
    ax.set_ylabel("|M| per site")
    ax.set_title("2D Ising relaxation in log-time\nUniversality test across L and T")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("ising_universality_fit.png", dpi=300)
    plt.show()
    print("Figure saved as ising_universality_fit.png")

if __name__ == "__main__":
    main()
