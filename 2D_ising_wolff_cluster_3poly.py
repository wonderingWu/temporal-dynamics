import numpy as np
import matplotlib.pyplot as plt

class Ising2D_Wolff:
    def __init__(self, size, temperature):
        self.size = size
        self.T = temperature
        self.beta = 1.0 / temperature
        self.spins = np.random.choice([-1, 1], size=(size, size))
        self.energy = self.calculate_energy()
        self.magnetization = np.sum(self.spins)

    def calculate_energy(self):
        E = 0
        for i in range(self.size):
            for j in range(self.size):
                S = self.spins[i, j]
                nb = self.spins[(i+1)%self.size, j] + self.spins[i, (j+1)%self.size]
                E += -S * nb
        return E

    def wolff_step(self):
        """执行一次 Wolff 簇翻转，返回 ΔE"""
        i, j = np.random.randint(0, self.size, size=2)
        cluster_spin = self.spins[i, j]
        cluster = set([(i, j)])
        frontier = [(i, j)]
        p_add = 1 - np.exp(-2 * self.beta)

        while frontier:
            x, y = frontier.pop()
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = (x+dx) % self.size, (y+dy) % self.size
                if (nx, ny) not in cluster and self.spins[nx, ny] == cluster_spin:
                    if np.random.rand() < p_add:
                        cluster.add((nx, ny))
                        frontier.append((nx, ny))

        dE = 0
        for (x,y) in cluster:
            spin = self.spins[x,y]
            nb = self.spins[(x+1)%self.size, y] + self.spins[(x-1)%self.size, y] + \
                 self.spins[x, (y+1)%self.size] + self.spins[x, (y-1)%self.size]
            dE += 2 * spin * nb

        for (x,y) in cluster:
            self.spins[x,y] *= -1

        self.energy += dE
        self.magnetization = np.sum(self.spins)
        return abs(dE)


def run_simulation(size, T, n_steps):
    model = Ising2D_Wolff(size, T)
    magnetization = np.zeros(n_steps)
    tau_proxy = np.zeros(n_steps)

    cumulative_dE = 0.0
    for step in range(n_steps):
        dE = model.wolff_step()
        cumulative_dE += dE
        magnetization[step] = abs(model.magnetization) / (size*size)
        tau_proxy[step] = cumulative_dE
    return magnetization, tau_proxy


def multi_run(size, T, n_steps, n_runs=10):
    all_mags, all_tau = [], []
    for r in range(n_runs):
        print(f" Run {r+1}/{n_runs} at T={T}")
        mag, tau = run_simulation(size, T, n_steps)
        all_mags.append(mag)
        all_tau.append(tau)
    all_mags, all_tau = np.array(all_mags), np.array(all_tau)
    return all_mags.mean(axis=0), all_mags.std(axis=0), all_tau.mean(axis=0)


def plot_multi_T(size=32, n_steps=5000, n_runs=10):
    Tc = 2.269
    temps = [1.5, Tc, 3.5]
    labels = [r"$T<T_c$", r"$T \approx T_c$", r"$T>T_c$"]
    colors = ["blue", "red", "green"]

    plt.figure(figsize=(8,6))

    for T, label, color in zip(temps, labels, colors):
        mean_mag, std_mag, mean_tau = multi_run(size, T, n_steps, n_runs)
        valid = mean_tau > 1
        log_tau = np.log(mean_tau[valid])
        mag = mean_mag[valid]
        std = std_mag[valid]

        plt.plot(log_tau, mag, color=color, lw=2, label=label)
        plt.fill_between(log_tau, mag-std, mag+std, color=color, alpha=0.25)

    plt.xlabel(r"$\ln(\tau_{\rm proxy})$")
    plt.ylabel(r"$|M|$ / site")
    plt.title(f"2D Ising (Wolff cluster), size={size}, runs={n_runs}\nUniversality in log-time with error band")
    plt.legend()
    plt.grid(True, alpha=0.3)
    fname = f"ising2D_wolff_multiT_size{size}_runs{n_runs}.png"
    plt.savefig(fname, dpi=200)
    print(f"Figure saved as {fname}")
    plt.show()


if __name__ == "__main__":
    plot_multi_T(size=32, n_steps=5000, n_runs=10)
