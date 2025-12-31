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
        energy = 0
        for i in range(self.size):
            for j in range(self.size):
                spin = self.spins[i, j]
                neighbors = self.spins[(i+1)%self.size, j] + self.spins[i, (j+1)%self.size] + \
                            self.spins[(i-1)%self.size, j] + self.spins[i, (j-1)%self.size]
                energy += -spin * neighbors
        return energy / 2

    def metropolis_step(self):
        i, j = np.random.randint(0, self.size, size=2)
        spin_old = self.spins[i, j]
        neighbors = self.spins[(i+1)%self.size, j] + self.spins[i, (j+1)%self.size] + \
                    self.spins[(i-1)%self.size, j] + self.spins[i, (j-1)%self.size]
        dE = 2 * spin_old * neighbors
        if dE < 0 or np.random.rand() < np.exp(-dE / self.T):
            self.spins[i, j] *= -1
            self.energy += dE
            self.magnetization += 2 * self.spins[i, j]
            return dE
        return 0.0

def simulate(size=16, n_steps=15000, temps=None):
    Tc = 2.269
    if temps is None:
        temps = [Tc, 3.5, 1.5]
    labels = [rf'$T = T_c \approx {Tc:.3f}$', rf'$T = {temps[1]}$', rf'$T = {temps[2]}$']
    results = {}

    for temp, label in zip(temps, labels):
        print(f"Simulating at {label}...")
        model = Ising2D(size, temp)

        magnetizations = np.zeros(n_steps)
        tau1, tau2, tau3 = np.zeros(n_steps), np.zeros(n_steps), np.zeros(n_steps)

        c1, c2, c3 = 0.0, 0.0, 0.0
        for step in range(n_steps):
            dE = model.metropolis_step()
            if dE != 0.0:
                c1 += abs(dE)        # tau1
                c2 += dE**2          # tau2
                c3 += 1              # tau3
            tau1[step], tau2[step], tau3[step] = c1, c2, c3
            magnetizations[step] = abs(model.magnetization) / (size*size)

        results[label] = {
            'magnetization': magnetizations,
            'tau1': tau1,
            'tau2': tau2,
            'tau3': tau3
        }
    return results, labels

def plot_results(results, labels, size, n_steps):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    proxies = ['tau1', 'tau2', 'tau3']
    proxy_labels = [r'$\tau_1 = \Sigma |\Delta E|$',
                    r'$\tau_2 = \Sigma (\Delta E)^2$',
                    r'$\tau_3 = \#\text{ accepted moves}$']

    for col, (proxy, p_label) in enumerate(zip(proxies, proxy_labels)):
        # (a) External time
        ax = axes[0, col]
        for label in labels:
            ax.plot(results[label]['magnetization'], label=label)
        ax.set_title(f'(a{col+1}) External Time\nL={size}, n_steps={n_steps}')
        ax.set_xlabel('MC steps')
        ax.set_ylabel('|M| per site')
        if col == 0: ax.legend()

        # (b) Proxy time
        ax = axes[1, col]
        for label in labels:
            ax.plot(results[label][proxy], results[label]['magnetization'], label=label)
        ax.set_title(f'(b{col+1}) Relaxation vs {p_label}')
        ax.set_xlabel(p_label)
        ax.set_ylabel('|M| per site')

        # (c) Log proxy time
        ax = axes[2, col]
        for label in labels:
            valid = results[label][proxy] > 1
            ax.plot(np.log(results[label][proxy][valid]), results[label]['magnetization'][valid], label=label)
        ax.set_title(f'(c{col+1}) Log-scaling collapse test')
        ax.set_xlabel(r'$\ln($'+p_label+r'$)$')
        ax.set_ylabel('|M| per site')

    plt.tight_layout()
    fname = f"ising_multi_proxy_L{size}_steps{n_steps}.png"
    plt.savefig(fname, dpi=300)
    print(f"Figure saved as {fname}")
    plt.show()

def main():
    print("Starting 2D Ising simulation...")
    size, n_steps = 16, 15000
    print(f"Parameters: size={size}, n_steps={n_steps}")
    results, labels = simulate(size=size, n_steps=n_steps)
    print("Simulation completed, plotting results...")
    plot_results(results, labels, size, n_steps)
    print("Script execution finished.")

if __name__ == "__main__":
    main()
