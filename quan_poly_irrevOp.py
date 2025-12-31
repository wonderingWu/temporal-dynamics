import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import kron, identity, csr_matrix
from scipy.sparse.linalg import expm_multiply
from scipy.optimize import curve_fit

# 基本泡利矩阵
sigma_x = csr_matrix([[0, 1], [1, 0]])
sigma_y = csr_matrix([[0, -1j], [1j, 0]])
sigma_z = csr_matrix([[1, 0], [0, -1]])
id2 = identity(2)


def build_hamiltonian(n_qubits, seed=None):
    """无序海森堡模型 Hamiltonian"""
    dim = 2 ** n_qubits
    H = csr_matrix((dim, dim), dtype=complex)
    rng = np.random.default_rng(seed)
    h_fields = rng.uniform(-1, 1, n_qubits)

    for i in range(n_qubits):
        # 外场项
        h_op = 1
        for j in range(n_qubits):
            h_op = kron(h_op, sigma_z if j == i else id2)
        H += h_fields[i] * h_op

        # 相邻 zz 相互作用
        if i < n_qubits - 1:
            Jzz_op = 1
            for j in range(n_qubits):
                if j == i or j == i + 1:
                    Jzz_op = kron(Jzz_op, sigma_z)
                else:
                    Jzz_op = kron(Jzz_op, id2)
            H += 0.5 * Jzz_op
    return H


def single_qubit_entropy(state, n_qubits, target=0):
    """计算目标 qubit 的冯诺依曼熵"""
    dim = 2 ** n_qubits
    rho = np.outer(state, state.conj())  # 密度矩阵
    # 重整化为 (2, 2^(n-1)) 形状
    psi = state.reshape((2, -1))
    rhoA = np.dot(psi, psi.conj().T)  # reduced density for 1 qubit
    eigvals = np.linalg.eigvalsh(rhoA)
    eigvals = eigvals[np.where(eigvals > 1e-12)]
    return -np.sum(eigvals * np.log(eigvals))


def simulate(n_qubits=6, n_steps=200, dt=0.1, n_realizations=10):
    """多次随机实例模拟"""
    dim = 2 ** n_qubits
    times = np.arange(0, n_steps * dt, dt)
    entropy_runs = []

    for run in range(n_realizations):
        H = build_hamiltonian(n_qubits, seed=run)
        psi = np.ones(dim) / np.sqrt(dim)  # 均匀叠加态
        cumulative_entropy = []
        total_S = 0.0

        for t in times:
            psi = expm_multiply(-1j * H * dt, psi)
            S = single_qubit_entropy(psi, n_qubits, target=0)
            total_S += S
            cumulative_entropy.append(total_S)
        entropy_runs.append(cumulative_entropy)

    entropy_runs = np.array(entropy_runs)
    mean_entropy = np.mean(entropy_runs, axis=0)
    std_entropy = np.std(entropy_runs, axis=0)
    return times, mean_entropy, std_entropy


def analyze_and_plot(times, tau_proxy, tau_std):
    """拟合并绘图"""
    start = len(times) // 5  # 去掉前段噪声
    t_data = times[start:]
    tau_data = tau_proxy[start:]
    tau_err = tau_std[start:]

    # 对数拟合 t ~ A ln(tau) + B
    def log_func(tau, A, B):
        return A * np.log(tau) + B

    params, cov = curve_fit(log_func, tau_data, t_data, p0=[1, 0])
    A_fit, B_fit = params
    perr = np.sqrt(np.diag(cov))

    tau_fit = np.linspace(min(tau_data), max(tau_data), 100)
    t_fit = log_func(tau_fit, A_fit, B_fit)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (a) t vs ln(tau)
    ax1 = axes[0]
    ax1.errorbar(np.log(tau_data), t_data, yerr=None, xerr=tau_err/tau_data,
                 fmt='bo', markersize=4, alpha=0.6, label='Simulation (mean ± std)')
    ax1.plot(np.log(tau_fit), t_fit, 'r-', lw=2,
             label=fr'Fit: $t = {A_fit:.2f}\ln\tau {B_fit:+.2f}$')
    ax1.set_xlabel(r'$\ln(\tau_{\rm proxy})$')
    ax1.set_ylabel('External Time $t$')
    ax1.set_title('(a) $t$ vs. $\ln(\tau)$ (averaged)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # (b) Emergent vs External time
    t_emergent = log_func(tau_data, A_fit, B_fit)
    ax2 = axes[1]
    ax2.plot(t_data, t_emergent, 'g-', label='Emergent time')
    ax2.plot([min(t_data), max(t_data)], [min(t_data), max(t_data)], 'k--', label='Ideal $t$')
    ax2.set_xlabel('External Time $t$')
    ax2.set_ylabel('Emergent Time $k\ln\tau$')
    ax2.set_title('(b) Validation: Emergent vs External')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("quan_poly_tau_proxy_avg.png", dpi=300)
    plt.show()

    print(f"Fit results: A={A_fit:.3f} ± {perr[0]:.3f}, B={B_fit:.3f} ± {perr[1]:.3f}")
    return A_fit, perr[0]


if __name__ == "__main__":
    for nq in [6, 8]:
        times, mean_tau, std_tau = simulate(n_qubits=nq, n_steps=150, dt=0.1, n_realizations=10)
        A_fit, A_err = analyze_and_plot(times, mean_tau, std_tau)
        print(f"System size {nq}: A = {A_fit:.3f} ± {A_err:.3f}")
