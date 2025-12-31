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
    """计算目标 qubit 的冯诺依曼熵 (使用更稳定的方法)"""
    dim = 2 ** n_qubits
    # 将态矢量重塑为 (2, 2^(n-1)) 的矩阵
    psi = state.reshape((2, dim // 2))
    # 计算约化密度矩阵 rho_A = psi * psi^dagger
    rho_A = psi @ psi.conj().T
    # 计算特征值
    eigvals = np.linalg.eigvalsh(rho_A)
    # 过滤掉数值上为零的特征值，避免 log(0)
    eigvals = eigvals[eigvals > 1e-10]
    # 计算冯诺依曼熵
    return -np.sum(eigvals * np.log(eigvals))


def simulate(n_qubits=6, n_steps=200, dt=0.1, n_realizations=10):
    """多次随机实例模拟"""
    dim = 2 ** n_qubits
    times = np.arange(0, n_steps * dt, dt)
    entropy_runs = []
    dE_runs = []  # 新增：用于存储每次运行的能量变化

    for run in range(n_realizations):
        H = build_hamiltonian(n_qubits, seed=run)
        psi = np.ones(dim) / np.sqrt(dim)  # 均匀叠加态
        cumulative_entropy = []
        cumulative_dE = []  # 新增：存储累积能量变化
        total_dE = 0.0  # 新增：累积的能量变化量

        # 计算初始能量
        E_prev = np.real(psi.conj() @ H @ psi)

        for t in times:
            psi = expm_multiply(-1j * H * dt, psi)
            S = single_qubit_entropy(psi, n_qubits, target=0)
            cumulative_entropy.append(S)

            # 计算当前能量和能量变化
            E_curr = np.real(psi.conj() @ H @ psi)
            dE = abs(E_curr - E_prev)  # 取绝对值，代表耗散
            total_dE += dE
            cumulative_dE.append(total_dE)
            E_prev = E_curr  # 更新前一时刻能量

        entropy_runs.append(cumulative_entropy)
        dE_runs.append(cumulative_dE)  # 存储本次运行的 dE 轨迹

    # 转换为 numpy 数组
    entropy_runs = np.array(entropy_runs)
    dE_runs = np.array(dE_runs)

    # 计算均值和标准差
    mean_entropy = np.mean(entropy_runs, axis=0)
    std_entropy = np.std(entropy_runs, axis=0)
    mean_dE = np.mean(dE_runs, axis=0)
    std_dE = np.std(dE_runs, axis=0)

    return times, mean_entropy, std_entropy, mean_dE, std_dE


def analyze_and_plot(times, mean_entropy, std_entropy, mean_dE, std_dE):
    """使用 mean_dE 作为 tau_proxy 进行拟合并绘图"""
    # 去掉前段噪声
    start = len(times) // 5
    t_data = times[start:]
    tau_data = mean_dE[start:]
    tau_err = std_dE[start:]
    S_data = mean_entropy[start:]
    S_err = std_entropy[start:]

    # 对数拟合 t ~ A ln(tau) + B
    def log_func(tau, A, B):
        return A * np.log(tau) + B

    try:
        params, cov = curve_fit(log_func, tau_data, t_data, p0=[1, 0])
        A_fit, B_fit = params
        perr = np.sqrt(np.diag(cov))
    except RuntimeError as e:
        print(f"Curve fitting failed: {e}")
        return

    tau_fit = np.linspace(min(tau_data), max(tau_data), 100)
    t_fit = log_func(tau_fit, A_fit, B_fit)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (a) t vs ln(tau)
    ax1 = axes[0]
    # 计算 ln(tau) 的误差 (使用误差传播)
    ln_tau_data = np.log(tau_data)
    ln_tau_err = tau_err / tau_data
    ax1.errorbar(ln_tau_data, t_data, yerr=S_err, xerr=ln_tau_err,
                 fmt='bo', markersize=4, alpha=0.6, label='Simulation (mean ± std)')
    ax1.plot(np.log(tau_fit), t_fit, 'r-', lw=2,
             label=rf'Fit: $t = {A_fit:.2f}\ln\tau {B_fit:+.2f}$')
    ax1.set_xlabel(r'$\ln(\tau_{\rm proxy})$')
    ax1.set_ylabel('External Time $t$')
    ax1.set_title(r'(a) $t$ vs. $\ln(\tau)$ (averaged)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # (b) Emergent vs External time
    t_emergent = log_func(tau_data, A_fit, B_fit)
    ax2 = axes[1]
    ax2.plot(t_data, t_emergent, 'g-', label='Emergent time')
    ax2.plot([min(t_data), max(t_data)], [min(t_data), max(t_data)], 'k--', label='Ideal $t$')
    ax2.set_xlabel('External Time $t$')
    ax2.set_ylabel(r'Emergent Time $k\ln\tau$')
    ax2.set_title('(b) Validation: Emergent vs External')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("quan_poly_tau_proxy_new.png", dpi=300)
    plt.show()

    print(f"Fit results: A={A_fit:.3f} ± {perr[0]:.3f}, B={B_fit:.3f} ± {perr[1]:.3f}")
    return A_fit, perr[0]


if __name__ == "__main__":
    for nq in [6, 8]:
        times, mean_S, std_S, mean_tau, std_tau = simulate(n_qubits=nq, n_steps=150, dt=0.1, n_realizations=10)
        A_fit, A_err = analyze_and_plot(times, mean_S, std_S, mean_tau, std_tau)
        print(f"System size {nq}: A = {A_fit:.3f} ± {A_err:.3f}")
        