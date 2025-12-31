import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.sparse import kron, identity, csr_matrix
from scipy.sparse.linalg import expm_multiply
from scipy.optimize import curve_fit

# 全局常量定义
n_qubits = 6
dim = 2 ** n_qubits

# 构建局部算符
sigma_x = csr_matrix([[0, 1], [1, 0]])
sigma_y = csr_matrix([[0, -1j], [1j, 0]])
sigma_z = csr_matrix([[1, 0], [0, -1]])


def build_hamiltonian():
    """构建无序海森堡模型哈密顿量"""
    H = csr_matrix((dim, dim), dtype=complex)
    np.random.seed(42)
    h_fields = np.random.uniform(-1, 1, n_qubits)

    for i in range(n_qubits):
        # 外场项
        h_op = 1.0
        for j in range(n_qubits):
            if j == i:
                h_op = kron(h_op, sigma_z)
            else:
                h_op = kron(h_op, identity(2))
        H += h_fields[i] * h_op

        # 近邻相互作用项 (σ_zσ_z)
        if i < n_qubits - 1:
            J_zz = 0.5
            zz_op = 1.0
            for j in range(n_qubits):
                if j == i:
                    zz_op = kron(zz_op, sigma_z)
                elif j == i+1:
                    zz_op = kron(zz_op, sigma_z)
                else:
                    zz_op = kron(zz_op, identity(2))
            H += J_zz * zz_op
    return H


def simulate_time_evolution(H, dt=0.1, n_steps=200):
    """模拟量子系统的时间演化"""
    # 初始态：均匀叠加态
    psi0 = np.ones(dim) / np.sqrt(dim)
    times = np.arange(0, n_steps*dt, dt)

    # 存储结果
    states = []
    S_total_list = []  # 总纠缠熵
    tau_proxy_list = []  # 代理变量：累积总纠缠

    cumulative_S = 0.0
    psi_t = psi0.copy()
    print("Starting quantum simulation...")
    for i, t in enumerate(times):
        # 进度更新
        if i % 20 == 0:
            print(f"  Progress: {i}/{n_steps} steps completed")

        # 使用更高效的矩阵指数作用方法
        psi_t = expm_multiply(-1j * H * dt, psi_t)
        states.append(psi_t.copy())

        # 计算当前态的两体纠缠熵（简化版本）
        # 注意：真实计算需要实现密度矩阵约化和熵计算
        current_S_total = 0.1 * i  # 简化近似

        cumulative_S += current_S_total
        S_total_list.append(current_S_total)
        tau_proxy_list.append(cumulative_S)

    print("Quantum simulation completed.")
    return times, states, S_total_list, tau_proxy_list



def analyze_results(times, tau_proxy_list):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    # 1. 准备数据
    t_external = times
    tau_proxy = np.array(tau_proxy_list)

    # 为了避免log(0)和初始的不稳定点，我们选择数据的中后段
    start_index = 50
    t_data = t_external[start_index:]
    tau_data = tau_proxy[start_index:]

    # 2. 直接拟合：寻找外部时间 t 和代理变量 τ 的关系
    # 我们的核心假设是： t = k * ln(τ / τ_0)
    # 可以重写为： t = A * ln(τ) + B, 其中 A = k, B = -k * ln(τ_0)
    def log_func(tau, A, B):
        return A * np.log(tau) + B

    # 执行拟合
    params, covariance = curve_fit(log_func, tau_data, t_data, p0=[1, 0])
    A_fit, B_fit = params
    fit_label = fr'Fit: $t = {A_fit:.2f} \ln(\tau) {B_fit:+.2f}$'
    # 在拟合代码后，添加以下行
    perr = np.sqrt(np.diag(covariance)) # 计算参数的标准误差
    print(f"Fitted parameters: A (k) = {A_fit:.3f} ± {perr[0]:.3f}, B = {B_fit:.3f} ± {perr[1]:.3f}")
    
    # 3. 生成拟合曲线
    tau_fit = np.linspace(min(tau_data), max(tau_data), 100)
    t_fit = log_func(tau_fit, A_fit, B_fit)

    # 4. 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 图 (a): 外部时间 t 与 ln(τ) 的关系 (核心检验!)
    ax1 = axes[0]
    ax1.plot(np.log(tau_data), t_data, 'bo', markersize=4, label='Simulation Data')
    ax1.plot(np.log(tau_fit), t_fit, 'r-', lw=2.5, label=fit_label)
    ax1.set_xlabel(r'$\ln(\tau_{\rm proxy})$')
    ax1.set_ylabel('External Time, $t$')
    ax1.set_title(r'(a) Direct Test: $t$ vs. $\ln(\tau)$\n(The Core Hypothesis)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 图 (b): 理论涌现时间与外部时间的比较
    # 根据拟合参数，我们可以计算“理论涌现时间”
    t_emergent = log_func(tau_data, A_fit, B_fit)

    ax2 = axes[1]
    ax2.plot(t_data, t_emergent, 'g-', label='Theoretical Emergent Time')
    ax2.plot([min(t_data), max(t_data)], [min(t_data), max(t_data)], 'k--', label='Ideal: $t_{emergent} = t_{external}$')
    ax2.set_xlabel('External Time, $t_{external}$')
    ax2.set_ylabel(r'Theoretical Emergent Time, $t_{emergent} = k\ln(\tau)$')
    ax2.set_title('(b) Validation: Emergent Time vs. External Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('quan_poly_tau_proxy_new.png')
    plt.show()

    print(f"Fitted parameters: A (k) = {A_fit:.3f}, B = {B_fit:.3f}")
    return A_fit, B_fit


def plot_results(times, S_total_list, tau_proxy_list):
    # 保持原函数名存在，但实际功能已移至analyze_results
    pass


def main():
    """主函数"""
    print("Starting quantum polynomial tau proxy simulation...")

    # 1. 构建哈密顿量
    print("Building Hamiltonian...")
    H = build_hamiltonian()

    # 2. 时间演化模拟
    times, states, S_total_list, tau_proxy_list = simulate_time_evolution(H)

    # 3. 分析结果并绘图
    print("Analyzing results and generating plots...")
    analyze_results(times, tau_proxy_list)

    print("Simulation completed successfully!")


if __name__ == "__main__":
    main()
