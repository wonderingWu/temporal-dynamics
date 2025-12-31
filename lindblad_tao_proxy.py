import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Configure matplotlib settings
plt.rcParams["axes.unicode_minus"] = False  # Fix minus sign display issue

# Define three different noise spectrum functions
def white_noise(t):
    return 1.0  # 常数

def one_over_f_noise(t):
    return 1.0 / (1.0 + t)  # Approximate 1/f behavior on short timescales

def lorentzian_noise(t):
    omega_c = 5.0  # 截止频率
    return 1.0 / (1.0 + (t / omega_c)**2) # 近似洛伦兹谱

# Define Lindblad master equation (pure dephasing only)
def master_equation(t, rho_vec, gamma_func):
    # rho_vec is the vector form of density matrix [rho_00, Re(rho_01), Im(rho_01), rho_11]
    # For pure dephasing, Hamiltonian part is 0, we only care about noise
    gamma_t = gamma_func(t)
    
    # Dephasing rate is proportional to gamma_t
    dephasing_rate = 0.1 * gamma_t
    
    # Right-hand side of master equation: d(rho)/dt = L[rho]
    drho = np.zeros(4)
    # Diagonal elements remain unchanged
    # Off-diagonal elements decay: d(rho_01)/dt = -dephasing_rate * rho_01
    drho[1] = -dephasing_rate * rho_vec[1]  # Re(rho_01)
    drho[2] = -dephasing_rate * rho_vec[2]  # Im(rho_01)
    return drho

# Initial state: superposition state |+> = (|0> + |1>)/sqrt(2)
rho0 = np.array([0.5, 0.5, 0.0, 0.5]) # [0.5, 0.5, 0, 0.5]

# Time points
t_eval = np.linspace(0, 10, 500)

# Store results
results = {}
gamma_integrals = {} # This will store our proxy variable τ_proxy = ∫γ(t')dt'

noise_types = {
    'White': white_noise,
    '1/f': one_over_f_noise,
    'Lorentzian': lorentzian_noise
}

# Simulate evolution under different noises
for name, noise_func in noise_types.items():
    # Solve master equation
    sol = solve_ivp(master_equation, [0, 10], rho0, t_eval=t_eval, args=(noise_func,), method='BDF')
    results[name] = sol.y
    
    # Calculate our proxy variable τ_proxy = Γ_cum(t) = ∫₀ᵗ γ(t') dt'
    # Numerical integration
    gamma_vals = np.array([noise_func(t) for t in sol.t])
    gamma_integral = np.cumsum(gamma_vals) * (sol.t[1] - sol.t[0]) # 累积积分
    gamma_integrals[name] = gamma_integral

# Calculate von Neumann entropy S = -Tr(ρ ln ρ)
entropies = {}
for name, rho in results.items():
    S_list = []
    for i in range(len(rho[0])):
        rho_matrix = np.array([[rho[0][i], rho[1][i] + 1j*rho[2][i]],
                              [rho[1][i] - 1j*rho[2][i], rho[3][i]]])
        eigvals = np.linalg.eigvalsh(rho_matrix)
        eigvals = eigvals[eigvals > 1e-12]  # 避免log(0)
        S = -np.sum(eigvals * np.log(eigvals))
        S_list.append(S)
    entropies[name] = np.array(S_list)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Figure 1: Entropy vs. regular time
ax1 = axes[0]
for name, S in entropies.items():
    ax1.plot(t_eval, S, label=name, lw=2)
ax1.set_xlabel('External Time, $t$')
ax1.set_ylabel('Von Neumann Entropy, $S(t)$')
ax1.set_title('Entropy Evolution in Different Noise Spectra')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Figure 2: Entropy vs. logarithm of our proxy variable Γ_cum (τ_proxy)
ax2 = axes[1]
for name, S in entropies.items():
    Γ_cum = gamma_integrals[name]
    # 取对数，加一个微小值避免log(0)
    log_Γ = np.log(Γ_cum + 1e-6)
    ax2.plot(log_Γ, S, label=name, lw=2)
ax2.set_xlabel(r'Log Cumulative Decoherence, $\ln(\Gamma_{\rm cum}(t))$')
ax2.set_ylabel('Von Neumann Entropy, $S$')
ax2.set_title('Universal Scaling under Logarithmic Projection\n(Proof of Concept)')
ax2.legend()
ax2.grid(True, alpha=0.3)

def main():
    # 初始状态：叠加态 |+> = (|0> + |1>)/sqrt(2)
    rho0 = np.array([0.5, 0.5, 0.0, 0.5]) # [0.5, 0.5, 0, 0.5]

    # 时间点
    t_eval = np.linspace(0, 10, 500)

    # 存储结果
    results = {}
    gamma_integrals = {} # 这将存储我们的代理变量 τ_proxy = ∫γ(t')dt'

    noise_types = {
        'White': white_noise,
        '1/f': one_over_f_noise,
        'Lorentzian': lorentzian_noise
    }

    print("Starting simulation of evolution under different noises...")
    # 模拟不同噪声下的演化
    for name, noise_func in noise_types.items():
        print(f"  Processing {name} noise...")
        # 求解主方程
        sol = solve_ivp(master_equation, [0, 10], rho0, t_eval=t_eval, args=(noise_func,), method='BDF')
        results[name] = sol.y

        # 计算我们的代理变量 τ_proxy = Γ_cum(t) = ∫₀ᵗ γ(t') dt'
        # 数值积分
        gamma_vals = np.array([noise_func(t) for t in sol.t])
        gamma_integral = np.cumsum(gamma_vals) * (sol.t[1] - sol.t[0]) # 累积积分
        gamma_integrals[name] = gamma_integral

    print("Calculating von Neumann entropy...")
    # 计算冯诺依曼熵 S = -Tr(ρ ln ρ)
    entropies = {}
    for name, rho in results.items():
        S_list = []
        for i in range(len(rho[0])):
            rho_matrix = np.array([[rho[0][i], rho[1][i] + 1j*rho[2][i]],
                                  [rho[1][i] - 1j*rho[2][i], rho[3][i]]])
            eigvals = np.linalg.eigvalsh(rho_matrix)
            eigvals = eigvals[eigvals > 1e-12]  # 避免log(0)
            S = -np.sum(eigvals * np.log(eigvals))
            S_list.append(S)
        entropies[name] = np.array(S_list)

    print("Plotting figures...")
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 图1：熵 vs 常规时间
    ax1 = axes[0]
    for name, S in entropies.items():
        ax1.plot(t_eval, S, label=name, lw=2)
    ax1.set_xlabel('External Time, $t$')
    ax1.set_ylabel('Von Neumann Entropy, $S(t)$')
    ax1.set_title('Entropy Evolution in Different Noise Spectra')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 图2：熵 vs 我们的代理变量 Γ_cum (τ_proxy) 的对数
    ax2 = axes[1]
    for name, S in entropies.items():
        Γ_cum = gamma_integrals[name]
        # 取对数，加一个微小值避免log(0)
        log_Γ = np.log(Γ_cum + 1e-6)
        ax2.plot(log_Γ, S, label=name, lw=2)
    ax2.set_xlabel(r'Log Cumulative Decoherence, $\ln(\Gamma_{\rm cum}(t))$')
    ax2.set_ylabel('Von Neumann Entropy, $S$')
    ax2.set_title('Universal Scaling under Logarithmic Projection (Proof of Concept)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('lindblad_tao_results.png')
    print("Figure saved as lindblad_tao_results.png")
    plt.show()
    print("Simulation completed!")

if __name__ == "__main__":
    main()

