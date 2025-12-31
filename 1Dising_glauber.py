import numpy as np
import matplotlib.pyplot as plt

class Ising1D_Glauber:
    def __init__(self, size, temperature):
        self.size = size
        self.T = temperature
        self.spins = np.random.choice([-1, 1], size=size)
        self.energy = self.calculate_energy()
        self.magnetization = np.sum(self.spins)

    def calculate_energy(self):
        energy = 0
        for i in range(self.size):
            spin = self.spins[i]
            # 使用模运算处理周期性边界条件
            neighbors = self.spins[(i-1) % self.size] + self.spins[(i+1) % self.size]
            energy += -spin * neighbors
        return energy / 2  # 每个键被计算了两次

    def glauber_step(self):
        i = np.random.randint(0, self.size)
        spin_old = self.spins[i]
        neighbors = self.spins[(i-1) % self.size] + self.spins[(i+1) % self.size]
        # 计算翻转自旋的能量变化
        dE = 2 * spin_old * neighbors
        
        # Glauber动力学接受概率
        p_flip = 1.0 / (1.0 + np.exp(dE / self.T))
        
        if np.random.rand() < p_flip:
            self.spins[i] *= -1
            self.energy += dE
            self.magnetization += 2 * self.spins[i]
            # 返回本次翻转带来的物理成本
            return abs(dE), dE ** 2, 1.0  # 分别对应 Σ|ΔE|, Σ(ΔE)^2, #accepted moves
        return 0.0, 0.0, 0.0

def run_simulation(size=128, T=0.1, n_steps=50000): # 修改1: 温度降至0.1，步数增加
    """
    运行Glauber动力学模拟，并记录三种代理时间。
    """
    model = Ising1D_Glauber(size, T)
    magnetizations = np.zeros(n_steps)
    tau1 = np.zeros(n_steps)  # Σ|ΔE|
    tau2 = np.zeros(n_steps)  # Σ(ΔE)^2
    tau3 = np.zeros(n_steps)  # #accepted flips
    
    cum1 = cum2 = cum3 = 0.0
    
    for step in range(n_steps):
        dE_abs, dE_sq, accepted = model.glauber_step()
        cum1 += dE_abs
        cum2 += dE_sq
        cum3 += accepted
        
        magnetizations[step] = abs(model.magnetization) / size
        tau1[step] = cum1
        tau2[step] = cum2
        tau3[step] = cum3
        
    return magnetizations, tau1, tau2, tau3

def plot_results(magnetizations, tau1, tau2, tau3, T, size):
    """
    绘制结果，比较三种代理时间的效果。
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    proxies = [tau1, tau2, tau3]
    proxy_names = ['Σ|ΔE|', 'Σ(ΔE)²', '#accepted moves']
    
    # 第一行: 外部时间 (Monte Carlo steps)
    for col in range(3):
        axes[0, col].plot(magnetizations, 'b-', alpha=0.7)
        axes[0, col].set_title(f'(a{col+1}) Relaxation in External Time\nT={T}, L={size}')
        axes[0, col].set_xlabel('MC Steps')
        axes[0, col].set_ylabel('|M| / site')
        axes[0, col].grid(True, alpha=0.3)
    
    # 第二行: 原始代理时间 τ
    for col, (tau, name) in enumerate(zip(proxies, proxy_names)):
        axes[1, col].plot(tau, magnetizations, 'r-', alpha=0.7)
        axes[1, col].set_title(f'(b{col+1}) Relaxation vs {name}')
        axes[1, col].set_xlabel(name)
        axes[1, col].set_ylabel('|M| / site')
        axes[1, col].grid(True, alpha=0.3)
    
    # 第三行: 对数代理时间 ln(τ) (关键测试)
    for col, (tau, name) in enumerate(zip(proxies, proxy_names)):
        # 过滤掉 tau <= 1 的点，避免 log(0) 和初始瞬态
        valid = tau > 1
        if np.any(valid):
            axes[2, col].plot(np.log(tau[valid]), magnetizations[valid], 'g-', alpha=0.7, lw=2)
        # 处理LaTeX特殊字符
        latex_name = name.replace('#', r'\\#').replace(' ', r'\\ ')
        # 修复LaTeX格式 - 避免在title中直接使用LaTeX
        title = f'(c{col+1}) Test of Universality in Log-Time ln({name})'
        axes[2, col].set_title(title)
        # 使用matplotlib的text.usetex参数或更简单的标签格式
        axes[2, col].set_xlabel(f'ln({name})')
        axes[2, col].set_ylabel('|M| / site')
        axes[2, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f"1Dising_glauber_T{T}_L{size}_revised.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Revised figure saved as {filename}")
    plt.show()

def main():
    # 修改2: 使用更低的温度和更大的系统以观察更清晰的弛豫
    size = 128
    T = 0.1  # 关键修改：使用低温
    n_steps = 50000  # 增加步数以观察完整弛豫
    
    print(f"Starting 1D Ising Glauber simulation at T={T}, L={size}...")
    mag, tau1, tau2, tau3 = run_simulation(size, T, n_steps)
    print("Simulation completed, plotting revised results...")
    plot_results(mag, tau1, tau2, tau3, T, size)
    print("Revised analysis finished.")

if __name__ == "__main__":
    main()
    