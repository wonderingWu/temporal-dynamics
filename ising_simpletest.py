import numpy as np
import matplotlib.pyplot as plt

class SimpleIsing1D:
    def __init__(self, size):
        self.size = size
        # 随机初始化自旋状态
        self.spins = np.random.choice([-1, 1], size=size)
    
    def calculate_energy(self):
        """计算系统能量（使用周期性边界条件）"""
        energy = 0
        for i in range(self.size):
            # 相邻自旋相互作用
            j = (i + 1) % self.size  # 周期性边界条件
            energy += -self.spins[i] * self.spins[j]
        return energy
    
    def get_state_index(self):
        """将自旋状态转换为整数索引，例如 [1, 0] -> 2"""
        # 将自旋状态映射到二进制表示
        # 注意：这里先将-1转换为0
        binary_state = np.where(self.spins == -1, 0, self.spins)
        # 将二进制数组转换为整数
        index = 0
        for bit in binary_state:
            index = (index << 1) | bit
        return index
    
    def metropolis_update(self, temperature):
        """使用Metropolis算法更新自旋状态"""
        # 随机选择一个自旋
        i = np.random.randint(0, self.size)
        
        # 计算翻转后的能量变化
        # 考虑左右邻居（使用周期性边界条件）
        left = (i - 1) % self.size
        right = (i + 1) % self.size
        
        # 能量变化 = 2 * spin_i * (spin_left + spin_right)
        dE = 2 * self.spins[i] * (self.spins[left] + self.spins[right])
        
        # Metropolis接受准则
        if dE <= 0 or np.random.rand() < np.exp(-dE / temperature):
            # 接受翻转
            self.spins[i] *= -1
            return True
        return False

def run_simulation(size=4, temperature=1.0, steps=1000):
    """运行简单的Ising模型模拟"""
    model = SimpleIsing1D(size)
    energies = []
    magnetizations = []
    
    for step in range(steps):
        # 执行Metropolis更新
        model.metropolis_update(temperature)
        
        # 记录能量和磁化强度
        energies.append(model.calculate_energy())
        magnetizations.append(np.sum(model.spins))
        
        # 每100步打印一次状态
        if step % 100 == 0:
            print(f"Step {step}, State: {model.spins}, Index: {model.get_state_index()}")
    
    return np.array(energies), np.array(magnetizations)

def plot_results(energies, magnetizations):
    """绘制模拟结果"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(energies, 'b-', alpha=0.7)
    ax1.set_xlabel('Monte Carlo Steps')
    ax1.set_ylabel('Energy')
    ax1.set_title('Ising Model Energy Evolution')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(magnetizations, 'r-', alpha=0.7)
    ax2.set_xlabel('Monte Carlo Steps')
    ax2.set_ylabel('Magnetization')
    ax2.set_title('Ising Model Magnetization Evolution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ising_simpletest_results.png')
    plt.close()

if __name__ == "__main__":
    print("Running simple Ising model test...")
    energies, magnetizations = run_simulation(size=4, temperature=1.0, steps=1000)
    plot_results(energies, magnetizations)
    print("Simulation completed. Results saved to 'ising_simpletest_results.png'")
    print(f"Final energy: {energies[-1]}, Final magnetization: {magnetizations[-1]}")