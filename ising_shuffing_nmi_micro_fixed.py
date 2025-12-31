import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
from tqdm import tqdm

# ==========================================
# 1. 物理引擎：2D Ising Model (带微观/宏观记录)
# ==========================================
def run_ising_simulation(L, T, num_steps, burn_in):
    """
    返回:
    1. global_mag: 全局磁化强度时间序列 (宏观)
    2. center_spin: 中心点的自旋时间序列 (微观)
    """
    # 初始化
    lattice = np.random.choice([-1, 1], size=(L, L))
    global_mag = []
    center_spin = []
    
    # 预计算玻尔兹曼因子
    exponentials = {dE: np.exp(-dE / T) for dE in [-8, -4, 0, 4, 8]}
    
    # 中心点坐标
    cx, cy = L // 2, L // 2
    
    total_steps = burn_in + num_steps
    
    for step in range(total_steps):
        # Metropolis 扫描
        for _ in range(L * L):
            i, j = np.random.randint(0, L, 2)
            s = lattice[i, j]
            nb = lattice[(i+1)%L, j] + lattice[(i-1)%L, j] + \
                 lattice[i, (j+1)%L] + lattice[i, (j-1)%L]
            dE = 2 * s * nb
            
            if dE <= 0 or np.random.random() < exponentials[dE]:
                lattice[i, j] *= -1
        
        # 记录数据 (仅在预热后)
        if step >= burn_in:
            # 宏观：取绝对值，消除对称性翻转的影响
            mag = np.abs(np.sum(lattice)) / (L * L)
            global_mag.append(mag)
            # 微观：直接记录中心自旋 (+1 或 -1)
            center_spin.append(lattice[cx, cy])
            
    return np.array(global_mag), np.array(center_spin)

# ==========================================
# 2. 分析工具：记忆深度与效率计算
# ==========================================
def analyze_memory(series, lags=[1], bins=10, is_discrete=False):
    """
    计算不同延迟下的互信息 (MI) 和 标准化互信息 (NMI)
    """
    results = {}
    
    # 1. 数据离散化
    if is_discrete:
        # 对于自旋 (-1, 1)，映射为 (0, 1)
        digitized = ((series + 1) / 2).astype(int)
    else:
        # 对于磁化强度，使用固定范围 [0, 1] 进行分箱，防止低熵下的统计偏差
        bin_edges = np.linspace(0, 1, bins + 1)
        digitized = np.digitize(series, bin_edges) - 1 # 0 to bins-1
        # 修正边界溢出
        digitized = np.clip(digitized, 0, bins - 1)

    # 2. 计算自身熵 H(X)
    # 用于计算 NMI = I(X;Y) / H(X)
    # 如果 H(X) 极低 (如低温冻结)，NMI 意义不大，设为 0
    c_counts = np.bincount(digitized)
    c_probs = c_counts[c_counts > 0] / len(digitized)
    entropy_x = -np.sum(c_probs * np.log(c_probs))
    
    results['entropy'] = entropy_x

    # 3. 计算各延迟的 MI - 修复：移除了错误的return语句
    for tau in lags:
        if tau >= len(series):
            results[f'mi_{tau}'] = 0
            results[f'nmi_{tau}'] = 0
            continue
            
        future = digitized[tau:]
        past = digitized[:-tau]
        
        mi = mutual_info_score(past, future)
        
        # 计算标准化互信息 (记忆效率)
        # 也就是：过去的状态解释了多少未来的不确定性？
        if entropy_x > 1e-5:
            nmi = mi / entropy_x
        else:
            nmi = 0.0 # 熵为0时，没有不确定性，也没有"记忆"的必要
            
        results[f'mi_{tau}'] = mi
        results[f'nmi_{tau}'] = nmi
        
    return results

def save_results_to_csv(data, temperatures, lattice_sizes, test_name):
    """保存结果到CSV文件"""
    import pandas as pd
    
    df_data = {'Temperature': temperatures}
    
    for size in lattice_sizes:
        df_data.update({
            f'MI_global_1_L{size}': data.get('global_mi_1', []),
            f'MI_global_50_L{size}': data.get('global_mi_50', []),
            f'NMI_global_1_L{size}': data.get('global_nmi_1', []),
            f'NMI_global_50_L{size}': data.get('global_nmi_50', []),
            f'MI_local_1_L{size}': data.get('local_mi_1', []),
            f'NMI_local_1_L{size}': data.get('local_nmi_1', []),
            f'MI_shuffled_L{size}': data.get('shuffled_mi', [])
        })
    
    df = pd.DataFrame(df_data)
    filename = f'ising_{test_name}_results_L{lattice_sizes[0]}.csv'
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

# ==========================================
# 3. 主实验
# ==========================================
if __name__ == "__main__":
    # 参数
    L = 30              # 稍微加大尺寸以减少有限尺寸效应
    steps = 5000        # 增加步数以获得更好统计
    burn_in = 2000      # 充分预热
    T_range = np.linspace(1.5, 3.5, 40) # 温度扫描
    
    # 测试的延迟：短期 vs 长期
    lags = [1, 50] 
    
    # 存储结果
    data = {
        'T': T_range,
        'global_mi_1': [], 'global_mi_50': [],
        'global_nmi_1': [], 'global_nmi_50': [], # NMI = Normalized MI
        'local_mi_1': [], 'local_nmi_1': [],
        'shuffled_mi': []
    }
    
    print(f"Running Enhanced Falsification Test (L={L})...")
    
    for T in tqdm(T_range):
        # 运行模拟
        g_series, l_series = run_ising_simulation(L, T, steps, burn_in)
        
        # 1. 分析宏观记忆 (Global)
        g_res = analyze_memory(g_series, lags=lags, bins=20, is_discrete=False)
        data['global_mi_1'].append(g_res['mi_1'])
        data['global_mi_50'].append(g_res['mi_50'])
        data['global_nmi_1'].append(g_res['nmi_1'])
        data['global_nmi_50'].append(g_res['nmi_50'])
        
        # 2. 分析微观记忆 (Local Spin)
        l_res = analyze_memory(l_series, lags=[1], is_discrete=True)
        data['local_mi_1'].append(l_res['mi_1'])
        data['local_nmi_1'].append(l_res['nmi_1'])
        
        # 3. 计算基准线 (Shuffled Global)
        g_shuffled = np.random.permutation(g_series)
        s_res = analyze_memory(g_shuffled, lags=[1], bins=20, is_discrete=False)
        data['shuffled_mi'].append(s_res['mi_1'])

    # ==========================================
    # 4. 绘图：多维度验证
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 图 1: 绝对互信息 (Raw MI) - 宏观
    # 验证是否存在仅仅因为"变慢"导致的虚假记忆
    ax = axes[0]
    ax.plot(data['T'], data['global_mi_1'], 'o-', label=r'Short Term ($\tau=1$)')
    ax.plot(data['T'], data['global_mi_50'], 's-', label=r'Long Term ($\tau=50$)')
    ax.plot(data['T'], data['shuffled_mi'], 'x--', color='gray', alpha=0.5, label='Shuffled (Null)')
    ax.axvline(2.269, color='r', linestyle=':', alpha=0.5)
    ax.set_title('Global Memory Capacity (Raw MI)')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Mutual Information (Nats)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 图 2: 标准化互信息 (Efficiency) - 宏观
    # 验证：在排除了熵的影响后，临界点是否依然特殊？
    # 如果低温区 NMI很高而临界区低，说明临界区只是噪音。
    # 如果临界区 NMI 依然有峰值，说明这是"涌现的记忆"。
    ax = axes[1]
    ax.plot(data['T'], data['global_nmi_1'], 'o-', color='purple', label=r'Efficiency ($\tau=1$)')
    ax.plot(data['T'], data['global_nmi_50'], 's-', color='orange', label=r'Efficiency ($\tau=50$)')
    ax.axvline(2.269, color='r', linestyle=':', alpha=0.5)
    ax.set_title('Memory Efficiency (MI / Entropy)')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Normalized MI (0-1)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 图 3: 宏观 vs 微观
    # 验证：记忆是存在于个体，还是存在于整体？
    ax = axes[2]
    # 为了对比，归一化到 0-1 显示趋势
    norm_global = np.array(data['global_mi_1']) / np.max(data['global_mi_1'])
    norm_local = np.array(data['local_mi_1']) / np.max(data['local_mi_1'])
    
    ax.plot(data['T'], norm_global, 'b-', label='Global Memory (Emergent)')
    ax.plot(data['T'], norm_local, 'g--', label='Local Memory (Spin)')
    ax.axvline(2.269, color='r', linestyle=':', alpha=0.5, label='$T_c$')
    ax.set_title('Macro vs Micro Memory (Normalized Trend)')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Relative Strength')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    plt.savefig(f'ising_analysis_L{L}.png')
    save_results_to_csv(data, T_range, [L], 'enhanced')