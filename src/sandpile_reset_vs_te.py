import numpy as np
import jpype
import jpype.imports
import os

# 路径检查
jar_path = "infodynamics.jar"
if not os.path.exists(jar_path):
    # 尝试在上一级目录查找，方便调试
    if os.path.exists(f"../{jar_path}"):
        jar_path = f"../{jar_path}"
    else:
        raise FileNotFoundError(f"JIDT jar not found. Please check path.")

# 避免重复启动 JVM
if not jpype.isJVMStarted():
    jpype.startJVM(classpath=[jar_path])

# 导入类
from infodynamics.measures.discrete import TransferEntropyCalculatorDiscrete

def run_sandpile(L=32, total_additions=10000, reset_every=None, seed=0, burn_in=5000):
    np.random.seed(seed)
    grid = np.zeros((L, L), dtype=np.int16)
    activity = []
    steps_since_reset = 0
    
    # === 预热阶段 (Burn-in) ===
    # 仅针对非重置模式，或者重置模式的第一个周期
    print(f"Burn-in phase ({burn_in} steps)...")
    for _ in range(burn_in):
        i, j = np.random.randint(0, L, 2)
        grid[i, j] += 1
        while np.any(grid >= 4):
            unstable = grid >= 4
            # 简化版矩阵操作，速度更快
            dgrid = np.zeros_like(grid)
            dgrid[:-1, :] += unstable[1:, :]
            dgrid[1:, :] += unstable[:-1, :]
            dgrid[:, :-1] += unstable[:, 1:]
            dgrid[:, 1:] += unstable[:, :-1]
            grid[unstable] -= 4
            grid += dgrid
        
        if reset_every: # 如果在预热期也要重置逻辑
            steps_since_reset += 1
            if steps_since_reset >= reset_every:
                grid.fill(0)
                steps_since_reset = 0

    # === 正式采样阶段 ===
    print("Sampling phase...")
    for _ in range(total_additions):
        i, j = np.random.randint(0, L, 2)
        grid[i, j] += 1
        
        # 计算雪崩大小
        total_active = 0
        while np.any(grid >= 4):
            unstable = grid >= 4
            count = np.sum(unstable)
            total_active += count
            
            dgrid = np.zeros_like(grid)
            dgrid[:-1, :] += unstable[1:, :]
            dgrid[1:, :] += unstable[:-1, :]
            dgrid[:, :-1] += unstable[:, 1:]
            dgrid[:, 1:] += unstable[:, :-1]
            grid[unstable] -= 4
            grid += dgrid
            
        activity.append(total_active)

        if reset_every:
            steps_since_reset += 1
            if steps_since_reset >= reset_every:
                grid.fill(0)
                steps_since_reset = 0

    return np.array(activity, dtype=int)

def compute_discrete_self_te(time_series, k=1, tau=1, max_symbols=10):
    """
    计算自传递熵 (AIS)。
    修正了构造函数调用和数据处理。
    """
    series = np.array(time_series, dtype=int)
    
    # 1. 对数分箱 (Log-Binning) 
    # SOC 雪崩跨度极大，必须分箱，否则 base 会爆炸导致内存溢出
    # 将 0 (无雪崩) 单独作为一类
    binned_series = np.zeros_like(series)
    
    mask_nonzero = series > 0
    if np.any(mask_nonzero):
        vals = series[mask_nonzero]
        # 使用较少的符号数，防止稀疏性偏差
        # 对数分箱：小雪崩分得细，大雪崩分得粗
        bins = np.logspace(0, np.log10(vals.max() + 1), max_symbols)
        # digitize 返回 1..max_symbols, 0 保留给 series==0
        binned_vals = np.digitize(vals, bins)
        binned_series[mask_nonzero] = binned_vals

    base = len(np.unique(binned_series))
    # 强制 base 至少为 2，避免 JIDT 报错
    base = max(base, 2) 

    # 2. 准备源和目标
    # TE(Source -> Dest). 对于 AIS: Source = X_{t-tau}, Dest = X_t
    dest = binned_series[tau:]
    source = binned_series[:-tau]

    if len(dest) == 0: return 0.0

    # 3. 调用 JIDT
    # 修正：只传递 base 和 k。tau 已经通过切片体现了。
    te_calc = TransferEntropyCalculatorDiscrete(base, k)
    te_calc.initialise()
    te_calc.addObservations(source.tolist(), dest.tolist()) # 注意 JIDT 通常是 (source, dest)
    
    result = te_calc.computeAverageLocalOfObservations()
    return float(result)

# --- 运行 ---
if __name__ == "__main__":
    # 增加模拟长度以获得统计显著性
    L_size = 32
    N_steps = 50000 
    
    print("Running SOC (Critical State)...")
    soc_data = run_sandpile(L=L_size, total_additions=N_steps, reset_every=None, seed=42, burn_in=10000)
    
    print("Running Suppressed (Frequent Reset)...")
    # reset_every 设为 2000，让它稍微积累一点东西，否则全是对 0 求熵
    supp_data = run_sandpile(L=L_size, total_additions=N_steps, reset_every=2000, seed=42, burn_in=0)

    print("\nCalculating TE...")
    # k=1 或 k=2 即可，太大会导致欠采样
    # max_symbols 不要太大，SOC 数据稀疏
    k_hist = 1
    te_soc = compute_discrete_self_te(soc_data, k=k_hist, tau=1, max_symbols=5)
    te_supp = compute_discrete_self_te(supp_data, k=k_hist, tau=1, max_symbols=5)

    print(f"\nResults (L={L_size}):")
    print(f"SOC TE (Information Storage): {te_soc:.5f} bits")
    print(f"Suppressed TE                 : {te_supp:.5f} bits")
    
    if te_supp > 0:
        print(f"Ratio (SOC/Supp): {te_soc / te_supp:.2f}x")
    
    jpype.shutdownJVM()



# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.neighbors import NearestNeighbors
# import warnings
# import os  # 提前导入os模块
# warnings.filterwarnings("ignore")

# # 定义脚本目录变量
# try:
#     script_dir = os.path.dirname(os.path.abspath(__file__))
# except NameError:
#     script_dir = os.getcwd()

# # -------------------------------------------------
# # 1. Sandpile Simulator
# # -------------------------------------------------
# def run_sandpile(L, total_additions, reset_every=None, seed=42):
#     """
#     Run sandpile model and return activity time series.
#     Args:
#         L: grid size (L x L)
#         total_additions: total number of grain additions
#         reset_every: reset grid every N additions (None = no reset)
#         seed: random seed
#     Returns: activity time series (avalanche sizes)
#     """
#     try:
#         np.random.seed(seed)
#         grid = np.zeros((L, L), dtype=np.int16)
#         activity = []
#         steps_since_reset = 0        
#         def topple():
#             nonlocal grid
#             total_active = 0
#             while np.any(grid >= 4):
#                 unstable = grid >= 4
#                 total_active += np.sum(unstable)
#                 # Simultaneous toppling
#                 dgrid = np.zeros_like(grid)
#                 dgrid[:-1, :] += unstable[1:, :]
#                 dgrid[1:, :] += unstable[:-1, :]
#                 dgrid[:, :-1] += unstable[:, 1:]
#                 dgrid[:, 1:] += unstable[:, :-1]
#                 grid[unstable] -= 4
#                 grid += dgrid
#             return total_active        
#         for step in range(total_additions):
#             if step % 100 == 0 and step > 0:
#                 print(f"    Sandpile step {step}/{total_additions}...")
#             i, j = np.random.randint(0, L, 2)
#             grid[i, j] += 1
#             act = topple()
#             activity.append(act)
            
#             if reset_every:
#                 steps_since_reset += 1
#                 if steps_since_reset >= reset_every:
#                     grid.fill(0)
#                     steps_since_reset = 0        
#         return np.array(activity, dtype=np.float32)
#     except Exception as e:
#         print(f"ERROR in run_sandpile: {e}")
#         import traceback
#         traceback.print_exc()
#         return np.array([0.0])

# # -------------------------------------------------
# # 2. Approximate Transfer Entropy (TE) Estimator
# # -------------------------------------------------
# def approx_te(time_series, k=1, delay=1, n_neighbors=3, subsample=5000):
#     """
#     Approximate TE using k-nearest neighbor entropy estimation.
#     TE(X -> Y) ≈ I(Y_t; X_{t-1}, ..., X_{t-k} | Y_{t-1}, ..., Y_{t-k})
#     Here we compute self-TE: X = Y = activity series.
#     """
#     X = time_series
#     N = len(X)
#     if N < 1000:
#         return 0.0
    
#     # Subsample for speed
#     if N > subsample:
#         idx = np.random.choice(N - k - delay, size=subsample, replace=False)
#     else:
#         idx = np.arange(N - k - delay)
    
#     # Prepare embedded vectors
#     past_X = np.column_stack([X[i - delay : i - delay - k : -delay] for i in idx + k + delay]).T
#     past_Y = past_X.copy()
#     current_Y = X[idx + k + delay]

#     # Helper: KNN entropy estimator
#     def knn_entropy(data, k=n_neighbors):
#         if len(data) < k + 1:
#             return 0.0
#         nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(data)
#         distances, _ = nbrs.kneighbors(data)
#         r = distances[:, -1]
#         r[r == 0] = 1e-10  # avoid log(0)
#         d = data.shape[1]
#         H = np.log(r).mean() + np.log(2 * np.pi * np.e) / 2 + np.log(k) / len(data)
#         return H

#     # Compute entropies for TE = H(Y_t|Y_{t-1}) - H(Y_t|Y_{t-1}, X_{t-1})
#     # Since X=Y, this is auto-TE
#     H_y_given_y = knn_entropy(np.column_stack([past_Y, current_Y])) - knn_entropy(past_Y)
#     H_y_given_xy = 0.0  # For auto-TE, this is the same as above with full past
#     # Simpler: use mutual info between current and past as proxy
#     from sklearn.feature_selection import mutual_info_regression
#     mi = mutual_info_regression(past_Y[:1000], current_Y[:1000], random_state=0).mean()
#     return mi

# # -------------------------------------------------
# # 3. Run Experiments with Uncertainty Quantification
# # -------------------------------------------------
# def run_experiment(L, total_additions, reset_every, seed):
#     """Run a single experiment and return the TE."""
#     try:
#         print(f"    Running sandpile with reset_every={reset_every}, seed={seed}...")
#         series = run_sandpile(L=L, total_additions=total_additions, reset_every=reset_every, seed=seed)
#         print(f"    Calculating TE...")
#         te = approx_te(series, k=3, delay=1)
#         return te
#     except Exception as e:
#         print(f"ERROR in run_experiment: {e}")
#         import traceback
#         traceback.print_exc()
#         return 0.0

# def run_multiple_experiments(L, total_additions, reset_every, num_runs=5):
#     """Run multiple experiments and return TE mean and std."""
#     tes = []
#     for seed in range(num_runs):
#         te = run_experiment(L, total_additions, reset_every, seed)
#         tes.append(te)
#     return np.mean(tes), np.std(tes)

# # Parameters
# L = 8  # 进一步减小晶格尺寸加速调试
# additions = 1000  # 进一步减少总步数加速调试

# # Part 1: Quantify uncertainty with 5 repetitions for standard SOC and specific reset interval
# print("Part 1: Quantifying uncertainty...")

# # Run standard SOC (reset_every=None) multiple times
# print("Running standard SOC sandpile (5 repetitions)...")

# # Add debug information to understand execution flow
# def debug_run_multiple_experiments(L, total_additions, reset_every, num_runs=5):
#     """Run multiple experiments with debug info"""
#     tes = []
#     for i in range(num_runs):
#         print(f"  Run {i+1}/{num_runs}...")
#         te = run_experiment(L, total_additions, reset_every, seed=i)
#         print(f"  Run {i+1}/{num_runs} completed, TE = {te:.6f}")
#         tes.append(te)
#     return np.mean(tes), np.std(tes)

# soc_te_mean, soc_te_std = debug_run_multiple_experiments(L, additions, reset_every=None, num_runs=5)  # Run with 5 repetitions as stated

# # Run suppressed sandpile with specific reset interval multiple times
# reset_interval = 200
# print(f"Running suppressed sandpile with reset_every={reset_interval} (5 repetitions)...")
# suppressed_te_mean, suppressed_te_std = run_multiple_experiments(L, additions, reset_every=reset_interval, num_runs=5)

# print(f"\nResults:")
# print(f"Standard SOC TE (proxy): {soc_te_mean:.4f} ± {soc_te_std:.4f}")
# print(f"Suppressed TE (proxy):   {suppressed_te_mean:.4f} ± {suppressed_te_std:.4f}")
# print(f"Ratio (SOC / Suppressed): {soc_te_mean / (suppressed_te_mean + 1e-8):.2f}x")

# # Part 2: Explore phase transition by scanning reset_every values
# print("\nPart 2: Exploring reset frequency phase transition...")
# reset_every_values = [100, 500, 1000, 2000, 5000, None]  # None represents ∞
# te_means = []
# te_stds = []

# for reset_every in reset_every_values:
#     print(f"Running experiments with reset_every={reset_every}...")
#     te_mean, te_std = run_multiple_experiments(L, additions, reset_every, num_runs=3)  # Use 3 runs for faster scanning
#     te_means.append(te_mean)
#     te_stds.append(te_std)

# # -------------------------------------------------
# # 4. Visualization
# # -------------------------------------------------
# # Figure 1: Comparison of activity time series for standard SOC and specific suppressed case
# plt.figure(figsize=(12, 5))

# # Run one more time to get activity series for plotting
# print("Generating visualization plots...")
# soc_series = run_sandpile(L=L, total_additions=additions, reset_every=None, seed=42)
# suppressed_series = run_sandpile(L=L, total_additions=additions, reset_every=reset_interval, seed=42)

# plt.subplot(1, 2, 1)
# plt.plot(soc_series[:5000], lw=0.5, color='steelblue')
# plt.title(f"Standard SOC (TE ≈ {soc_te_mean:.3f} ± {soc_te_std:.3f})")
# plt.ylabel("Avalanche Size")
# plt.xlabel("Time Step")

# plt.subplot(1, 2, 2)
# plt.plot(suppressed_series[:5000], lw=0.5, color='orange')
# plt.title(f"Suppressed (reset_every={reset_interval}, TE ≈ {suppressed_te_mean:.3f} ± {suppressed_te_std:.3f})")
# plt.xlabel("Time Step")

# plt.tight_layout()
# activity_plot_file = os.path.join(script_dir, "sandpile_te_comparison.png")
# plt.savefig(activity_plot_file, dpi=300)
# print(f"Activity comparison plot saved to: {activity_plot_file}")

# # Figure 2: TE vs. Reset Period
# plt.figure(figsize=(10, 6))

# # Convert reset_every values for plotting (replace None with a large number for visualization)
# reset_labels = [str(val) if val is not None else "∞" for val in reset_every_values]
# reset_plot_values = [val if val is not None else 10000 for val in reset_every_values]  # Use 10000 for ∞

# plt.errorbar(reset_plot_values, te_means, yerr=te_stds, fmt='o-', capsize=5, color='purple', linewidth=2, markersize=8)
# plt.xlabel("Reset Period")
# plt.ylabel("TE (proxy) - Mean ± Std")
# plt.title("TE vs. Reset Period (Phase Transition Exploration)")
# plt.grid(True, alpha=0.3)

# # Set x-axis ticks to show actual values
# plt.xticks(reset_plot_values, reset_labels)

# # Highlight the standard SOC case (infinite reset period)
# plt.axvline(x=10000, color='gray', linestyle='--', alpha=0.5, label='Standard SOC (no reset)')
# plt.legend()

# phase_plot_file = os.path.join(script_dir, "sandpile_te_phase_transition.png")
# plt.savefig(phase_plot_file, dpi=300)
# print(f"Phase transition plot saved to: {phase_plot_file}")

# # -------------------------------------------------
# # 5. Save Results
# # -------------------------------------------------
# import pandas as pd

# # Save uncertainty results
# results_uncertainty = pd.DataFrame({
#     "System": ["Standard SOC", "Suppressed"],
#     "Reset_Every": ["∞", reset_interval],
#     "TE_Mean": [soc_te_mean, suppressed_te_mean],
#     "TE_Std": [soc_te_std, suppressed_te_std]
# })
# uncertainty_file = os.path.join(script_dir, "sandpile_te_uncertainty.csv")
# results_uncertainty.to_csv(uncertainty_file, index=False)
# print(f"Uncertainty results saved to: {uncertainty_file}")

# # Save phase transition results
# results_phase = pd.DataFrame({
#     "Reset_Every": reset_labels,
#     "Reset_Plot_Values": reset_plot_values,
#     "TE_Mean": te_means,
#     "TE_Std": te_stds
# })
# phase_file = os.path.join(script_dir, "sandpile_te_phase_transition.csv")
# results_phase.to_csv(phase_file, index=False)
# print(f"Phase transition results saved to: {phase_file}")

# # Save all results to a single file for convenience
# all_results = pd.concat([results_uncertainty, results_phase], ignore_index=True)

# all_results_file = os.path.join(script_dir, "sandpile_te_all_results.csv")
# all_results.to_csv(all_results_file, index=False)
# print(f"All results saved to: {all_results_file}")

# print("\nAnalysis complete!")
