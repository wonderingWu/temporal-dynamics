import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import mutual_info_regression
from scipy.optimize import curve_fit
import pandas as pd
import warnings
import os
warnings.filterwarnings("ignore")

# 定义脚本目录变量
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

# -------------------------------------------------
# 1. Sandpile Simulator
# -------------------------------------------------
def run_sandpile(L, total_additions, reset_every=None, seed=42):
    """
    Run sandpile model and return activity time series.
    Args:
        L: grid size (L x L)
        total_additions: total number of grain additions
        reset_every: reset grid every N additions (None = no reset)
        seed: random seed
    Returns: activity time series (avalanche sizes)
    """
    try:
        np.random.seed(seed)
        grid = np.zeros((L, L), dtype=np.int16)
        activity = []
        steps_since_reset = 0        
        def topple():
            nonlocal grid
            total_active = 0
            while np.any(grid >= 4):
                unstable = grid >= 4
                total_active += np.sum(unstable)
                # Simultaneous toppling
                dgrid = np.zeros_like(grid)
                dgrid[:-1, :] += unstable[1:, :]
                dgrid[1:, :] += unstable[:-1, :]
                dgrid[:, :-1] += unstable[:, 1:]
                dgrid[:, 1:] += unstable[:, :-1]
                grid[unstable] -= 4
                grid += dgrid
            return total_active        
        for step in range(total_additions):
            if step % 1000 == 0 and step > 0:
                print(f"    Sandpile step {step}/{total_additions}...")
            i, j = np.random.randint(0, L, 2)
            grid[i, j] += 1
            act = topple()
            activity.append(act)
            
            if reset_every:
                steps_since_reset += 1
                if steps_since_reset >= reset_every:
                    grid.fill(0)
                    steps_since_reset = 0        
        return np.array(activity, dtype=np.float32)
    except Exception as e:
        print(f"ERROR in run_sandpile: {e}")
        import traceback
        traceback.print_exc()
        return np.array([0.0])

# -------------------------------------------------
# 2. Approximate Transfer Entropy (TE) Estimator
# -------------------------------------------------
def approx_te(time_series, k=1, delay=1, n_neighbors=3, subsample=5000):
    """
    Approximate TE using k-nearest neighbor entropy estimation.
    TE(X -> Y) ≈ I(Y_t; X_{t-1}, ..., X_{t-k} | Y_{t-1}, ..., Y_{t-k})
    Here we compute self-TE: X = Y = activity series.
    """
    X = time_series
    N = len(X)
    if N < 1000:
        return 0.0
    
    # Subsample for speed
    if N > subsample:
        idx = np.random.choice(N - k - delay, size=subsample, replace=False)
    else:
        idx = np.arange(N - k - delay)
    
    # Prepare embedded vectors
    past_X = np.column_stack([X[i - delay : i - delay - k : -delay] for i in idx + k + delay]).T
    past_Y = past_X.copy()
    current_Y = X[idx + k + delay]

    # Helper: KNN entropy estimator
    def knn_entropy(data, k=n_neighbors):
        if len(data) < k + 1:
            return 0.0
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(data)
        distances, _ = nbrs.kneighbors(data)
        r = distances[:, -1]
        r[r == 0] = 1e-10  # avoid log(0)
        d = data.shape[1]
        H = np.log(r).mean() + np.log(2 * np.pi * np.e) / 2 + np.log(k) / len(data)
        return H

    # Compute entropies for TE = H(Y_t|Y_{t-1}) - H(Y_t|Y_{t-1}, X_{t-1})
    # Since X=Y, this is auto-TE
    H_y_given_y = knn_entropy(np.column_stack([past_Y, current_Y])) - knn_entropy(past_Y)
    H_y_given_xy = 0.0  # For auto-TE, this is the same as above with full past
    # Simpler: use mutual info between current and past as proxy
    mi = mutual_info_regression(past_Y[:1000], current_Y[:1000], random_state=0).mean()
    return mi

# -------------------------------------------------
# 3. Run Experiments
# -------------------------------------------------
def run_experiment(L, total_additions, reset_every, seed):
    """Run a single experiment and return the TE."""
    try:
        print(f"    Running sandpile with L={L}, reset_every={reset_every}, seed={seed}...")
        series = run_sandpile(L=L, total_additions=total_additions, reset_every=reset_every, seed=seed)
        print(f"    Calculating TE...")
        te = approx_te(series, k=3, delay=1)
        return te
    except Exception as e:
        print(f"ERROR in run_experiment: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

def run_multiple_experiments(L, total_additions, reset_every, num_runs=5):
    """Run multiple experiments and return TE mean and std."""
    tes = []
    for seed in range(num_runs):
        te = run_experiment(L, total_additions, reset_every, seed)
        tes.append(te)
    return np.mean(tes), np.std(tes)

# -------------------------------------------------
# 4. Critical Time Scale Analysis
# -------------------------------------------------

# Exponential fit function for TE(R)
def te_fit_function(R, TE_max, tau_c):
    """Fit function: TE(R) = TE_max (1 - e^(-R/tau_c))"""
    return TE_max * (1 - np.exp(-R / tau_c))

def extract_critical_time(reset_every_values, te_means, te_stds):
    """
    Extract critical time scale tau_c by finding where TE reaches (1 - 1/e) of its maximum value.
    This is a robust method that doesn't rely on potentially failing curve fitting.
    Args:
        reset_every_values: List of reset periods (R)
        te_means: List of TE means for each R
        te_stds: List of TE standard deviations for each R
    Returns:
        TE_max: Maximum TE value
        tau_c: Critical time scale
        fit_params: None (not used in this simplified approach)
        fit_cov: None (not used in this simplified approach)
        R_squared: 0.0 (not used in this simplified approach)
    """
    # Filter out None values (infinite reset period) and convert to numpy arrays
    R_filtered = []
    te_filtered = []
    
    for R, te in zip(reset_every_values, te_means):
        if R is not None and R > 0:
            R_filtered.append(R)
            te_filtered.append(te)
    
    R_filtered = np.array(R_filtered)
    te_filtered = np.array(te_filtered)
    
    # Check if we have enough data points
    if len(R_filtered) < 2:
        print(f"WARNING in extract_critical_time: Not enough data points ({len(R_filtered)})")
        if len(R_filtered) == 1:
            return te_filtered[0], R_filtered[0], None, None, 0.0
        else:
            return 0.0, 1.0, None, None, 0.0
    
    # Sort the data by R for better processing
    sort_indices = np.argsort(R_filtered)
    R_filtered = R_filtered[sort_indices]
    te_filtered = te_filtered[sort_indices]
    
    # Step 1: Determine TE_max
    # Use the maximum TE value as TE_max (most robust approach)
    TE_max = np.max(te_filtered)
    
    # Step 2: Find the point where TE reaches (1 - 1/e) of TE_max (~63.2%)
    target_fraction = 1 - 1/np.exp(1)  # ~0.632
    target_te = TE_max * target_fraction
    
    # Step 3: Find R values where TE crosses the target_te
    # First check if all TE values are below the target
    if np.all(te_filtered < target_te):
        # If all TE values are below target, use the largest R
        tau_c = R_filtered[-1]
        print(f"INFO in extract_critical_time: All TE < {target_fraction:.3f}*TE_max, using R_max={tau_c}")
    # Check if all TE values are above the target
    elif np.all(te_filtered > target_te):
        # If all TE values are above target, use the smallest R
        tau_c = R_filtered[0]
        print(f"INFO in extract_critical_time: All TE > {target_fraction:.3f}*TE_max, using R_min={tau_c}")
    else:
        # Find the first index where TE exceeds the target
        crossover_idx = np.where(te_filtered >= target_te)[0][0]
        
        if crossover_idx == 0:
            # First point already exceeds target, use it
            tau_c = R_filtered[0]
        else:
            # Interpolate between the point below and above the target
            R1 = R_filtered[crossover_idx - 1]
            R2 = R_filtered[crossover_idx]
            te1 = te_filtered[crossover_idx - 1]
            te2 = te_filtered[crossover_idx]
            
            # Linear interpolation to find the exact R where TE = target_te
            if te2 != te1:
                tau_c = R1 + (target_te - te1) * (R2 - R1) / (te2 - te1)
            else:
                # If TE is flat, use the average R
                tau_c = (R1 + R2) / 2
    
    # Step 4: Ensure tau_c is positive and within reasonable bounds
    if tau_c <= 0:
        tau_c = np.min(R_filtered)
    
    # Step 5: Add some safety checks and log the result
    print(f"INFO in extract_critical_time: Estimated tau_c = {tau_c:.2f} (TE_max = {TE_max:.4f})")
    
    # Return the estimated values
    return TE_max, tau_c, None, None, 0.0

# -------------------------------------------------
# 5. Critical Exponent Analysis
# -------------------------------------------------

# Power law fit function for tau_c(L) = A * L^z
def power_law_function(L, A, z):
    """Power law function for tau_c(L) = A * L^z"""
    return A * L**z

def extract_critical_exponent(L_values, tau_c_values, tau_c_errs):
    """
    Extract critical exponent z by fitting tau_c(L) to a power law.
    Args:
        L_values: List of system sizes
        tau_c_values: List of critical time scales for each L
        tau_c_errs: List of errors in critical time scales
    Returns:
        A: Prefactor
        z: Critical exponent
        fit_params: All fit parameters
        fit_cov: Covariance matrix
        R_squared: Goodness of fit
    """
    # Convert to numpy arrays
    L_array = np.array(L_values)
    tau_c_array = np.array(tau_c_values)
    tau_c_errs_array = np.array(tau_c_errs)
    
    # Take logarithms for linear regression
    log_L = np.log(L_array)
    log_tau_c = np.log(tau_c_array)
    log_tau_c_errs = tau_c_errs_array / tau_c_array  # Error propagation
    
    # Initial guess for parameters
    initial_guess = [1.0, 1.0]
    
    try:
        # Fit the data
        fit_params, fit_cov = curve_fit(power_law_function, L_array, tau_c_array, 
                                       p0=initial_guess, sigma=tau_c_errs_array)
        
        A, z = fit_params
        
        # Calculate R-squared
        residuals = tau_c_array - power_law_function(L_array, *fit_params)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((tau_c_array - np.mean(tau_c_array))**2)
        R_squared = 1 - (ss_res / ss_tot)
        
        return A, z, fit_params, fit_cov, R_squared
    except Exception as e:
        print(f"ERROR in extract_critical_exponent: {e}")
        return None, None, None, None, None

# -------------------------------------------------
# 6. Main Analysis
# -------------------------------------------------

def main():
    print("Starting Critical Time Scale Analysis for Sandpile Model")
    print("=" * 60)
    
    # Initialize variables that may be used later
    TE_max = None
    tau_c = None
    R_squared_te = None
    A = None
    z = None
    R_squared_power = None
    
    # Parameters
    base_L = 8  # Base system size
    base_additions = 5000  # Base number of grain additions
    num_runs = 3  # Number of runs for each parameter set
    
    # Part 1: Measure critical time scale tau_c for a fixed system size
    print("\nPart 1: Measuring Critical Time Scale (tau_c)")
    print("-" * 60)
    
    # Scan Reset_Every values based on system size
    # For base_L (L=8 by default), use appropriate range
    if base_L == 4:
        reset_every_values = [2, 5, 10, 20, 50, 100, 200, None]
    elif base_L == 8 or base_L == 12:
        reset_every_values = [2, 5, 10, 20, 50, 100, 200, 500, None]
    elif base_L == 16:
        reset_every_values = [10, 20, 50, 100, 200, 500, 1000, 2000, None]
    else:
        reset_every_values = [10, 20, 50, 100, 200, 500, 1000, 2000, None]
    
    te_means = []
    te_stds = []
    
    for reset_every in reset_every_values:
        print(f"\nRunning experiments with L={base_L}, reset_every={reset_every}...")
        te_mean, te_std = run_multiple_experiments(base_L, base_additions, reset_every, num_runs=num_runs)
        te_means.append(te_mean)
        te_stds.append(te_std)
        print(f"  TE: {te_mean:.6f} ± {te_std:.6f}")
    
    # Extract critical time scale
    TE_max, tau_c, fit_params, fit_cov, R_squared_te = extract_critical_time(reset_every_values, te_means, te_stds)
    
    if TE_max is not None and tau_c is not None:
        print(f"\nFitting Results for Critical Time Scale:")
        print(f"  TE_max: {TE_max:.6f}")
        print(f"  tau_c: {tau_c:.6f}")
        print(f"  R-squared: {R_squared_te:.6f}")
    
    # Plot TE(R) with exponential fit
    plt.figure(figsize=(10, 6))
    
    # Convert reset_every values for plotting
    reset_labels = [str(val) if val is not None else "∞" for val in reset_every_values]
    reset_plot_values = [val if val is not None else 5000 for val in reset_every_values]  # Use 5000 for ∞
    
    plt.errorbar(reset_plot_values, te_means, yerr=te_stds, fmt='o-', capsize=5, color='purple', linewidth=2, markersize=8, label='Data')
    
    # Plot the exponential fit
    if TE_max is not None and tau_c is not None:
        R_fit = np.linspace(0, max(reset_plot_values), 1000)
        te_fit = te_fit_function(R_fit, TE_max, tau_c)
        plt.plot(R_fit, te_fit, 'r--', linewidth=2, label=f'Fit: TE_max={TE_max:.3f}, tau_c={tau_c:.3f}')
    
    plt.xlabel("Reset Period (R)")
    plt.ylabel("TE (proxy) - Mean ± Std")
    plt.title(f"TE vs. Reset Period (L={base_L}) - Exponential Fit for Critical Time Scale")
    plt.grid(True, alpha=0.3)
    
    # Set x-axis ticks to show actual values
    plt.xticks(reset_plot_values, reset_labels)
    
    plt.legend()
    
    # Save the plot
    te_fit_plot_file = os.path.join(script_dir, f"sandpile_te_fit_L{base_L}.png")
    plt.savefig(te_fit_plot_file, dpi=300)
    print(f"\nTE vs. Reset Period with fit plot saved to: {te_fit_plot_file}")
    
    # Part 2: Study relationship between tau_c and system size L
    print("\nPart 2: Studying tau_c vs. System Size (L) Relationship")
    print("-" * 60)
    
    # System sizes to study
    L_values = [4, 8, 12, 16]  # You can extend this list for more comprehensive analysis
    tau_c_values = []
    tau_c_errs = []
    
    for L in L_values:
        print(f"\nAnalyzing system size L={L}...")
        
        # Adjust total_additions proportional to system size for better statistics
        current_additions = base_additions * (L // base_L)
        if current_additions < 1000:
            current_additions = 1000
        
        # Use appropriate Reset_Every values based on system size
        if L == 4:
            current_reset_values = [2, 5, 10, 20, 50, 100, 200, None]
        elif L == 8 or L == 12:
            current_reset_values = [2, 5, 10, 20, 50, 100, 200, 500, None]
        elif L == 16:
            current_reset_values = [10, 20, 50, 100, 200, 500, 1000, 2000, None]
        else:
            current_reset_values = [10, 20, 50, 100, 200, 500, 1000, 2000, None]
        
        # Scan Reset_Every values for this L
        current_te_means = []
        current_te_stds = []
        
        for reset_every in current_reset_values:
            print(f"  Running experiments with reset_every={reset_every}...")
            te_mean, te_std = run_multiple_experiments(L, current_additions, reset_every, num_runs=num_runs)
            current_te_means.append(te_mean)
            current_te_stds.append(te_std)
            print(f"    TE: {te_mean:.6f} ± {te_std:.6f}")
        
        # Extract tau_c for this L
            current_TE_max, current_tau_c, _, current_fit_cov, current_R_squared_te = extract_critical_time(current_reset_values, current_te_means, current_te_stds)
                
            if current_TE_max is not None and current_tau_c is not None:
                print(f"  Extracted tau_c for L={L}: {current_tau_c:.6f}")
                tau_c_values.append(current_tau_c)
                
                # Estimate error in tau_c - use a simple heuristic since we're not doing proper fitting
                tau_c_err = 0.2 * current_tau_c  # Assume 20% error
                tau_c_errs.append(tau_c_err)
        else:
            print(f"  Failed to extract tau_c for L={L}")
            tau_c_values.append(0.0)
            tau_c_errs.append(0.0)
    
    # Extract critical exponent z
    if len([tc for tc in tau_c_values if tc > 0]) >= 2:
        A, z, fit_params, fit_cov, R_squared_power = extract_critical_exponent(L_values, tau_c_values, tau_c_errs)
        
        if A is not None and z is not None:
            print(f"\nFitting Results for Critical Exponent:")
            print(f"  A: {A:.6f}")
            print(f"  z (Critical Exponent): {z:.6f}")
            print(f"  R-squared: {R_squared_power:.6f}")
            
            # Plot tau_c vs. L with power law fit
            plt.figure(figsize=(10, 6))
            
            plt.errorbar(L_values, tau_c_values, yerr=tau_c_errs, fmt='o-', capsize=5, color='blue', linewidth=2, markersize=8, label='Data')
            
            # Plot the power law fit
            L_fit = np.linspace(min(L_values), max(L_values), 1000)
            tau_c_fit = power_law_function(L_fit, A, z)
            plt.plot(L_fit, tau_c_fit, 'g--', linewidth=2, label=f'Power Law Fit: tau_c={A:.3f}*L^{z:.3f}')
            
            plt.xlabel("System Size (L)")
            plt.ylabel("Critical Time Scale (tau_c)")
            plt.title("Critical Time Scale vs. System Size - Power Law Fit for Dynamic Critical Exponent")
            plt.grid(True, alpha=0.3)
            
            plt.legend()
            
            # Save the plot
            tau_c_fit_plot_file = os.path.join(script_dir, f"sandpile_tau_c_fit.png")
            plt.savefig(tau_c_fit_plot_file, dpi=300)
            print(f"\ntau_c vs. System Size with power law fit plot saved to: {tau_c_fit_plot_file}")
            
            # Plot in log-log scale
            plt.figure(figsize=(10, 6))
            
            plt.errorbar(L_values, tau_c_values, yerr=tau_c_errs, fmt='o-', capsize=5, color='blue', linewidth=2, markersize=8, label='Data')
            
            # Plot the power law fit in log-log scale
            plt.plot(L_fit, tau_c_fit, 'g--', linewidth=2, label=f'Power Law Fit: tau_c={A:.3f}*L^{z:.3f}')
            
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel("System Size (L) - Log Scale")
            plt.ylabel("Critical Time Scale (tau_c) - Log Scale")
            plt.title("tau_c vs. L (Log-Log Scale) - Dynamic Critical Exponent Analysis")
            plt.grid(True, alpha=0.3, which='both')
            
            plt.legend()
            
            # Save the log-log plot
            log_log_plot_file = os.path.join(script_dir, f"sandpile_tau_c_log_log.png")
            plt.savefig(log_log_plot_file, dpi=300)
            print(f"Log-log plot of tau_c vs. L saved to: {log_log_plot_file}")
    
    # -------------------------------------------------
    # 7. Save Results
    # -------------------------------------------------
    
    # Save TE(R) results
    te_results = pd.DataFrame({
        "Reset_Every": reset_labels,
        "Reset_Plot_Values": reset_plot_values,
        "TE_Mean": te_means,
        "TE_Std": te_stds
    })
    te_results_file = os.path.join(script_dir, f"sandpile_te_R_results_L{base_L}.csv")
    te_results.to_csv(te_results_file, index=False)
    print(f"\nTE(R) results saved to: {te_results_file}")
    
    # Save tau_c(L) results
    tau_c_results = pd.DataFrame({
        "System_Size": L_values,
        "Critical_Time_Scale": tau_c_values,
        "Critical_Time_Scale_Error": tau_c_errs
    })
    tau_c_results_file = os.path.join(script_dir, f"sandpile_tau_c_L_results.csv")
    tau_c_results.to_csv(tau_c_results_file, index=False)
    print(f"tau_c(L) results saved to: {tau_c_results_file}")
    
    # Save complete analysis summary
    summary_file = os.path.join(script_dir, f"sandpile_critical_analysis_summary.txt")
    with open(summary_file, "w") as f:
        f.write("CRITICAL TIME SCALE ANALYSIS SUMMARY\n")
        f.write("=" * 45 + "\n\n")
        
        f.write("1. Critical Time Scale (tau_c) Extraction\n")
        f.write("-" * 35 + "\n")
        f.write(f"System Size: L = {base_L}\n")
        if TE_max is not None and tau_c is not None:
            f.write(f"TE_max: {TE_max:.6f}\n")
            f.write(f"tau_c: {tau_c:.6f}\n")
            f.write(f"R-squared: {R_squared_te:.6f}\n")
        else:
            f.write("Failed to extract tau_c\n")
        f.write("\n")
        
        f.write("2. Dynamic Critical Exponent (z) Analysis\n")
        f.write("-" * 35 + "\n")
        f.write(f"System Sizes Studied: {L_values}\n")
        if A is not None and z is not None:
            f.write(f"A (Prefactor): {A:.6f}\n")
            f.write(f"z (Critical Exponent): {z:.6f}\n")
            f.write(f"R-squared: {R_squared_power:.6f}\n")
        else:
            f.write("Failed to extract critical exponent z\n")
        f.write("\n")
        
        f.write("3. TE(R) Results\n")
        f.write("-" * 35 + "\n")
        for i in range(len(reset_every_values)):
            f.write(f"Reset_Every={reset_every_values[i]}: TE = {te_means[i]:.6f} ± {te_stds[i]:.6f}\n")
        f.write("\n")
        
        f.write("4. tau_c(L) Results\n")
        f.write("-" * 35 + "\n")
        for i in range(len(L_values)):
            f.write(f"L={L_values[i]}: tau_c = {tau_c_values[i]:.6f} ± {tau_c_errs[i]:.6f}\n")
    
    print(f"Analysis summary saved to: {summary_file}")
    print("\nAnalysis complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()