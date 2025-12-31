import numpy as np
import matplotlib.pyplot as plt
import os

# Get absolute path of the directory containing the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Data Preparation
# Import csv data from out/ising folder using absolute path
t_data_path = os.path.join(current_dir, 'out', 'ising', 'ising_t_series.csv')
# Read file with utf-8 encoding and skip potential error lines
# Skip first header line
t_data = np.loadtxt(t_data_path, delimiter=',', encoding='utf-8', skiprows=1)
print(f"Successfully read ising_t_series.csv, data shape: {t_data.shape}")
t = t_data[:, 0]  # First column of t_data
t = np.atleast_2d(t).T  # Ensure it's a column vector
E_t = t_data[:, 1]  # Second column of t_data
var_t = t_data[:, 2]  # Third column of t_data

tau_data_path = os.path.join(current_dir, 'out', 'ising', 'ising_tau_series.csv')
# Read file with utf-8 encoding and skip potential error lines
# Skip first header line
tau_data = np.loadtxt(tau_data_path, delimiter=',', encoding='utf-8', skiprows=1)
print(f"Successfully read ising_tau_series.csv, data shape: {tau_data.shape}")
tau = tau_data[:, 0]  # First column of tau_data
tau = np.atleast_2d(tau).T  # Ensure it's a column vector
E_tau = tau_data[:, 1]  # Second column of tau_data
var_tau = tau_data[:, 2]  # Third column of tau_data
t_tau = 200 * np.exp(tau)  # Convert to actual steps

# Calculate final energy and convergence threshold from data
final_energy = E_t[-1]
convergence_threshold_t = var_t[-1]  # Assume last variance value as convergence threshold

# Create figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.subplots_adjust(hspace=0.3)

# Energy evolution comparison
ax1.plot(t, E_t, 'b-', label='Linear Time (t)')
ax1.plot(t_tau, E_tau, 'r--', label='Log Time (τ)')
ax1.axhline(y=final_energy, color='k', linestyle=':', alpha=0.5, label='Final Energy')
ax1.set_ylabel('Energy')
ax1.set_title('Energy Evolution Comparison')
ax1.legend()
ax1.grid(True)

# Variance evolution comparison
ax2.semilogy(t, var_t, 'b-', label='Var(t)')
ax2.semilogy(t_tau, var_tau, 'r--', label='Var(τ)')
ax2.axhline(y=convergence_threshold_t, color='g', linestyle='--', label='Convergence Threshold')
ax2.set_xlabel('Simulation Steps')
ax2.set_ylabel('Variance (log scale)')
ax2.set_title('Variance Evolution Comparison')
ax2.legend()
ax2.grid(True)

# Save figure
plot_save_path = os.path.join(current_dir, 'out', 'ising', 'comparison_plot.png')
plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
plt.close()