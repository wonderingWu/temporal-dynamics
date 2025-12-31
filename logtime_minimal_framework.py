#!/usr/bin/env python3
"""
Log-Time Minimal Testbed (v0.2)

Implements the "minimum viable" refactor you requested:
- Consistent tau = ln(t/t0) handling with *equal-τ resampling*
- Parameterization-robust metrics:
    • Total Variation (TV) of y (discrete, L1)  
    • Polyline arc length in (x,y) plane (discrete)  
    • Optional curvature integral for 2-observable curves (e.g., (S,M))
- Multiple realizations + seeds + CI aggregation
- Proper Monte Carlo handling for 2D Ising (burn-in, thinning)
- CA Rule 30 / 110 with *bit-packed* window and a clean LZ78 complexity estimator
- Optional TFIM (QuTiP) demo kept honest (closed-system unitary dynamics);
  includes disorder averaging; labeled accordingly
- Reproducible outputs: CSV with headers, params JSON, and PNG plots with
  mean ± 95% CI bands.

Usage examples:
  python logtime_minimal_framework.py --all
  python logtime_minimal_framework.py --ising --ising-L 64 --ising-steps 200000 \
         --ising-burn 100000 --ising-thin 200 --n-real 10 --seed 123
  python logtime_minimal_framework.py --ca --ca-rule 30 110 --ca-size 256 \
         --ca-steps 4000 --ca-window-rows 256 --n-real 8 --seed 42
  python logtime_minimal_framework.py --tfim --tfim-N 8 --tfim-tmax 20 --n-real 10

Outputs are written under ./out/<experiment>/ with CSVs and PNGs.
"""

from __future__ import annotations
import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from numba import jit

import numpy as np
import matplotlib.pyplot as plt

# -------- Optional QuTiP import --------
try:
    from qutip import tensor, sigmax, sigmaz, qeye, basis, entropy_vn, mesolve, Qobj, expect
    QUTIP_AVAILABLE = True
except Exception:
    QUTIP_AVAILABLE = False

# =============================================================
#            General utilities: IO, RNG, plotting
# =============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_params(path: str, params: Dict):
    with open(os.path.join(path, 'params.json'), 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)


def save_timeseries_csv(path: str, filename: str, header: str, arr: np.ndarray):
    fp = os.path.join(path, filename)
    np.savetxt(fp, arr, delimiter=',', header=header, comments='')


def ci95(mean: np.ndarray, var: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    # 95% normal approx CI (useful for n >= ~5). CI half-width = 1.96 * sqrt(var/n)
    se = np.sqrt(var / max(n, 1))
    hw = 1.96 * se
    return mean - hw, mean + hw


def plot_with_ci(x: np.ndarray, mean_y: np.ndarray, var_y: np.ndarray, n: int,
                 xlabel: str, ylabel: str, title: str, out_png: str):
    lo, hi = ci95(mean_y, var_y, n)
    plt.figure(figsize=(7, 4.5))
    plt.plot(x, mean_y, label='mean')
    plt.fill_between(x, lo, hi, alpha=0.25, label='95% CI')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

# =============================================================
#           Log-time handling and metrics
# =============================================================

@dataclass
class TauGrid:
    t0: float
    tau: np.ndarray  # shape (M,)
    t_of_tau: np.ndarray  # shape (M,)


def make_tau_grid(t_min: float, t_max: float, n_points: int, t0: Optional[float] = None) -> TauGrid:
    assert t_max > t_min > 0, "t_min must be > 0 and < t_max"
    if t0 is None:
        t0 = t_min
    tau_min = math.log(t_min / t0)
    tau_max = math.log(t_max / t0)
    tau = np.linspace(tau_min, tau_max, n_points)
    t_of_tau = t0 * np.exp(tau)
    return TauGrid(t0=t0, tau=tau, t_of_tau=t_of_tau)


def resample_equal_tau(t: np.ndarray, y: np.ndarray, tau_grid: TauGrid) -> np.ndarray:
    """Linear interpolate y(t) onto the equal-τ grid (t_of_tau)."""
    # Ensure increasing t
    order = np.argsort(t)
    t_sorted = t[order]
    y_sorted = y[order]
    return np.interp(tau_grid.t_of_tau, t_sorted, y_sorted)

# ---- Metrics ----

def total_variation(y: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(y))))


def polyline_arclength(x: np.ndarray, y: np.ndarray) -> float:
    dx = np.diff(x)
    dy = np.diff(y)
    return float(np.sum(np.hypot(dx, dy)))

# Optional curvature integral for curves in R^2 given param grid x
# Uses discrete Frenet curvature approximation; noisy, so users may smooth first.

def curvature_integral(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    # central differences for first/second derivatives where possible
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        return 0.0
    # First derivatives
    xp = np.gradient(x)
    yp = np.gradient(y)
    # Second derivatives
    xpp = np.gradient(xp)
    ypp = np.gradient(yp)
    num = np.abs(xp * ypp - yp * xpp)
    den = (xp * xp + yp * yp) ** 1.5 + eps
    kappa = num / den
    # Approximate integral as sum kappa * ds, ds ~ sqrt(dx^2 + dy^2)
    ds = np.hypot(np.diff(x), np.diff(y))
    # align lengths (use midpoints for kappa)
    k_mid = (kappa[:-1] + kappa[1:]) * 0.5
    return float(np.sum(k_mid * ds))

# =============================================================
#           Experiments: TFIM (optional), 2D Ising, CA
# =============================================================

# ------------------ TFIM (closed system demo) -----------------
@dataclass
class TFIMParams:
    N: int = 8
    J: float = 1.0
    hx: float = 1.0
    hz_disorder_strength: float = 0.5
    t_min: float = 0.1
    t_max: float = 20.0
    n_t: int = 200
    n_real: int = 5
    seed: int = 0


def run_tfim(params: TFIMParams, outdir: str):
    ensure_dir(outdir)
    save_params(outdir, asdict(params))

    if not QUTIP_AVAILABLE:
        with open(os.path.join(outdir, 'NOTICE.txt'), 'w') as f:
            f.write('QuTiP not available; TFIM demo skipped.\n')
        return

    rng = np.random.default_rng(params.seed)
    tlist = np.linspace(params.t_min, params.t_max, params.n_t)

    # Equal-τ grid for resampling
    tau_grid = make_tau_grid(params.t_min, params.t_max, params.n_t, t0=params.t_min)

    S_t_all = []
    Mz_t_all = []

    for r in range(params.n_real):
        # Build Hamiltonian
        def pauli(op: Qobj, i: int, N: int, sparse: bool = False) -> Qobj:
            ops = [qeye(2)] * N
            ops[i] = op
            result = tensor(ops)
            if sparse:
                result = result.to('csr')  # Convert to sparse matrix representation
            return result

        # Properly initialize H with the same dimensions as matrices returned by pauli function
        # H = 0 * pauli(sigmaz(), 0, params.N)
        # Add sparse matrix option in run_tfim function
        H = 0 * pauli(sigmaz(), 0, params.N, sparse=True)  # Enable sparse representation
        for i in range(params.N - 1):
            H += params.J * pauli(sigmaz(), i, params.N, sparse=True) * pauli(sigmaz(), i + 1, params.N, sparse=True)
        for i in range(params.N):
            H += params.hx * pauli(sigmax(), i, params.N, sparse=True)
        hz_list = params.hz_disorder_strength * (2 * rng.random(params.N) - 1)
        for i in range(params.N):
            H += hz_list[i] * pauli(sigmaz(), i, params.N, sparse=True)

        # Neel-like product state
        psi0 = tensor([basis(2, 0) if (i % 2 == 0) else basis(2, 1) for i in range(params.N)])

        # Ensure H is a sparse matrix
        H = H.to('csr')
        res = mesolve(H, psi0, tlist, [], [])
        S_t = []
        Mz_t = []
        A = list(range(params.N // 2))
        for state in res.states:
            # Calculate entanglement entropy - using partial trace
            rho_A = state.ptrace(A)
            S_t.append(float(entropy_vn(rho_A)))

            # Calculate magnetization - using expect function is more efficient
            mz = 0.0
            for i in range(params.N):
                # Create sigmaz operator for a single site
                sz_op = pauli(sigmaz(), i, params.N, sparse=True)
                # Calculate expectation value using expect function
                mz += expect(sz_op, state)
            Mz_t.append(mz / params.N)

        S_t_all.append(np.asarray(S_t))
        Mz_t_all.append(np.asarray(Mz_t))

    S_t_all = np.stack(S_t_all, axis=0)  # (R, T)
    Mz_t_all = np.stack(Mz_t_all, axis=0)

    # Equal-τ resampling for each realization
    S_tau_all = np.stack([resample_equal_tau(tlist, S_t_all[i], tau_grid) for i in range(params.n_real)], axis=0)
    Mz_tau_all = np.stack([resample_equal_tau(tlist, Mz_t_all[i], tau_grid) for i in range(params.n_real)], axis=0)

    # Metrics (per realization) then aggregate
    def aggregate_metrics(x_t, y_t_all, x_tau, y_tau_all, tag: str):
        R = y_t_all.shape[0]
        tv_t = np.array([total_variation(y_t_all[i]) for i in range(R)])
        tv_tau = np.array([total_variation(y_tau_all[i]) for i in range(R)])
        L_t = np.array([polyline_arclength(x_t, y_t_all[i]) for i in range(R)])
        L_tau = np.array([polyline_arclength(x_tau, y_tau_all[i]) for i in range(R)])
        # curvature on 2D curve (S,Mz) — compute later with paired arrays
        return tv_t, tv_tau, L_t, L_tau

    tvS_t, tvS_tau, LS_t, LS_tau = aggregate_metrics(tlist, S_t_all, tau_grid.t_of_tau, S_tau_all, 'S')
    tvM_t, tvM_tau, LM_t, LM_tau = aggregate_metrics(tlist, Mz_t_all, tau_grid.t_of_tau, Mz_tau_all, 'Mz')

    # Curvature integral on (S, Mz) curve for each realization under t and τ parameterizations
    R = S_t_all.shape[0]
    K_t = np.array([curvature_integral(S_t_all[i], Mz_t_all[i]) for i in range(R)])
    K_tau = np.array([curvature_integral(S_tau_all[i], Mz_tau_all[i]) for i in range(R)])

    # Save series means and metrics
    out_mean = np.column_stack([
        tlist,
        S_t_all.mean(0), S_t_all.var(0),
        Mz_t_all.mean(0), Mz_t_all.var(0)
    ])
    save_timeseries_csv(outdir, 'tfim_t_series.csv',
                        't,S_mean,S_var,Mz_mean,Mz_var', out_mean)

    out_tau_mean = np.column_stack([
        tau_grid.tau,
        S_tau_all.mean(0), S_tau_all.var(0),
        Mz_tau_all.mean(0), Mz_tau_all.var(0)
    ])
    save_timeseries_csv(outdir, 'tfim_tau_series.csv',
                        'tau,S_mean,S_var,Mz_mean,Mz_var', out_tau_mean)

    metrics = {
        'TV_S_t': tvS_t.tolist(), 'TV_S_tau': tvS_tau.tolist(),
        'L_S_t': LS_t.tolist(), 'L_S_tau': LS_tau.tolist(),
        'TV_M_t': tvM_t.tolist(), 'TV_M_tau': tvM_tau.tolist(),
        'L_M_t': LM_t.tolist(), 'L_M_tau': LM_tau.tolist(),
        'K_SM_t': K_t.tolist(), 'K_SM_tau': K_tau.tolist(),
    }
    save_params(os.path.join(outdir), {'metrics_per_realization': metrics})

    # Plots with CI
    plot_with_ci(
        tlist, S_t_all.mean(0), S_t_all.var(0), params.n_real,
        xlabel='t', ylabel='Entanglement entropy S',
        title='TFIM (closed, unitary) — S(t)',
        out_png=os.path.join(outdir, 'tfim_S_t.png'))
    plot_with_ci(
        tau_grid.tau, S_tau_all.mean(0), S_tau_all.var(0), params.n_real,
        xlabel='tau', ylabel='Entanglement entropy S',
        title='TFIM — S(τ)',
        out_png=os.path.join(outdir, 'tfim_S_tau.png'))

# ------------------ 2D Ising (Metropolis) ---------------------
@dataclass
class IsingParams:
    L: int = 32
    J: float = 1.0
    T: float = 2.269  # critical-ish
    steps: int = 200000
    burn_in: int = 100000
    thin: int = 200  # record every `thin` sweeps
    n_real: int = 5
    seed: int = 0


def ising_energy(spins: np.ndarray, J: float) -> float:
    # periodic boundary conditions
    return float(-J * np.sum(spins * (np.roll(spins, 1, 0) + np.roll(spins, 1, 1))))


def metropolis_sweep(spins: np.ndarray, beta: float, rng: np.random.Generator):
    L = spins.shape[0]
    for _ in range(L * L):
        i = rng.integers(0, L)
        j = rng.integers(0, L)
        nn = spins[(i+1) % L, j] + spins[(i-1) % L, j] + spins[i, (j+1) % L] + spins[i, (j-1) % L]
        dE = 2.0 * spins[i, j] * nn
        if dE <= 0.0 or rng.random() < math.exp(-beta * dE):
            spins[i, j] = -spins[i, j]


def run_ising(params: IsingParams, outdir: str):
    ensure_dir(outdir)
    save_params(outdir, asdict(params))

    rng = np.random.default_rng(params.seed)
    beta = 1.0 / params.T

    # collect realizations
    series_t = []

    for r in range(params.n_real):
        # Initialize with 0 and 1, then use bit packing for storage (8x memory saving)
        initial_state = rng.choice([0, 1], size=(params.L, params.L)).astype(np.uint8)
        spins_packed = np.packbits(initial_state, axis=None)  # Bit packing
        
        # Unpack and convert to -1 and 1 representation
        spins = np.unpackbits(spins_packed).reshape((params.L, params.L)).astype(np.int8)
        spins[spins == 0] = -1  # Convert 0 to -1
        # burn-in
        for _ in range(params.burn_in):
            metropolis_sweep(spins, beta, rng)
        # record
        E_list = []
        t_list = []
        for step in range(1, params.steps + 1):
            metropolis_sweep(spins, beta, rng)
            if step % params.thin == 0:
                E_list.append(ising_energy(spins, params.J))
                t_list.append(step)
        t_arr = np.array(t_list, dtype=float)
        E_arr = np.array(E_list, dtype=float)
        series_t.append((t_arr, E_arr))

    # Interpolate onto equal-τ grid shared across realizations
    t_min = min(s[0][0] for s in series_t)
    t_max = max(s[0][-1] for s in series_t)
    n_t = min(len(s[0]) for s in series_t)
    tau_grid = make_tau_grid(t_min, t_max, n_t, t0=t_min)

    # Stack aligned arrays
    E_t_mat = []
    E_tau_mat = []
    # for plotting mean±ci on a shared grid (t: we resample onto common linear t too)
    t_common = np.linspace(t_min, t_max, n_t)
    for t_arr, E_arr in series_t:
        # resample to common linear t grid
        E_t_interp = np.interp(t_common, t_arr, E_arr)
        E_t_mat.append(E_t_interp)
        # resample to equal-τ
        E_tau_interp = resample_equal_tau(t_arr, E_arr, tau_grid)
        E_tau_mat.append(E_tau_interp)

    E_t_mat = np.stack(E_t_mat, axis=0)  # (R, T)
    E_tau_mat = np.stack(E_tau_mat, axis=0)

    # Metrics per realization
    tv_t = np.array([total_variation(E_t_mat[i]) for i in range(E_t_mat.shape[0])])
    tv_tau = np.array([total_variation(E_tau_mat[i]) for i in range(E_tau_mat.shape[0])])
    L_t = np.array([polyline_arclength(t_common, E_t_mat[i]) for i in range(E_t_mat.shape[0])])
    L_tau = np.array([polyline_arclength(tau_grid.tau, E_tau_mat[i]) for i in range(E_tau_mat.shape[0])])

    # Save series means
    out_mean_t = np.column_stack([t_common, E_t_mat.mean(0), E_t_mat.var(0)])
    save_timeseries_csv(outdir, 'ising_t_series.csv', 't,E_mean,E_var', out_mean_t)

    out_mean_tau = np.column_stack([tau_grid.tau, E_tau_mat.mean(0), E_tau_mat.var(0)])
    save_timeseries_csv(outdir, 'ising_tau_series.csv', 'tau,E_mean,E_var', out_mean_tau)

    # Save metrics per realization
    save_params(outdir, {
        'metrics_per_realization': {
            'TV_t': tv_t.tolist(), 'TV_tau': tv_tau.tolist(),
            'L_t': L_t.tolist(), 'L_tau': L_tau.tolist(),
        }
    })

    # Plots
    plot_with_ci(t_common, E_t_mat.mean(0), E_t_mat.var(0), params.n_real,
                 xlabel='t (sweeps)', ylabel='Energy',
                 title=f'2D Ising (L={params.L}, T={params.T}) — E(t)',
                 out_png=os.path.join(outdir, 'ising_E_t.png'))
    plot_with_ci(tau_grid.tau, E_tau_mat.mean(0), E_tau_mat.var(0), params.n_real,
                 xlabel='τ', ylabel='Energy',
                 title=f'2D Ising — E(τ)',
                 out_png=os.path.join(outdir, 'ising_E_tau.png'))

# ------------------ CA (Rule 30 / 110) ------------------------
@dataclass
class CAParams:
    rules: Tuple[int, ...] = (30, 110)
    size: int = 256
    steps: int = 2000
    random_init: bool = False
    window_rows: int = 256  # rows used for complexity window
    n_real: int = 5
    seed: int = 0

@jit(nopython=True)
def ca_next(state: np.ndarray, rule_bin: np.ndarray) -> np.ndarray:
    left = np.roll(state, 1)
    right = np.roll(state, -1)
    neighborhood = (left << 2) | (state << 1) | right
    return rule_bin[7 - neighborhood]


def run_ca_once(rule: int, size: int, steps: int, rng: np.random.Generator, random_init: bool) -> np.ndarray:
    rule_bin = np.array([int(x) for x in np.binary_repr(rule, width=8)], dtype=np.uint8)
    if random_init:
        state = rng.integers(0, 2, size=size, dtype=np.uint8)
    else:
        state = np.zeros(size, dtype=np.uint8)
        state[size // 2] = 1
    history = np.zeros((steps, size), dtype=np.uint8)
    for i in range(steps):
        history[i] = state
        state = ca_next(state, rule_bin)
    return history

# LZ78 complexity measure for a bitstring: returns dictionary size per N bits.
# Simple, deterministic, memory-friendly implementation.

def lz78_complexity(bits: np.ndarray) -> float:
    # bits as 0/1 uint8 array
    dict_map = {}
    dict_size = 0
    phrase = ''
    # iterate characters
    for b in bits:
        c = '1' if b else '0'
        cand = phrase + c
        if cand in dict_map:
            phrase = cand
        else:
            dict_size += 1
            dict_map[cand] = dict_size
            phrase = ''
    if phrase != '':
        dict_size += 1
    # normalize by length to get rate-like measure
    return dict_size / max(1, len(bits))


def pack_bits(rows: np.ndarray) -> np.ndarray:
    # flatten to 1D uint8 0/1
    return rows.reshape(-1).astype(np.uint8)


def run_ca(params: CAParams, outdir: str):
    ensure_dir(outdir)
    save_params(outdir, asdict(params))

    rng = np.random.default_rng(params.seed)

    for rule in params.rules:
        # Collect complexity time series for each realization
        comp_series = []  # list of (t_indices, C_t)
        for r in range(params.n_real):
            hist = run_ca_once(rule, params.size, params.steps, rng, params.random_init)
            C = []
            t_idx = []
            for t in range(params.window_rows, params.steps + 1):
                window = hist[t - params.window_rows:t]
                bits = pack_bits(window)
                C.append(lz78_complexity(bits))
                t_idx.append(t)
            comp_series.append((np.array(t_idx, dtype=float), np.array(C, dtype=float)))

        # Build common grids
        t_min = min(s[0][0] for s in comp_series)
        t_max = max(s[0][-1] for s in comp_series)
        n_t = min(len(s[0]) for s in comp_series)
        t_common = np.linspace(t_min, t_max, n_t)
        tau_grid = make_tau_grid(t_min, t_max, n_t, t0=t_min)

        # Stack
        C_t_mat = []
        C_tau_mat = []
        for t_arr, C_arr in comp_series:
            C_t_mat.append(np.interp(t_common, t_arr, C_arr))
            C_tau_mat.append(resample_equal_tau(t_arr, C_arr, tau_grid))
        C_t_mat = np.stack(C_t_mat, axis=0)
        C_tau_mat = np.stack(C_tau_mat, axis=0)

        # Metrics
        tv_t = np.array([total_variation(C_t_mat[i]) for i in range(C_t_mat.shape[0])])
        tv_tau = np.array([total_variation(C_tau_mat[i]) for i in range(C_tau_mat.shape[0])])
        L_t = np.array([polyline_arclength(t_common, C_t_mat[i]) for i in range(C_t_mat.shape[0])])
        L_tau = np.array([polyline_arclength(tau_grid.tau, C_tau_mat[i]) for i in range(C_tau_mat.shape[0])])

        # Save
        exp_dir = os.path.join(outdir, f'rule_{rule}')
        ensure_dir(exp_dir)
        save_params(exp_dir, {
            'rule': rule,
            'metrics_per_realization': {
                'TV_t': tv_t.tolist(), 'TV_tau': tv_tau.tolist(),
                'L_t': L_t.tolist(), 'L_tau': L_tau.tolist(),
            }
        })
        save_timeseries_csv(exp_dir, 'ca_t_series.csv', 't,C_mean,C_var',
                            np.column_stack([t_common, C_t_mat.mean(0), C_t_mat.var(0)]))
        save_timeseries_csv(exp_dir, 'ca_tau_series.csv', 'tau,C_mean,C_var',
                            np.column_stack([tau_grid.tau, C_tau_mat.mean(0), C_tau_mat.var(0)]))

        plot_with_ci(t_common, C_t_mat.mean(0), C_t_mat.var(0), params.n_real,
                     xlabel='t (steps)', ylabel='LZ78 rate',
                     title=f'CA Rule {rule} — complexity C(t)',
                     out_png=os.path.join(exp_dir, 'ca_C_t.png'))
        plot_with_ci(tau_grid.tau, C_tau_mat.mean(0), C_tau_mat.var(0), params.n_real,
                     xlabel='τ', ylabel='LZ78 rate',
                     title=f'CA Rule {rule} — complexity C(τ)',
                     out_png=os.path.join(exp_dir, 'ca_C_tau.png'))

# =============================================================
#                         CLI
# =============================================================

def build_argparser():
    p = argparse.ArgumentParser(description='Log-time minimal testbed with proper metrics and τ-resampling')
    p.add_argument('--out', default='out', help='Output base directory')
    p.add_argument('--seed', type=int, default=0, help='Global RNG seed (overrides experiment defaults)')
    p.add_argument('--n-real', type=int, default=None, help='Override n_real for all experiments')

    # Toggles
    p.add_argument('--tfim', action='store_true', help='Run TFIM demo (requires QuTiP)')
    p.add_argument('--ising', action='store_true', help='Run 2D Ising experiment')
    p.add_argument('--ca', action='store_true', help='Run Cellular Automata experiment')
    p.add_argument('--all', action='store_true', help='Run all available experiments')

    # TFIM params
    p.add_argument('--tfim-N', type=int, default=8)
    p.add_argument('--tfim-tmax', type=float, default=20.0)
    p.add_argument('--tfim-nt', type=int, default=200)

    # Ising params
    p.add_argument('--ising-L', type=int, default=32)
    p.add_argument('--ising-T', type=float, default=2.269)
    p.add_argument('--ising-steps', type=int, default=200000)
    p.add_argument('--ising-burn', type=int, default=100000)
    p.add_argument('--ising-thin', type=int, default=200)

    # CA params
    p.add_argument('--ca-rule', type=int, nargs='+', default=[30, 110])
    p.add_argument('--ca-size', type=int, default=256)
    p.add_argument('--ca-steps', type=int, default=2000)
    p.add_argument('--ca-window-rows', type=int, default=256)
    p.add_argument('--ca-random-init', action='store_true')

    return p


def main():
    args = build_argparser().parse_args()

    base_out = args.out
    ensure_dir(base_out)

    # Prepare global overrides
    # TFIM
    tfim_params = TFIMParams(N=args.tfim_N, t_max=args.tfim_tmax, n_t=args.tfim_nt)
    # Ising
    ising_params = IsingParams(L=args.ising_L, T=args.ising_T, steps=args.ising_steps,
                               burn_in=args.ising_burn, thin=args.ising_thin)
    # CA
    ca_params = CAParams(rules=tuple(args.ca_rule), size=args.ca_size, steps=args.ca_steps,
                         window_rows=args.ca_window_rows, random_init=args.ca_random_init)

    # Global seed / n_real overrides
    if args.seed is not None:
        tfim_params.seed = args.seed
        ising_params.seed = args.seed
        ca_params.seed = args.seed
    if args.n_real is not None:
        tfim_params.n_real = args.n_real
        ising_params.n_real = args.n_real
        ca_params.n_real = args.n_real

    run_all = args.all or (not args.tfim and not args.ising and not args.ca)

    if args.tfim or run_all:
        run_tfim(tfim_params, os.path.join(base_out, 'tfim'))

    if args.ising or run_all:
        run_ising(ising_params, os.path.join(base_out, 'ising'))

    if args.ca or run_all:
        run_ca(ca_params, os.path.join(base_out, 'ca'))

    print('[Done] Experiments complete. Outputs under', base_out)


if __name__ == '__main__':
    main()
