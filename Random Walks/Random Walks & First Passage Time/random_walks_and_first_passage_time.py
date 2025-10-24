# -*- coding: utf-8 -*-
"""Random_Walks_and_first_passage_time.py
.
# -*- coding: utf-8 -*-

"""random_walks_and_first_passage_time.ipynb

Random Walks, First Passage Times, and Recurrence Phenomena
-----------------------------------------------------------
Author: Petros Agridos

Description:
This notebook provides a comprehensive numerical study of random walks in 1D, 2D, and 3D
with emphasis on first passage times (FPT) and recurrence properties.

The simulations explore key stochastic phenomena such as:
    - Mean first passage times and their scaling with system size (L)
    - Probability distributions of first passage times, including power-law tails
    - Dimensional dependence of recurrence: 1D and 2D (recurrent) vs 3D (transient)

Modules:
    1. One-Dimensional First Passage Time (FPT)
       - Simulation with adjustable bias and absorbing boundary
       - Distribution P(T) and scaling of ⟨T⟩ with L
    2. Mean First Passage Time vs System Size
       - Numerical estimation of ⟨T⟩ ∝ L² for unbiased walks
    3. Multi-Dimensional Recurrence and Transience
       - Random walks in 1D, 2D, and 3D
       - Comparison of return probabilities P_return(t)
    4. Unified Visualization
       - Histograms, log–log scaling plots, and recurrence comparison graphs

Dependencies:
    numpy, matplotlib, scipy, IPython.display

Usage:
    Adjust simulation parameters (number of walkers, max_steps, lattice size)
    at the top of the script for desired precision. Suitable for use in
    Jupyter notebooks or standalone Python scripts.

"""

# First Passage Time & Recurrence / Transience unified module
# Run in a Jupyter notebook cell (%matplotlib inline recommended)

# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from IPython.display import display, HTML

# ----------------------------
# Utility RNG seed
# ----------------------------
SEED = 42
np.random.seed(SEED)

# ----------------------------
# 1D FIRST PASSAGE TIME (absorbing at x = L)
# vectorized simulation for many walkers
# ----------------------------
def simulate_1d_fpt(L=50, p_right=0.5, n_walkers=10000, max_steps=10000):
    """
    Simulate n_walkers starting at x=0 on integer lattice.
    Each step: +1 with probability p_right, -1 otherwise.
    Absorbing boundary at x = L (right side).
    Returns:
      first_passage_times: array length n_walkers (np.nan if walker didn't hit L within max_steps)
      absorbed_fraction: fraction of walkers that hit L within max_steps
    """
    positions = np.zeros(n_walkers, dtype=np.int32)
    alive = np.ones(n_walkers, dtype=bool)   # those not yet absorbed
    fpt = np.full(n_walkers, np.nan)        # record first passage time
    for t in range(1, max_steps+1):
        # generate steps only for alive walkers
        r = np.random.rand(n_walkers)
        steps = np.where(r < p_right, 1, -1)
        positions[alive] += steps[alive]
        # check absorption (first time crossing exactly L or >= L)
        hit = alive & (positions >= L)
        if np.any(hit):
            fpt[hit] = t
            alive[hit] = False
        # early break if none alive
        if not alive.any():
            break
    absorbed_fraction = np.sum(~np.isnan(fpt)) / n_walkers
    return fpt, absorbed_fraction

# ----------------------------
# 1D: statistics & plots for FPT
# ----------------------------
def analyze_1d_fpt(L=50, p=0.5, n_walkers=5000, max_steps=20000, bins=100):
    fpt, frac = simulate_1d_fpt(L=L, p_right=p, n_walkers=n_walkers, max_steps=max_steps)
    hits = fpt[~np.isnan(fpt)]
    print(f"1D FPT: L={L}, p={p}, walkers={n_walkers}, absorbed fraction (within {max_steps} steps) = {frac:.4f}")
    if len(hits)==0:
        print("No absorptions observed within max_steps.")
        return fpt
    # Histogram P(T)
    plt.figure(figsize=(9,4))
    plt.subplot(1,2,1)
    plt.hist(hits, bins=bins, density=True, alpha=0.7, color='magenta')
    plt.xlabel('First passage time T (steps)')
    plt.ylabel('P(T)')
    plt.title('1D FPT histogram')
    plt.grid(True)
    # log-log tail plot to inspect power-law
    plt.subplot(1,2,2)
    counts, edges = np.histogram(hits, bins=bins)
    centers = 0.5*(edges[:-1]+edges[1:])
    nonzero = counts>0
    plt.loglog(centers[nonzero], counts[nonzero]/counts[nonzero].sum(), 'o', markersize=4)
    plt.xlabel('T (log)')
    plt.ylabel('P(T) (log)')
    plt.title('1D FPT (log-log)')
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    # mean first passage time (over absorbed walkers)
    mean_T = hits.mean()
    median_T = np.median(hits)
    print(f"Mean FPT (absorbed walkers) = {mean_T:.2f} steps, median = {median_T:.2f}")
    return fpt

# ----------------------------
# 1D: mean FPT vs L (scaling)
# ----------------------------
def mean_fpt_vs_L(L_values, p=0.5, n_walkers=2000, max_steps=20000):
    means = []
    frac_absorbed = []
    for L in L_values:
        fpt, frac = simulate_1d_fpt(L=int(L), p_right=p, n_walkers=n_walkers, max_steps=max_steps)
        hits = fpt[~np.isnan(fpt)]
        means.append(np.nan if len(hits)==0 else hits.mean())
        frac_absorbed.append(frac)
        print(f"L={L}: mean FPT = {means[-1]}, absorbed fraction = {frac:.3f}")
    L_values = np.array(L_values)
    means_arr = np.array(means, dtype=float)
    # Plot (log-log)
    plt.figure(figsize=(8,5))
    mask = ~np.isnan(means_arr)
    plt.loglog(L_values[mask], means_arr[mask], 'o-', color='magenta')
    plt.xlabel('L (log)')
    plt.ylabel('<T> (log)')
    plt.title('Mean First Passage Time vs L (log-log)')
    plt.grid(True, which='both', ls='--', alpha=0.6)
    plt.show()
    return L_values, means_arr, frac_absorbed

# ----------------------------
# 2D & 3D recurrence/transience simulation (lattice nearest-neighbor)
# We simulate many walkers up to max_steps and record the fraction that have
# returned to the origin at least once by time t -> P_return(t).
# ----------------------------
def simulate_recurrence(dim=2, n_walkers=2000, max_steps=5000):
    """
    dim = 1, 2 or 3
    returns:
      P_return: array length max_steps+1, P_return[t] fraction returned by time t (t=0..max_steps)
      first_return_times: array (n_walkers,) containing first return time (np.nan if never)
    """
    # positions arrays
    pos = np.zeros((n_walkers, dim), dtype=np.int32)
    returned = np.zeros(n_walkers, dtype=bool)
    first_return_time = np.full(n_walkers, np.nan)
    P_return = np.zeros(max_steps+1, dtype=float)
    # initial time t=0 -> everyone at origin => we consider return after leaving and coming back;
    # For recurrence test we typically consider first return after t>0. So initial P_return[0]=0.
    for t in range(1, max_steps+1):
        # make steps: choose axis and sign randomly
        # For each walker, choose axis 0..dim-1 and sign ±1
        axes = np.random.randint(0, dim, size=n_walkers)
        signs = np.where(np.random.rand(n_walkers) < 0.5, 1, -1)
        # update positions
        for ax in range(dim):
            mask = (axes == ax)
            if np.any(mask):
                pos[mask, ax] += signs[mask]
        # check return to origin (pos all zeros) for walkers not yet returned
        just_returned = (~returned) & np.all(pos == 0, axis=1)
        if np.any(just_returned):
            first_return_time[just_returned] = t
            returned[just_returned] = True
        P_return[t] = returned.mean()
    return P_return, first_return_time

# ----------------------------
# Compare recurrence 1D / 2D / 3D on same plot
# ----------------------------
def compare_recurrence(n_walkers=1000, max_steps=2000):
    prs = {}
    for dim in (1,2,3):
        print(f"Simulating dim={dim} ...")
        P_return, f_rt = simulate_recurrence(dim=dim, n_walkers=n_walkers, max_steps=max_steps)
        prs[dim] = P_return
    # plot
    plt.figure(figsize=(8,5))
    t = np.arange(len(P_return))
    plt.plot(t, prs[1], label='1D', color='blue')
    plt.plot(t, prs[2], label='2D', color='red')
    plt.plot(t, prs[3], label='3D', color='green')
    plt.xlabel('Time steps')
    plt.ylabel('P_return(t) (fraction returned by time t)')
    plt.title(f'Recurrence / Transience comparison (n_walkers={n_walkers})')
    plt.legend()
    plt.grid(True)
    plt.show()
    return prs

# ----------------------------
# Example runs (default parameters)
# ----------------------------
if __name__ == "__main__":
    # 1) 1D FPT example
    print("=== 1D First Passage Time example ===")
    fpt = analyze_1d_fpt(L=50, p=0.5, n_walkers=10000, max_steps=100000, bins=120)

    # 1D mean FPT vs L scaling
    print("\n=== Mean FPT vs L (1D) ===")
    L_vals = [10, 20, 40, 80, 160]
    mean_fpt_vs_L(L_vals, p=0.5, n_walkers=10000, max_steps=100000)

    # 2) Compare recurrence/transience (1D vs 2D vs 3D)
    print("\n=== Recurrence / Transience comparison ===")
    prs = compare_recurrence(n_walkers=10000, max_steps=100000)

    # 3) Analyze first-return time distribution in 2D (optional)
    # you can extract first return times from simulate_recurrence outputs for further histograms
    print("\nDone. Adjust parameters at the top of the file if you need more precision/longer runs.")

