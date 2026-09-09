"""Generate publication-quality demo plots for README.
Creates 4 images in plots/:
 - readme_qap_matrices.png : Flow & Distance matrices
 - readme_sa_trace.png : SA cost trace vs Metropolis step
 - readme_sa_vs_ground.png : SA best vs exact ground state (N=10)
 - readme_scaling.png : SA scaling with problem size
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from hemiltonian_energy import qap_cost
from pure_simulated_annealing import pure_simulated_annealing
from exact_qap_ground_state import brute_force_ground_state

PLOTS = Path("plots")
PLOTS.mkdir(exist_ok=True)

# Style
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

# 1. QAP matrices heatmap for N=10 instance 1 + coordinates
print("1/4 : QAP matrices")
data = np.load("instances/10-1.npz")
F = data["F"]
D = data["D"]
# Recover coordinates from D? Instead just plot matrices
fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
# Flow
im0 = axes[0].imshow(F, cmap="YlOrRd", interpolation="nearest")
axes[0].set_title("Flow matrix F (N=10, inst=1)")
axes[0].set_xlabel("Facility j"); axes[0].set_ylabel("Facility i")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(D, cmap="Blues", interpolation="nearest")
axes[1].set_title("Distance matrix D (Euclidean)")
axes[1].set_xlabel("Location j"); axes[1].set_ylabel("Location i")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

# Also plot permutation cost landscape sketch: random perms costs
rng = np.random.default_rng(0)
costs = []
for _ in range(2000):
    p = rng.permutation(10)
    costs.append(qap_cost(F, D, p))
axes[2].hist(costs, bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
# mark ground state
ground = brute_force_ground_state(F, D, max_n=10)
axes[2].axvline(ground.best_cost, color="red", linestyle="--", linewidth=1.5, label=f"Ground state {ground.best_cost:.0f}")
# mark identity
p0 = np.arange(10)
axes[2].axvline(qap_cost(F,D,p0), color="orange", linestyle=":", linewidth=1.5, label=f"Identity {qap_cost(F,D,p0):.0f}")
axes[2].set_title("Cost distribution (2000 random perms)")
axes[2].set_xlabel("QAP cost"); axes[2].set_ylabel("Count")
axes[2].legend()
fig.tight_layout()
fig.savefig(PLOTS/"readme_qap_matrices.png", bbox_inches="tight")
plt.close(fig)
print(f"  saved {PLOTS/'readme_qap_matrices.png'}")

# 2. SA trace vs step for different schedules
print("2/4 : SA trace")
p0 = np.arange(10)
steps = 4500
configs = [
    (3.5, 0.998, "Standard (T0=3.5, α=0.998)", "#4C72B0"),
    (10.0, 0.999, "Hot & slow (T0=10, α=0.999)", "#55A868"),
    (0.5, 0.995, "Cold & fast (T0=0.5, α=0.995)", "#C44E52"),
]
fig, ax = plt.subplots(figsize=(7, 3.6))
for T0, alpha, label, color in configs:
    res = pure_simulated_annealing(p0, F, D, initial_temp=T0, cooling_rate=alpha, steps=steps, seed=42)
    # also compute best-cost trace
    best_trace = np.minimum.accumulate(res.cost_trace)
    ax.plot(res.cost_trace, alpha=0.22, linewidth=0.7, color=color)
    ax.plot(best_trace, linewidth=1.6, color=color, label=label)
    # final best
    print(f"   {label}: best={res.best_cost:.1f} accepted={res.accepted_moves}/{res.attempted_moves}")
ax.axhline(ground.best_cost, color="black", linestyle="--", linewidth=1, label=f"Ground state ({ground.best_cost:.0f})")
ax.set_title("Simulated Annealing: Cost vs Metropolis Swap (N=10, inst=1)")
ax.set_xlabel("Metropolis step t")
ax.set_ylabel("QAP cost")
ax.grid(True, alpha=0.25)
ax.legend(ncol=2, framealpha=0.9)
fig.tight_layout()
fig.savefig(PLOTS/"readme_sa_trace.png", bbox_inches="tight")
plt.close(fig)
print(f"  saved {PLOTS/'readme_sa_trace.png'}")

# Also generate a second trace for N=50 to show scaling of convergence
data50 = np.load("instances/50-1.npz")
F50 = data50["F"]; D50 = data50["D"]
p050 = np.arange(50)
steps50 = min(8000, 50*450)
res50 = pure_simulated_annealing(p050, F50, D50, initial_temp=3.5, cooling_rate=0.998, steps=steps50, seed=42)
fig, ax = plt.subplots(figsize=(7, 3.0))
ax.plot(res50.cost_trace, color="#8C8C8C", alpha=0.35, linewidth=0.6, label="Current cost")
ax.plot(np.minimum.accumulate(res50.cost_trace), color="#4C72B0", linewidth=1.6, label="Best-so-far")
ax.set_title(f"SA trace for large instance (N=50, steps={steps50})  — best {res50.best_cost:.0f}")
ax.set_xlabel("Metropolis step t"); ax.set_ylabel("QAP cost")
ax.grid(True, alpha=0.25); ax.legend()
fig.tight_layout()
fig.savefig(PLOTS/"readme_sa_trace_N50.png", bbox_inches="tight")
plt.close(fig)
print(f"  saved {PLOTS/'readme_sa_trace_N50.png'}")

# 3. SA vs ground for ~12 instances (to keep runtime ~ 5 min)
print("3/4 : SA vs ground (12 instances, N=10)  ~ 5 min")
import time
INSTS = list(range(1, 13))
gaps = []
ground_costs = []
sa_costs = []
for idx in INSTS:
    d = np.load(f"instances/10-{idx}.npz")
    F_ = d["F"]; D_ = d["D"]
    t0 = time.time()
    g = brute_force_ground_state(F_, D_, max_n=10)
    sa = pure_simulated_annealing(np.arange(10), F_, D_, initial_temp=3.5, cooling_rate=0.998, steps=4500, seed=42)
    gaps.append(sa.best_cost - g.best_cost)
    ground_costs.append(g.best_cost)
    sa_costs.append(sa.best_cost)
    print(f"  inst {idx}: ground {g.best_cost:.1f} sa {sa.best_cost:.1f} gap {sa.best_cost-g.best_cost:.1f} t={time.time()-t0:.1f}s")

# Plot gap per instance + histogram
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
x = np.arange(len(INSTS))
axes[0].bar(x, gaps, color=["#55A868" if g < 1e-9 else "#C44E52" for g in gaps], edgecolor="white")
axes[0].axhline(0, color="black", linewidth=0.9)
axes[0].set_xticks(x); axes[0].set_xticklabels([str(i) for i in INSTS])
axes[0].set_title("SA gap to ground state (N=10)")
axes[0].set_xlabel("Instance"); axes[0].set_ylabel("Gap: SA best − ground")
success_rate = np.mean(np.array(gaps) <= 1e-9)
axes[0].text(0.98, 0.95, f"Success {success_rate:.0%} ({int(np.sum(np.array(gaps)<=1e-9))}/{len(gaps)})", ha="right", va="top", transform=axes[0].transAxes, bbox=dict(boxstyle="round", fc="white", alpha=0.8))

axes[1].scatter(ground_costs, sa_costs, s=65, color="#4C72B0", edgecolor="white", zorder=3)
mn = min(min(ground_costs), min(sa_costs)); mx = max(max(ground_costs), max(sa_costs))
axes[1].plot([mn,mx],[mn,mx],"k--", linewidth=1, label="y=x (optimal)")
axes[1].set_title("SA best cost vs exact ground state")
axes[1].set_xlabel("Ground-state cost"); axes[1].set_ylabel("SA best cost")
axes[1].legend(); axes[1].grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(PLOTS/"readme_sa_vs_ground.png", bbox_inches="tight")
plt.close(fig)
print(f"  saved {PLOTS/'readme_sa_vs_ground.png'}")

# 4. Scaling across sizes
print("4/4 : Scaling across sizes N=10..50")
from calculation import run_all_calculations_bundle
sizes = [10, 15, 20, 25, 50]
num_samples = 8  # per size
import csv, time
rows_best = {s: [] for s in sizes}
rows_time = {s: [] for s in sizes}
rows_residual = {s: [] for s in sizes}
for s in sizes:
    for inst in range(1, num_samples+1):
        d = np.load(f"instances/{s}-{inst}.npz")
        F_ = d["F"]; D_ = d["D"]
        perm_seed = np.arange(s, dtype=float)
        bundle = run_all_calculations_bundle(perm_seed, F_, D_)
        sa_metrics = bundle["metrics"]["Simulated Annealing"]
        rows_best[s].append(sa_metrics["best_energy"])
        rows_time[s].append(sa_metrics["time_taken"])
        rows_residual[s].append(sa_metrics["residual_energy"])
        print(f"  N={s} inst={inst} best={sa_metrics['best_energy']:.0f} time={sa_metrics['time_taken']:.3f}s residual={sa_metrics['residual_energy']:.1f}")

fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))
# Boxplot best cost
data_best = [rows_best[s] for s in sizes]
bp = axes[0].boxplot(data_best, positions=sizes, widths=3, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor("#AEC7E8")
axes[0].set_title("SA best cost vs N")
axes[0].set_xlabel("Problem size N"); axes[0].set_ylabel("Best cost")
axes[0].grid(True, alpha=0.25, axis="y")

data_time = [rows_time[s] for s in sizes]
axes[1].boxplot(data_time, positions=sizes, widths=3, patch_artist=True,
                boxprops=dict(facecolor="#FFBB78"))
# also mean
means = [np.mean(rows_time[s]) for s in sizes]
axes[1].plot(sizes, means, marker="o", color="#C44E52", linewidth=1.5, label="Mean")
axes[1].set_title("Runtime vs N (5 trials, SA)")
axes[1].set_xlabel("Problem size N"); axes[1].set_ylabel("Time (s)")
axes[1].grid(True, alpha=0.25, axis="y")
axes[1].legend()

# residual (vs QAP Objective reference inside bundle)
data_res = [rows_residual[s] for s in sizes]
axes[2].boxplot(data_res, positions=sizes, widths=3, patch_artist=True,
                boxprops=dict(facecolor="#98DF8A"))
axes[2].axhline(0, color="black", linestyle="--", linewidth=0.9)
axes[2].set_title("Residual vs N")
axes[2].set_xlabel("Problem size N"); axes[2].set_ylabel("Best − reference best")
axes[2].grid(True, alpha=0.25, axis="y")
fig.tight_layout()
fig.savefig(PLOTS/"readme_scaling.png", bbox_inches="tight")
plt.close(fig)
print(f"  saved {PLOTS/'readme_scaling.png'}")

print("All done.")
