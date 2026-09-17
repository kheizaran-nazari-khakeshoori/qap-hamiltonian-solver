"""Run ground-state vs SA experiment for all 100 size-10 instances.

This script:
- Loads instances/10-k.npz for k=1..100
- Computes the exact QAP ground state via brute-force
- Runs Simulated Annealing 100 times per instance with the same
  schedule used in batch experiments (calculation.py keeps trials=5
  for speed across 700 instances; here we use 100 for rigorous N=10 stats)
- Writes a CSV summarizing gaps and success for each instance

Usage (from project root):

    python run_size10_groundstate_experiment.py

Make sure you have already generated the instances via:

    python generate_instances.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from exact_qap_ground_state import brute_force_ground_state
from pure_simulated_annealing import pure_simulated_annealing
from hemiltonian_energy import qap_cost


INSTANCES_DIR = Path("instances")
SIZE = 10
INSTANCES_PER_SIZE = 100
TRIALS = 100
OUTPUT_CSV = Path("size10_ground_vs_sa.csv")


def main() -> None:
	if not INSTANCES_DIR.is_dir():
		raise SystemExit(
			f"Instances directory '{INSTANCES_DIR}' not found. "
			"Run generate_instances.py first."
		)

	fieldnames = [
		"size",
		"instance",
		"ground_cost",
		"sa_initial_cost",
		"sa_best_cost",
		"sa_avg_best_cost",
		"sa_final_cost",
		"sa_steps",
		"sa_accepted_moves",
		"sa_attempted_moves",
		"trials",
		"prob_solved",
		"gap_sa_minus_ground",
		"sa_reached_ground",  # 1 if best-of-100 == ground within tolerance
	]

	with OUTPUT_CSV.open("w", newline="") as f_out:
		writer = csv.DictWriter(f_out, fieldnames=fieldnames)
		writer.writeheader()

		for inst_idx in range(1, INSTANCES_PER_SIZE + 1):
			fname = INSTANCES_DIR / f"{SIZE}-{inst_idx}.npz"
			if not fname.is_file():
				raise SystemExit(f"Missing instance file: {fname}. Generate it first.")

			data = np.load(fname)
			if "F" not in data or "D" not in data:
				raise SystemExit(f"File {fname} must contain 'F' and 'D' arrays.")

			F = np.asarray(data["F"], dtype=float)
			D = np.asarray(data["D"], dtype=float)

			if F.shape != (SIZE, SIZE) or D.shape != (SIZE, SIZE):
				raise SystemExit(f"File {fname} has inconsistent shapes: F={F.shape}, D={D.shape}.")

			# Exact ground state for this instance.
			ground = brute_force_ground_state(F, D, max_n=SIZE)

			# 100 SA runs, same schedule as calculation.run_all_calculations_bundle.
			# Different start (np.roll) + seed per trial, like the 5-trial bundle.
			n = F.shape[0]
			p0 = np.arange(n, dtype=int)
			sa_steps = max(1_500, min(8_000, n * 450))

			best_costs = np.empty(TRIALS, dtype=float)
			accepted_sum = 0
			for trial in range(TRIALS):
				result = pure_simulated_annealing(
					p0=np.roll(p0, trial),
					F=F,
					D=D,
					initial_temp=3.5,
					cooling_rate=0.998,
					steps=sa_steps,
					seed=42 + trial,
				)
				best_costs[trial] = float(result.best_cost)
				accepted_sum += int(result.accepted_moves)
				if trial == 0:
					final_cost_first = float(result.final_cost)

			initial_cost = float(qap_cost(F, D, p0))
			best_of_100 = float(np.min(best_costs))
			avg_best = float(np.mean(best_costs))
			solved = int(np.sum(best_costs <= ground.best_cost + 1e-9))
			prob_solved = solved / float(TRIALS)
			gap = float(best_of_100 - ground.best_cost)
			success = 1 if abs(gap) <= 1e-9 else 0

			row = {
				"size": SIZE,
				"instance": inst_idx,
				"ground_cost": float(ground.best_cost),
				"sa_initial_cost": float(initial_cost),
				"sa_best_cost": float(best_of_100),
				"sa_avg_best_cost": float(avg_best),
				"sa_final_cost": float(final_cost_first),
				"sa_steps": float(sa_steps),
				"sa_accepted_moves": float(accepted_sum / float(TRIALS)),
				"sa_attempted_moves": float(sa_steps),
				"trials": float(TRIALS),
				"prob_solved": float(prob_solved),
				"gap_sa_minus_ground": float(gap),
				"sa_reached_ground": float(success),
			}
			writer.writerow(row)

	print(f"Wrote results for size={SIZE}, {INSTANCES_PER_SIZE} instances to {OUTPUT_CSV}")


if __name__ == "__main__":
	main()
