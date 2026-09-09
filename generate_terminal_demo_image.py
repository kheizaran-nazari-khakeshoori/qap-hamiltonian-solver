import matplotlib.pyplot as plt
import textwrap

output = """$ python compare_sa_to_ground_state.py --size 10 --instance 1
Loaded instance: size=10, index=1
Brute-force max_n = 10

Exact ground state (brute-force):
---------------------------------
Best cost (ground state): 126.884331
Best permutation        : [0, 2, 6, 3, 9, 1, 7, 5, 4, 8]
Permutations evaluated  : 3628800

Simulated Annealing result:
---------------------------
Initial cost           : 453.969155
SA best cost           : 126.884331
SA final cost          : 126.884331
SA accepted moves      : 214/4500

Comparison:
-----------
Ground state cost      : 126.884331
SA best cost           : 126.884331
Difference (SA - exact): 0.000000
SA reached the exact ground state within numerical tolerance.

$ python pure_simulated_annealing.py
Pure Simulated Annealing (QAP)
Initial cost: 2927.000000
Final cost  : 2580.000000
Best cost   : 2580.000000
Accepted moves: 14/20000

$ python run_size10_groundstate_experiment.py  (excerpt for 12 instances)
Summary: mean gap 0.95, success rate 50% (6/12)
"""

fig, ax = plt.subplots(figsize=(8.2, 4.2))
ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.axis("off")
# dark terminal background
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")

# title bar
ax.text(0.02, 0.96, "● ● ●  Terminal — QAP Hamiltonian Solver Demo", color="#8b949e", fontsize=8, va="top", ha="left", family="monospace",
        bbox=dict(boxstyle="square,pad=0.3", fc="#21262d", ec="none"))

ax.text(0.02, 0.88, output, color="#c9d1d9", fontsize=7.2, va="top", ha="left", family="monospace", linespacing=1.45,
        bbox=dict(boxstyle="square,pad=0.4", fc="#0d1117", ec="none"))

plt.tight_layout(pad=0.5)
plt.savefig("plots/readme_terminal_demo.png", dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
print("saved plots/readme_terminal_demo.png")
plt.close()

# Also create a small architecture/banner image? optional - create workflow diagram via matplotlib
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(9, 2.8))
ax.set_xlim(0,10); ax.set_ylim(0,3)
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

boxes = [
    (0.5, 1.2, 1.6, 0.6, "Input\nF, D, p₀", "#DBEAFE", "#2563EB"),
    (2.6, 1.2, 1.6, 0.6, "QAP Objective\ncost(P)=ΣF·D", "#FEF3C7", "#D97706"),
    (4.7, 1.2, 1.7, 0.6, "Simulated\nAnnealing", "#DCFCE7", "#16A34A"),
    (6.9, 1.2, 1.6, 0.6, "Exact Solver\n(brute-force)", "#FCE7F3", "#DB2777"),
    (8.6, 1.35, 1.0, 0.35, "Report\n+ Plots", "#EDE9FE", "#7C3AED"),
]
for x,y,w,h,label,fc,ec in boxes:
    rect = patches.FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.05", facecolor=fc, edgecolor=ec, linewidth=1.2)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=7.5, weight="bold", color="#1f2937")

# arrows
for i in range(len(boxes)-1):
    x1 = boxes[i][0]+boxes[i][2]
    y1 = boxes[i][1]+boxes[i][3]/2
    x2 = boxes[i+1][0]
    y2 = boxes[i+1][1]+boxes[i+1][3]/2
    # if diverging after second box?
    if i==1:
        # split to SA and Exact
        ax.annotate("", xy=(4.7+0.85, 1.5), xytext=(4.2, 1.5), arrowprops=dict(arrowstyle="->", color="#6b7280", lw=1.3))
        ax.annotate("", xy=(6.9+0.8, 1.5), xytext=(6.4, 1.5), arrowprops=dict(arrowstyle="->", color="#6b7280", lw=1.3))
        # also arrow from input to objective
        ax.annotate("", xy=(2.6, 1.5), xytext=(2.1, 1.5), arrowprops=dict(arrowstyle="->", color="#6b7280", lw=1.3))
    elif i>1:
        ax.annotate("", xy=(x2, y2), xytext=(x1+0.1, y1), arrowprops=dict(arrowstyle="->", color="#6b7280", lw=1.3, linestyle="dashed"))

# extra arrow from objective to next stage label handling already done
# vertical annotation for comparison
ax.text(6.2, 0.55, "Ground truth for N≤10  →  gap = SA best − optimum", ha="center", fontsize=7, color="#6b7280", style="italic")
ax.text(5.0, 2.35, "QAP Hamiltonian Solver  —  Simulated Annealing vs Exact Ground State", ha="center", fontsize=9, weight="bold", color="#111827")
fig.tight_layout(pad=0.4)
plt.savefig("plots/readme_workflow.png", dpi=180, bbox_inches="tight")
print("saved plots/readme_workflow.png")
plt.close()
