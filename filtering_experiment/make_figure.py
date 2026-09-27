"""
Reproduces the paper's headline figure (Figure 1) and Table 1 directly from
the bundled results/phase5_summary.json -- no raw document text needed.

Usage:
    python make_figure.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

CONDITIONS = ["unfiltered", "hl_filter", "hk_filter"]
GENS = list(range(7))
COLORS = {"unfiltered": "#E69F00", "hl_filter": "#56B4E9", "hk_filter": "#009E73"}
LABELS = {"unfiltered": "Unfiltered", "hl_filter": r"$H_L$ filter", "hk_filter": r"$\hat{H}_K$ filter"}
MARKERS = {"unfiltered": "s", "hl_filter": "D", "hk_filter": "o"}

with open(os.path.join(os.path.dirname(__file__), "results", "phase5_summary.json")) as f:
    records = json.load(f)


def by_gen(cond, key):
    return {g: np.array([r[key] for r in records
                          if r["condition"] == cond and r["generation"] == g and r.get(key) is not None])
            for g in GENS}


plt.rcParams.update({"font.family": "serif", "font.size": 9, "figure.dpi": 150,
                      "savefig.dpi": 200, "savefig.bbox": "tight"})

fig = plt.figure(figsize=(7.5, 2.8))
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.38)
ax_traj, ax_bar = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

# ── (a) Hk trajectory ─────────────────────────────────────────────────────────
for cond in CONDITIONS:
    hk_by_gen = by_gen(cond, "hk")
    means = [np.nanmean(hk_by_gen[g]) for g in GENS]
    sems = [np.nanstd(hk_by_gen[g]) / np.sqrt(max(len(hk_by_gen[g]), 1)) for g in GENS]
    ax_traj.errorbar(GENS, means, yerr=sems, color=COLORS[cond], marker=MARKERS[cond],
                      label=LABELS[cond], capsize=3, markeredgecolor="white",
                      markeredgewidth=0.5, zorder=3)
    ax_traj.fill_between(GENS, [m - s for m, s in zip(means, sems)],
                          [m + s for m, s in zip(means, sems)], color=COLORS[cond], alpha=0.12)
ax_traj.set_xlabel("Generation"); ax_traj.set_ylabel(r"$\hat{H}_K$ (bits/word)")
ax_traj.set_xticks(GENS); ax_traj.set_xlim(-0.3, 6.3)
ax_traj.legend(frameon=True, edgecolor="0.8", loc="lower left")
ax_traj.text(-0.18, 1.04, "(a)", transform=ax_traj.transAxes, fontsize=9, fontweight="bold", va="top")

# ── (b) text-diversity % change at gen 6 ─────────────────────────────────────
metric_keys, metric_names, invert = ["dist3", "vocab", "rep4"], ["Distinct-3", "Vocabulary", "Rep-4"], [False, False, True]
x = np.arange(len(metric_keys))
for cond, offset in zip(["hl_filter", "hk_filter"], [-0.15, 0.15]):
    pcts, ci_lo, ci_hi, pvals = [], [], [], []
    for ki, key in enumerate(metric_keys):
        u, f = by_gen("unfiltered", key)[6], by_gen(cond, key)[6]
        sign = -1 if invert[ki] else 1
        pct = sign * (np.mean(f) - np.mean(u)) / np.mean(u) * 100
        rng = np.random.default_rng(42)
        boot = [sign * (np.mean(rng.choice(f, len(f))) - np.mean(rng.choice(u, len(u)))) / np.mean(u) * 100
                for _ in range(2000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        _, p = stats.ttest_ind(f, u)
        pcts.append(pct); ci_lo.append(pct - lo); ci_hi.append(hi - pct); pvals.append(p)
    bars = ax_bar.bar(x + offset, pcts, 0.27, color=COLORS[cond], alpha=0.85,
                       yerr=[ci_lo, ci_hi], capsize=3, label=LABELS[cond], edgecolor="white", linewidth=0.5)
    for bar, p in zip(bars, pvals):
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3, star,
                    ha="center", va="bottom", fontsize=7, color="0.3")
ax_bar.set_xticks(x); ax_bar.set_xticklabels(metric_names)
ax_bar.set_ylabel("% change vs. unfiltered\n(positive = more diverse)")
ax_bar.axhline(0, color="0.5", linewidth=0.5)
ax_bar.legend(frameon=True, edgecolor="0.8", loc="upper right", fontsize=7)
ax_bar.text(-0.22, 1.04, "(b)", transform=ax_bar.transAxes, fontsize=9, fontweight="bold", va="top")

os.makedirs("../figures", exist_ok=True)
plt.savefig("../figures/fig1_hk_trajectory.pdf")
print("Saved ../figures/fig1_hk_trajectory.pdf")

# ── Table 1 ────────────────────────────────────────────────────────────────────
print("\nTable 1 -- generation-6 outcomes:")
print(f"{'Metric':<16}{'Unfilt.':>10}{'Hl-filt.':>10}{'Hk-filt.':>10}")
hk6 = {c: by_gen(c, 'hk')[6] for c in CONDITIONS}
print(f"{'Hk (bits/word)':<16}{np.mean(hk6['unfiltered']):>10.2f}{np.mean(hk6['hl_filter']):>10.2f}{np.mean(hk6['hk_filter']):>10.2f}")
for key, name, inv in zip(metric_keys, metric_names, invert):
    u = by_gen('unfiltered', key)[6]
    row = [name]
    for cond in ['hl_filter', 'hk_filter']:
        f = by_gen(cond, key)[6]
        sign = -1 if inv else 1
        pct = sign * (np.mean(f) - np.mean(u)) / np.mean(u) * 100
        row.append(f"{pct:+.0f}%")
    print(f"{row[0]:<16}{'--':>10}{row[1]:>10}{row[2]:>10}")
