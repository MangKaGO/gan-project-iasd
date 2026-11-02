"""Slide figures, drawn at their final size in the 16:9 deck (Latin Modern Sans, A1 palette).

    python figs.py      (from presentation/v2/; or set REPO_ROOT=<repository>)

Bars = mean over seeds, dots = one seed each. Collapsed runs are drawn apart.
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402

ROOT = os.environ.get("REPO_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for d in ["/usr/share/texmf/fonts/opentype/public/lm", "/usr/share/texlive/texmf-dist/fonts/opentype/public/lm"]:
    for f in ["lmsans10-regular.otf", "lmsans10-bold.otf"]:
        if os.path.exists(os.path.join(d, f)):
            fm.fontManager.addfont(os.path.join(d, f))
INK, NAVY, SKY, GREY, LIGHT, PALE, ORANGE = ("#1f2933", "#3d5a80", "#98b0cf", "#7b8794",
                                             "#c3ccd7", "#eef1f4", "#c0622f")
plt.rcParams.update({
    "font.family": "Latin Modern Sans", "font.size": 8.5, "axes.edgecolor": GREY, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": GREY, "xtick.labelsize": 7.8, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.linewidth": 0.6, "ytick.major.width": 0, "xtick.major.width": 0,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": PALE, "grid.linewidth": 0.7,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "mathtext.fontset": "cm"})

logs = {os.path.basename(os.path.dirname(p)): pd.read_csv(p) for p in glob.glob(f"{ROOT}/runs/*/log.csv")}
DEAD = {r for r, d in logs.items()
        if d.iloc[-1].D_fake < 0.01 or d.iloc[-1].D_fake > 0.99 or d.iloc[-1].D_real < 0.01}
T = pd.read_csv(f"{ROOT}/results/platform/training_metrics.csv")
S = pd.read_csv(f"{ROOT}/results/platform/sampling_metrics.csv", keep_default_na=False)


def minutes(cfg, epoch):
    runs = [r for r in T[T.config == cfg].run.unique() if r not in DEAD]
    return np.mean([logs[r][logs[r].epoch <= epoch].epoch_time_s.sum() / 60 for r in runs])


def bars(ax, groups, metric, ylim, fmt, top_pad):
    """groups: list of (x, colour, values of the live seeds, n collapsed)."""
    for x, c, v, n_dead in groups:
        ax.bar(x, v.mean(), 0.62, color=c, zorder=2)
        jitter = np.linspace(-0.12, 0.12, len(v)) if len(v) > 1 else [0]
        ax.scatter(x + np.asarray(jitter), v, s=9, color=INK, alpha=0.6, lw=0, zorder=3)
        ax.text(x, max(v.max(), v.mean()) + top_pad, fmt.format(v.mean()), ha="center",
                fontsize=8, color=INK, fontweight="bold" if c == NAVY else "normal")
        if n_dead:
            ax.text(x, max(v.max(), v.mean()) + top_pad * 4.2, f"+{n_dead} collapsed",
                    ha="center", fontsize=6.6, color=ORANGE)
    ax.set_ylim(*ylim)


# A. Training ablation: FID and recall per configuration --------------------------------
TRAIN = [("baseline", 100, LIGHT, "starter"),
         ("baseline_paper", 100, GREY, "paper\nsetting"),
         ("sn_paper", 100, SKY, "+ SN"),
         ("sn_paper_ndis5", 100, "#7f97b8", "+ SN\n×5 $D$"),
         ("sn_paper_ndis5", 500, NAVY, "+ SN\n×5 $D$")]
fig, axes = plt.subplots(1, 2, figsize=(5.9, 2.15), gridspec_kw={"wspace": 0.18})
for ax, metric, title, ylim, fmt, pad in [
        (axes[0], "fid", "FID  (lower is better)", (0, 80), "{:.1f}", 2.0),
        (axes[1], "recall", "recall  (higher is better)", (0, 0.42), "{:.2f}", 0.012)]:
    groups = []
    for i, (cfg, ep, c, _) in enumerate(TRAIN):
        d = T[(T.config == cfg) & (T.epoch == ep)]
        live = d[~d.run.isin(DEAD)][metric].astype(float).values
        groups.append((i, c, live, int(d.run.isin(DEAD).sum())))
    bars(ax, groups, metric, ylim, fmt, pad)
    ax.set_title(title, loc="left", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(TRAIN)))
    ax.set_xticklabels([t[3] for t in TRAIN], fontsize=7.2)
    for i, (cfg, ep, c, _) in enumerate(TRAIN):
        ax.text(i, -0.30 * ylim[1], f"{round(ep * 938 / (5 if 'ndis5' in cfg else 1) / 1000)}k",
                ha="center", fontsize=6.8, color=GREY)
        ax.text(i, -0.39 * ylim[1], f"{minutes(cfg, ep):.0f} min", ha="center", fontsize=6.8, color=GREY)
    if ax is axes[0]:
        ax.text(-0.55, -0.30 * ylim[1], "$G$ steps", ha="right", fontsize=6.8, color=GREY)
        ax.text(-0.55, -0.39 * ylim[1], "time", ha="right", fontsize=6.8, color=GREY)
fig.savefig("s_train.pdf")
plt.close(fig)

# B. Inference: same two generators, three ways of sampling -----------------------------
SAMP = [("vanilla", "none", 1.0, "", "vanilla\n1×"),
        ("drs", "calibrated", 5.0, "", "DRS\n5×"),
        ("truncation", "none", 1.0, "1.0", "truncation\n1×")]
MODELS = [("baseline_paper", GREY), ("sn_paper_ndis5", NAVY)]
fig, axes = plt.subplots(1, 3, figsize=(6.0, 2.0), gridspec_kw={"wspace": 0.2})
for ax, metric, title, ylim, fmt, pad in [
        (axes[0], "fid", "FID  (lower is better)", (0, 34), "{:.1f}", 0.8),
        (axes[1], "precision", "precision", (0, 0.8), "{:.2f}", 0.02),
        (axes[2], "recall", "recall", (0, 0.38), "{:.2f}", 0.01)]:
    groups = []
    for i, (smp, dv, K, psi, _) in enumerate(SAMP):
        for j, (cfg, c) in enumerate(MODELS):
            d = S[(S.config == cfg) & (S.sampler == smp) & (S.D_variant == dv)
                  & (S.K.astype(float) == K) & (S.psi.astype(str) == psi) & ~S.run.isin(DEAD)]
            groups.append((i + (j - 0.5) * 0.36, c, d[metric].astype(float).values, 0))
    for x, c, v, _ in groups:
        ax.bar(x, v.mean(), 0.33, color=c, zorder=2)
        ax.scatter([x] * len(v), v, s=7, color=INK, alpha=0.55, lw=0, zorder=3)
        ax.text(x, v.max() + pad, fmt.format(v.mean()), ha="center", fontsize=6.2, color=INK)
    ax.set_ylim(*ylim)
    ax.set_title(title, loc="left", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(SAMP)))
    ax.set_xticklabels([s[4] for s in SAMP], fontsize=7.2)
from matplotlib.patches import Patch  # noqa: E402
fig.legend(handles=[Patch(color=GREY, label="paper setting, no SN"),
                    Patch(color=NAVY, label="SN, 5 $D$ steps")],
           loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False, fontsize=8)
fig.savefig("s_inf.pdf")
plt.close(fig)
print("collapsed:", sorted(DEAD))
