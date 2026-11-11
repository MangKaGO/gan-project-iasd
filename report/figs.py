"""Figures of the A2 report (group CLM 34), in the style of the slides.

    python figs.py <repository root>

All metrics use the corrected pixel conversion (x+1)/2 (results/platform_rescale/).
Bars = mean over the seeds that train, dots = one seed each.
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results", "platform_rescale")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)
for f in glob.glob("/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyreheros-*.otf"):
    fm.fontManager.addfont(f)
INK, TEAL, MINT, GREY, LIGHT, PALE, BRICK, ROSE = ("#222222", "#1f6f6b", "#9cc5c1", "#8c8c8c",
                                                   "#cfcfcf", "#efefef", "#b5452f", "#e4b1a5")
plt.rcParams.update({
    "font.family": "TeX Gyre Heros", "font.size": 8.5, "axes.edgecolor": GREY, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": GREY, "xtick.labelsize": 7.4, "ytick.labelsize": 7.2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.linewidth": 0.6, "ytick.major.width": 0, "xtick.major.width": 0,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": PALE, "grid.linewidth": 0.8,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "mathtext.fontset": "cm"})
BATCHES = 938                                   # 60 000 / 64 batches per epoch
DEAD = {"baseline_seed1"}                       # starter run that collapses after 2 epochs

T = pd.read_csv(f"{RES}/training_metrics.csv")
S = pd.read_csv(f"{RES}/sampling_metrics.csv", keep_default_na=False)
logs = {os.path.basename(os.path.dirname(p)): pd.read_csv(p) for p in glob.glob(f"{ROOT}/runs/*/log.csv")}
METRICS = [("fid", "FID  (lower is better)", "{:.1f}"), ("precision", "precision", "{:.2f}"),
           ("recall", "recall", "{:.2f}")]
# the four cells of the ablation, same colour for both losses
CELL = {"": ("plain", GREY), "sn": ("SN", MINT), "ndis5": ("5 $D$", ROSE), "sn_ndis5": ("SN\n+5 $D$", TEAL)}
CFG = {"bce": {"": "baseline", "sn": "sn", "ndis5": "baseline_ndis5", "sn_ndis5": "sn_ndis5"},
       "hinge": {"": "baseline_paper", "sn": "sn_paper", "ndis5": "baseline_paper_ndis5",
                 "sn_ndis5": "sn_paper_ndis5"}}


def final(cfg):
    d = T[T.config == cfg]
    return d[d.epoch == d.epoch.max()]


def minutes(cfg):
    runs = [r for r in logs if r.startswith(cfg + "_seed")]
    return np.mean([logs[r].epoch_time_s.sum() / 60 for r in runs])


def bar(ax, x, v, color, width, fmt, pad, ymax, fs=6.6, bold=False, label=True):
    """Bar = mean, dots = seeds. A bar above the axis is cut, with its value written inside."""
    m = v.mean()
    ax.bar(x, min(m, ymax), width, color=color, zorder=2)
    jit = np.linspace(-width / 5, width / 5, len(v)) if len(v) > 1 else [0]
    if m > ymax:
        ax.text(x, ymax * 0.97, fmt.format(m) + r" $\rightarrow$", ha="center", va="top", fontsize=fs,
                color="white", rotation=90, fontweight="bold", zorder=4)
        return
    ax.scatter(x + np.asarray(jit), v, s=6, color=INK, alpha=0.55, lw=0, zorder=3)
    if not label:
        return
    if True:
        ax.text(x, max(min(v.max(), ymax), m) + pad, fmt.format(m), ha="center", fontsize=fs, color=INK,
                fontweight="bold" if bold else "normal")


# ------------------------------------------------------------------ Figure 1: the ablation
fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.25), gridspec_kw={"wspace": 0.2})
YMAX = {"fid": 80, "precision": 0.8, "recall": 0.75}
xs = {("bce", c): i for i, c in enumerate(CELL)} | {("hinge", c): 4.6 + i for i, c in enumerate(CELL)}
for ax, (m, title, fmt) in zip(axes, METRICS):
    for (loss, cell), x in xs.items():
        cfg = CFG[loss][cell]
        d = final(cfg)
        v = d[~d.run.isin(DEAD)][m].values
        bar(ax, x, v, CELL[cell][1], 0.72, fmt, YMAX[m] * 0.025, YMAX[m], fs=5.9,
            bold=cell == "sn_ndis5")
        n_single = int((d.class_entropy < 0.3).sum())
        if n_single == len(d) and m == "fid":   # every seed ends on a single digit
            ax.text(x, 0.03 * YMAX[m], "one digit", ha="center", va="bottom", fontsize=5.4, color=BRICK,
                    rotation=90, zorder=4, fontweight="bold")
        elif 0 < n_single < len(d):
            ax.text(x, max(v.max(), v.mean()) + 0.11 * YMAX[m], f"+{n_single} collapsed", ha="center",
                    fontsize=5.4, color=BRICK)
    ax.set_ylim(0, YMAX[m] * 1.08)
    ax.set_title(title, loc="left", fontsize=8.5, color=INK)
    ax.set_xticks(list(xs.values()))
    ax.set_xticklabels([CELL[c][0] for _, c in xs], fontsize=5.8)
    for x0, name in [(1.5, "BCE (starter loss)"), (6.1, "hinge loss")]:
        ax.text(x0, -0.27 * YMAX[m], name, ha="center", fontsize=6.8, color=INK, fontweight="bold")
    ax.axvline(3.8, color=LIGHT, lw=0.6, ymax=0.93)
fig.text(0.995, 1.03, "final checkpoint, 93.8k $G$ updates  ·  bar = mean of 3 seeds  ·  dot = one seed",
         ha="right", fontsize=7.2, color=TEAL, fontweight="bold")
fig.savefig(f"{OUT}/fig_ablation.pdf")
fig.savefig(f"{OUT}/fig_ablation.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ Figure 2: sample grids
GRIDS = [("baseline_paper_seed1", 100, "Hinge", GREY), ("sn_paper_seed1", 100, "Hinge + SN", GREY),
         ("baseline_paper_ndis5_seed0", 500, "Hinge + 5 $D$", BRICK),
         ("sn_paper_ndis5_seed1", 500, "Hinge + SN + 5 $D$", TEAL),
         ("baseline_paper_ep500_seed1", 500, "Hinge, 5× longer", BRICK),
         ("baseline_ndis5_seed0", 500, "BCE + 5 $D$", BRICK), ("sn_ndis5_seed0", 500, "BCE + SN + 5 $D$", TEAL)]
fig, axes = plt.subplots(1, len(GRIDS), figsize=(7.1, 1.55), gridspec_kw={"wspace": 0.05})
for ax, (run, ep, lab, col) in zip(axes, GRIDS):
    im = Image.open(f"{ROOT}/runs/{run}/grids/epoch_{ep:03d}.png").convert("L")
    im = im.crop((0, 0, im.width, 2 + 6 * 30))            # first 6 rows of the 8x8 grid (48 samples)
    ax.imshow(np.asarray(im), cmap="gray", vmin=0, vmax=255)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(lab, fontsize=6.8, pad=2.5, color=col, fontweight="bold")
fig.savefig(f"{OUT}/fig_grids.pdf", dpi=300)
fig.savefig(f"{OUT}/fig_grids.png", dpi=200)
plt.close(fig)


# ------------------------------------------------------------------ Figure 3: training dynamics (hinge)
def traj(cfg):
    d = T[(T.config == cfg) & ~T.run.isin(DEAD)].copy()
    d["k"] = d.epoch * BATCHES / 1e3                        # discriminator updates (thousands)
    return d.groupby("epoch").mean(numeric_only=True).reset_index()


LINES = [("baseline_paper", "Hinge", GREY, "-"), ("sn_paper", "Hinge + SN", MINT, "-"),
         ("baseline_paper_ndis5", "Hinge + 5 $D$", ROSE, "-"), ("sn_paper_ndis5", "Hinge + SN + 5 $D$", TEAL, "-"),
         ("baseline_paper_ep500", "Hinge, 5× longer", BRICK, (0, (3, 1.5)))]
fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.05), gridspec_kw={"wspace": 0.3})
ax = axes[0]
for cfg, lab, col, ls in LINES:
    tr = traj(cfg)
    ax.plot(tr.recall, tr.precision, color=col, ls=ls, lw=1.6, marker="o", ms=2.2, zorder=3)
    ax.plot(tr.recall.iloc[-1], tr.precision.iloc[-1], "o", ms=6, color=col, mec="white", mew=0.8, zorder=4)
ax.set_xlim(0, 0.68); ax.set_ylim(0, 0.82)
ax.set_xlabel("recall (diversity)", fontsize=7.2); ax.set_ylabel("precision (quality)", fontsize=7.2)
ax.grid(axis="both")
ax.set_title("precision vs. recall", loc="left", fontsize=8.5)
ax = axes[1]
for cfg, lab, col, ls in LINES:
    tr = traj(cfg)
    ax.plot(tr.k, tr.fid, color=col, ls=ls, lw=1.6, marker="o", ms=2.2)
ax.set_yscale("log"); ax.set_yticks([20, 40, 80, 160]); ax.set_yticklabels(["20", "40", "80", "160"])
ax.minorticks_off(); ax.set_xlim(0, 480)
ax.set_title("FID  (log scale)", loc="left", fontsize=8.5)
ax = axes[2]
for cfg, lab, col, ls in LINES:
    tr = traj(cfg)
    ax.plot(tr.k, tr.class_entropy, color=col, ls=ls, lw=1.6, marker="o", ms=2.2)
ax.set_ylim(-0.03, 1.05); ax.set_xlim(0, 480)
ax.set_title(r"digit balance $\bar H$", loc="left", fontsize=8.5)
for ax in axes[1:]:
    ax.axvline(93.8, color=GREY, lw=0.6, ls=":")
    ax.set_xlabel("$D$ updates (thousands)", fontsize=7.2)
axes[1].text(97, 165, "end of\n1-step runs", fontsize=5.8, color=GREY, va="top")
fig.legend(handles=[Line2D([], [], color=c, ls=ls, lw=1.8, label=l) for _, l, c, ls in LINES],
           loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=5, frameon=False, fontsize=7.4,
           handlelength=2.2, columnspacing=1.4)
fig.savefig(f"{OUT}/fig_dynamics.pdf")
fig.savefig(f"{OUT}/fig_dynamics.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ Figure 4: DRS
GROUPS = [("baseline_paper", "Hinge"), ("sn_paper_ndis5", "Hinge + SN + 5 $D$")]
VARS = [("no rejection", LIGHT, "none"), ("final $D$", TEAL, "final"), ("fine-tuned $D$", MINT, "calibrated"),
        ("$D$ from scratch", "#5a5a5a", "scratch")]


def pick(cfg, var, K):
    d = S[(S.config == cfg) & ~S.run.isin(DEAD)]
    if var == "none":
        return d[d.sampler == "vanilla"]
    return d[(d.sampler == "drs") & (d.D_variant == var) & (d.K.astype(float) == K)]


fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.0), gridspec_kw={"wspace": 0.3, "width_ratios": [1, 1, 1, 1.15]})
YM = {"fid": 52, "precision": 0.8, "recall": 0.75}
for ax, (m, title, fmt) in zip(axes[:3], METRICS):
    for i, (cfg, _) in enumerate(GROUPS):
        for j, (_, c, var) in enumerate(VARS):
            v = pick(cfg, var, 5.0)[m].astype(float).values
            bar(ax, i + (j - 1.5) * 0.21, v, c, 0.19, fmt, YM[m] * 0.025, YM[m], label=False)
    ax.set_ylim(0, YM[m] * 1.08)
    ax.set_title(title, loc="left", fontsize=8.5)
    ax.set_xticks(range(2)); ax.set_xticklabels([g[1] for g in GROUPS], fontsize=6.4)
ax = axes[3]
for cfg, ls, gname in [("baseline_paper", "-", "Hinge"), ("sn_paper_ndis5", (0, (3, 1.5)), "Hinge + SN + 5 D")]:
    base = pick(cfg, "none", 1).fid.mean()
    for _, c, var in VARS[1:]:
        vals = [0.0] + [pick(cfg, var, K).fid.mean() - base for K in (2.0, 5.0, 10.0)]
        ax.plot(range(4), vals, color=c, ls=ls, lw=1.5, marker="o", ms=2.6)
ax.axhline(0, color=GREY, lw=0.6)
ax.set_xticks(range(4)); ax.set_xticklabels(["1", "2", "5", "10"])
ax.set_xlabel("budget $K$", fontsize=7.2)
ax.set_title(r"$\Delta$FID vs. budget $K$", loc="left", fontsize=8.5)
ax.text(3.05, 2.9, "Hinge + SN + 5 $D$\n(dashed)", fontsize=5.6, color=INK, ha="right", va="bottom")
ax.text(0.1, -5.4, "Hinge (solid)", fontsize=5.6, color=INK, ha="left", va="center")
ax.set_ylim(-6.2, 4.6)
fig.legend(handles=[Patch(color=c, label=n) for n, c, _ in VARS], loc="upper center",
           bbox_to_anchor=(0.42, 1.12), ncol=4, frameon=False, fontsize=7.4)
fig.text(0.995, -0.1, "bars: $K = 5$  ·  mean of 3 seeds  ·  dot = one seed", ha="right", fontsize=7,
         color=TEAL, fontweight="bold")
fig.savefig(f"{OUT}/fig_drs.pdf")
fig.savefig(f"{OUT}/fig_drs.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ numbers used in the text
for cfg in ["baseline", "sn", "baseline_ndis5", "sn_ndis5", "baseline_paper", "sn_paper", "baseline_paper_ndis5",
            "sn_paper_ndis5", "baseline_paper_ep500"]:
    d = final(cfg); ok = d[~d.run.isin(DEAD)]
    print(f"{cfg:22s} FID {ok.fid.mean():6.1f}±{ok.fid.std():5.1f} P {ok.precision.mean():.2f} "
          f"R {ok.recall.mean():.2f} H {ok.class_entropy.mean():.2f} single {(d.class_entropy < 0.3).sum()}/3 "
          f"{minutes(cfg):.0f} min")
