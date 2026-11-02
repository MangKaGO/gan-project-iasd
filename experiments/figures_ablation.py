"""Clean figures and tables of the SN ablation and of the inference comparison.

    python experiments/figures_ablation.py          (from the repository root)

Inputs : runs/*/log.csv, results/platform/training_metrics.csv,
         results/platform/sampling_metrics.csv
Outputs: figures/ablation/*.pdf (report) and *.png (slides),
         results/platform/ablation_tables.md and ablation_tables.tex

Only the course metrics are shown: FID, precision, recall (and, for inference,
acceptance rate and time). Collapsed runs are excluded from means and counted apart.
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures", "ablation")
RES = os.path.join(ROOT, "results", "platform")
BATCHES_PER_EPOCH = 938                      # 60 000 MNIST images / batch size 64

# Latin Modern Sans (font of the slides) if installed, otherwise the default font.
for d in ["/usr/share/texmf/fonts/opentype/public/lm",
          "/usr/share/texlive/texmf-dist/fonts/opentype/public/lm"]:
    for f in ["lmsans10-regular.otf", "lmsans10-bold.otf"]:
        if os.path.exists(os.path.join(d, f)):
            fm.fontManager.addfont(os.path.join(d, f))
FONT = "Latin Modern Sans" if any("Latin Modern Sans" in f.name for f in fm.fontManager.ttflist) else "DejaVu Sans"

INK, NAVY, SKY, GREY, LIGHT, PALE, ORANGE = ("#1f2933", "#3d5a80", "#98b0cf", "#7b8794",
                                             "#b2becd", "#eef1f4", "#c0622f")
plt.rcParams.update({
    "font.family": FONT, "font.size": 9, "axes.edgecolor": GREY, "axes.labelcolor": INK,
    "xtick.color": GREY, "ytick.color": GREY, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": PALE, "grid.linewidth": 0.7,
    "lines.linewidth": 1.8, "legend.frameon": False, "legend.fontsize": 8,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "savefig.dpi": 300, "pdf.fonttype": 42,
    "mathtext.fontset": "cm",
})

# Training configurations of the ablation, in the order of the table.
TRAIN = [  # (config, name, colour, line style, D steps per G step)
    ("baseline", r"Starter baseline (BCE, $\beta_1$ 0.9)", LIGHT, (0, (3, 2)), 1),
    ("baseline_paper", r"Paper setting (hinge, $\beta_1$ 0.5)", GREY, "-", 1),
    ("sn_paper", "+ SN (1 D step per G step)", SKY, "-", 1),
    ("sn_paper_ndis5", "+ SN, 5 D steps per G step", NAVY, "-", 5),
]
STARTER = [("sn", "SN"), ("ttur", "TTUR"), ("sn_ttur", "SN + TTUR")]   # starter setting


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)


def load_logs():
    logs = {}
    for p in glob.glob(os.path.join(ROOT, "runs", "*", "log.csv")):
        logs[os.path.basename(os.path.dirname(p))] = pd.read_csv(p)
    return logs


def collapsed(logs):
    """D rejects (or accepts) everything at the end of training: G gets no gradient."""
    dead = set()
    for run, d in logs.items():
        last = d.iloc[-1]
        if last.D_fake < 0.01 or last.D_fake > 0.99 or last.D_real < 0.01:
            dead.add(run)
    return dead


def train_minutes(logs, runs, epoch):
    """Mean wall-clock training time (min) up to `epoch`, over the given runs."""
    t = [logs[r][logs[r].epoch <= epoch].epoch_time_s.sum() / 60 for r in runs if r in logs]
    return float(np.mean(t)) if t else np.nan


def pm(values, digits):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return "–"
    if len(v) == 1:
        return f"{v[0]:.{digits}f}"
    return f"{v.mean():.{digits}f} ± {v.std(ddof=1):.{digits}f}"


# ----------------------------------------------------------------------------
# Figure 1: FID at equal generator updates and at equal training time
# ----------------------------------------------------------------------------
def fig_budget(T, logs, dead):
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4), sharey=True, gridspec_kw={"wspace": 0.08})
    for cfg, name, color, ls, ndis in TRAIN:
        d = T[(T.config == cfg) & ~T.run.isin(dead)]
        if d.empty:
            continue
        g = d.groupby("epoch").fid
        epochs = np.array(sorted(d.epoch.unique()))
        mean = g.mean().loc[epochs].values
        updates = epochs * BATCHES_PER_EPOCH / ndis / 1000
        runs = d.run.unique()
        minutes = np.array([train_minutes(logs, runs, e) for e in epochs])
        for ax, x in zip(axes, (updates, minutes)):
            ax.plot(x, mean, color=color, ls=ls, marker="o", ms=3.5, zorder=3, label=name)
            # each seed, faint
            for e, xe in zip(epochs, x):
                ax.scatter([xe] * len(g.get_group(e)), g.get_group(e), s=9, color=color,
                           alpha=0.35, lw=0, zorder=2)
    axes[0].set_xlabel("Generator updates (thousands)")
    axes[1].set_xlabel("Training time (min, one GPU)")
    axes[0].set_ylabel("FID ↓")
    axes[0].set_title("Equal generator updates", loc="left", fontsize=9, color=INK)
    axes[1].set_title("Equal training time", loc="left", fontsize=9, color=INK)
    axes[0].set_ylim(15, 125)
    for ax in axes:
        ax.set_xlim(left=0)
    axes[0].axvline(BATCHES_PER_EPOCH * 100 / 1000, color=PALE, lw=1, zorder=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.13),
               handlelength=2.2, columnspacing=1.5)
    save(fig, "fig_sn_budget")


# ----------------------------------------------------------------------------
# Figure 2: inference methods on the two best training configurations
# ----------------------------------------------------------------------------
SAMPLERS = [  # (sampler, D variant, K, psi, label)
    ("vanilla", "none", 1.0, "", "Vanilla"),
    ("drs", "final", 5.0, "", "DRS\nD fin."),
    ("drs", "calibrated", 5.0, "", "DRS\nD rec."),
    ("truncation", "none", 1.0, "1.5", "Trunc.\n$\\psi$ 1.5"),
    ("truncation", "none", 1.0, "1.0", "Trunc.\n$\\psi$ 1.0"),
]
INF_CONFIGS = [("baseline_paper", "Paper setting, no SN", GREY),
               ("sn_paper_ndis5", "SN, 5 D steps", NAVY)]


def pick(S, cfg, sampler, dvar, K, psi):
    return S[(S.config == cfg) & (S.sampler == sampler) & (S.D_variant == dvar)
             & (S.K.astype(float) == K) & (S.psi.astype(str) == psi)]


def fig_inference(S, dead):
    S = S[~S.run.isin(dead)]
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.3), gridspec_kw={"wspace": 0.22})
    x = np.arange(len(SAMPLERS))
    w = 0.36
    for ax, (metric, label) in zip(axes, [("fid", "FID ↓"), ("precision", "Precision ↑"),
                                          ("recall", "Recall ↑")]):
        for j, (cfg, name, color) in enumerate(INF_CONFIGS):
            xs = x + (j - 0.5) * w
            for xi, (smp, dv, K, psi, _) in zip(xs, SAMPLERS):
                v = pick(S, cfg, smp, dv, K, psi)[metric].astype(float)
                if v.empty:
                    continue
                ax.bar(xi, v.mean(), w * 0.9, color=color, zorder=2,
                       label=name if xi == xs[0] else None)
                ax.scatter([xi] * len(v), v, s=7, color=INK, alpha=0.55, lw=0, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([s[-1] for s in SAMPLERS], fontsize=7)
        ax.set_title(label, loc="left", fontsize=9, color=INK)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylim(0, 32)
    axes[1].set_ylim(0, 0.75)
    axes[2].set_ylim(0, 0.36)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.1))
    save(fig, "fig_inference")


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------
def training_rows(T, logs, dead):
    rows = []

    def row(name, cfg, epoch, ndis):
        runs = sorted(T[T.config == cfg].run.unique())
        if not runs:
            return
        alive = [r for r in runs if r not in dead]
        d = T[(T.config == cfg) & (T.epoch == epoch) & T.run.isin(alive)]
        rows.append({
            "Training": name,
            "G updates": f"{round(epoch * BATCHES_PER_EPOCH / ndis / 1000)}k",
            "Time (min)": f"{train_minutes(logs, alive, epoch):.0f}",
            "Collapsed": f"{len(runs) - len(alive)}/{len(runs)}",
            "FID ↓": pm(d.fid, 1), "Precision ↑": pm(d.precision, 3), "Recall ↑": pm(d.recall, 3)})

    row("Starter baseline (BCE, β1 = 0.9)", "baseline", 100, 1)
    row("Paper setting (hinge, β1 = 0.5)", "baseline_paper", 100, 1)
    row("+ SN", "sn_paper", 100, 1)
    row("+ SN, 5 D steps (equal time)", "sn_paper_ndis5", 100, 5)
    row("+ SN, 5 D steps (equal G updates)", "sn_paper_ndis5", 500, 5)
    starter = []
    for cfg, name in STARTER:
        n = len(rows)
        row(f"Starter setting + {name}", cfg, 100, 1)
        if len(rows) > n:
            starter.append(rows.pop())
    return rows, starter


def inference_rows(S, dead):
    S = S[~S.run.isin(dead)]
    rows = []
    for cfg, name, _ in INF_CONFIGS:
        for smp, dv, K, psi, label in SAMPLERS[:3] + [("drs", "calibrated", 10.0, "", "DRS recal. D (K = 10)")] + SAMPLERS[3:]:
            d = pick(S, cfg, smp, dv, K, psi)
            if d.empty:
                continue
            label = (label.replace("\n", " ").replace("$\\psi$ ", "ψ = ").replace("D fin.", "final D")
                     .replace("D rec.", "recal. D"))
            if smp == "drs" and "K =" not in label:
                label += f" (K = {K:.0f})"
            rows.append({
                "Model": name, "Sampling": label,
                "Acceptance": f"{d.acceptance_rate.astype(float).mean():.2f}",
                "Time / 10k (s)": f"{d.time_s.astype(float).mean():.2f}",
                "FID ↓": pm(d.fid, 1), "Precision ↑": pm(d.precision, 3), "Recall ↑": pm(d.recall, 3)})
    return rows


def md_table(rows):
    if not rows:
        return ""
    cols = list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out) + "\n"


def tex_table(rows, caption):
    if not rows:
        return ""
    cols = list(rows[0])
    esc = lambda s: (str(s).replace("±", r"$\pm$").replace("↓", r"$\downarrow$")
                     .replace("↑", r"$\uparrow$").replace("β1", r"$\beta_1$").replace("ψ", r"$\psi$"))
    lines = [r"\begin{table}[t]", r"\centering\small", r"\begin{tabular}{l" + "r" * (len(cols) - 1) + "}",
             r"\toprule", " & ".join(esc(c) for c in cols) + r" \\", r"\midrule"]
    lines += [" & ".join(esc(r[c]) for c in cols) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}", r"\end{table}", ""]
    return "\n".join(lines)


def main():
    logs = load_logs()
    dead = collapsed(logs)
    T = pd.read_csv(os.path.join(RES, "training_metrics.csv"))
    S = pd.read_csv(os.path.join(RES, "sampling_metrics.csv"), keep_default_na=False)
    print("Collapsed runs:", ", ".join(sorted(dead)) or "none")

    fig_budget(T, logs, dead)
    fig_inference(S, dead)

    train, starter = training_rows(T, logs, dead)
    inf = inference_rows(S, dead)
    note = ("Mean ± std over the non-collapsed seeds (3 seeds per configuration). "
            "FID on 10 000 images (Inception), precision and recall on 9 984 images "
            "(VGG-16, k = 5), as on the course platform.")
    with open(os.path.join(RES, "ablation_tables.md"), "w", encoding="utf-8") as f:
        f.write("# Training ablation (vanilla sampling)\n\n" + note + "\n\n" + md_table(train))
        f.write("\nStarter setting (same budget as the starter baseline):\n\n" + md_table(starter))
        f.write("\n# Inference (same generator, 10 000 images)\n\n"
                "Acceptance = accepted / proposed samples (cost = 1 / acceptance generator passes). "
                "The recalibrated D is retrained for 3 epochs with G frozen.\n\n" + md_table(inf))
    with open(os.path.join(RES, "ablation_tables.tex"), "w", encoding="utf-8") as f:
        f.write(tex_table(train, "Training ablation, vanilla sampling. " + note))
        f.write(tex_table(inf, "Inference methods on the same generators."))
    print(f"Figures in {OUT}\nTables in {RES}/ablation_tables.md and .tex")


if __name__ == "__main__":
    main()
