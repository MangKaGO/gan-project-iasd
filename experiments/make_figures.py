"""Build every figure and summary table of the report from the result CSVs.

    python experiments/make_figures.py

Inputs : runs/*/log.csv, runs/*/grids/, results/<features>/training_metrics*.csv,
         results/<features>/sampling_metrics*.csv (each part is skipped if missing; CSVs
         computed on different machines are merged)
Outputs: figures/<features>/*.pdf (report) and *.png (slides),
         results/<features>/summary.md (tables, mean ± std over seeds)

Collapsed runs (D rejects every fake at the end of training) are detected from
log.csv. They are EXCLUDED from averaged metrics (a dead run averaged with
working ones describes neither), SHOWN separately in the figures, and REPORTED
in the tables.
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import run_label  # noqa: E402

# ----------------------------------------------------------------------------
# Style: one fixed colour per entity (never per rank), colour-blind safe.
# ----------------------------------------------------------------------------
CONFIG_COLORS = {"baseline": "#2a78d6", "sn": "#eb6834", "ttur": "#1baf7a", "sn_ttur": "#c98500",
                 "sn_ndis5": "#6250d6", "baseline_paper": "#7b8794", "sn_paper": "#e34948",
                 "sn_paper_ndis5": "#008300"}
CONFIG_NAMES = {"baseline": "Baseline", "sn": "Spectral norm", "ttur": "TTUR", "sn_ttur": "SN + TTUR",
                "sn_ndis5": "SN, 5 D steps", "baseline_paper": "Baseline (paper setting)",
                "sn_paper": "SN (paper setting)", "sn_paper_ndis5": "SN (paper setting, 5 D steps)"}
SAMPLER_COLORS = {"drs": "#2a78d6", "obrs": "#eb6834", "truncation": "#1baf7a"}
SAMPLER_NAMES = {"drs": "DRS", "obrs": "OBRS", "truncation": "Truncation"}
INK, MUTED, SOFT, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e9e8e4"
HEAD = {"fid": "FID ↓", "precision": "Precision ↑", "recall": "Recall ↑", "density": "Density ↑",
        "coverage": "Coverage ↑", "class_entropy": "Class entropy ↑"}
METRICS = [("fid", "FID", "lower is better"), ("precision", "Precision", "quality"),
           ("recall", "Recall", "diversity")]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 10,
    "axes.labelsize": 11, "axes.labelcolor": MUTED, "axes.edgecolor": SOFT,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "lines.linewidth": 2.2, "lines.solid_capstyle": "round", "legend.frameon": False,
    "legend.fontsize": 10, "figure.facecolor": "white", "savefig.dpi": 220,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
})


def save(fig, name, out_dir):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"))
    plt.close(fig)
    print(f"saved {out_dir}/{name}.pdf/.png")


def configs_in(df):
    return [c for c in CONFIG_COLORS if c in set(df["config"])]


def mean_std(df, by, cols):
    g = df.groupby(by)[cols]
    return g.mean().add_suffix("_mean").join(g.std().fillna(0).add_suffix("_std")).reset_index()


def fmt(m, s, digits):
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def end_label(ax, x, y, text, color, dy=0):
    """Direct label at the end of a line (instead of a legend box)."""
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", va="center",
                fontsize=10, color=color, fontweight="bold", annotation_clip=False)


def suptitle(fig, title, subtitle=None):
    fig.text(0.01, 1.07 if subtitle else 1.02, title, fontsize=13, fontweight="bold",
             color=INK, ha="left", transform=fig.transFigure)
    if subtitle:
        fig.text(0.01, 1.01, subtitle, fontsize=10, color=MUTED, ha="left", transform=fig.transFigure)


# ----------------------------------------------------------------------------
# Runs and collapse detection
# ----------------------------------------------------------------------------

def load_logs(runs_dir):
    """All runs/*/log.csv in one table, with the run name and its configuration."""
    rows = []
    for path in glob.glob(os.path.join(runs_dir, "*", "log.csv")):
        log = pd.read_csv(path)
        with open(os.path.join(os.path.dirname(path), "config.json")) as f:
            log["config"] = run_label(json.load(f))
        log["run"] = os.path.basename(os.path.dirname(path))
        rows.append(log)
    return pd.concat(rows, ignore_index=True) if rows else None


def collapsed_runs(logs, threshold=0.01):
    """Dead runs: at the last epoch the sigmoid of D is saturated, i.e. D gives
    the same certain answer to every fake (mean D(fake) < threshold or
    > 1 - threshold) or rejects every real image (mean D(real) < threshold).
    The BCE loss is then clipped and G receives no useful gradient."""
    if logs is None:
        return []
    last = logs.sort_values("epoch").groupby("run").tail(1)
    dead = (last.D_fake < threshold) | (last.D_fake > 1 - threshold) | (last.D_real < threshold)
    return sorted(last[dead].run)


# ----------------------------------------------------------------------------
# Training lever
# ----------------------------------------------------------------------------

def fig_training_curves(df, collapsed, out):
    """FID / precision / recall along training: thin line per seed, bold mean."""
    ok = df[~df.run.isin(collapsed)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (m, name, hint) in zip(axes, METRICS):
        for cfg in configs_in(df):
            c = CONFIG_COLORS[cfg]
            for _, s in ok[ok.config == cfg].groupby("run"):
                s = s.sort_values("epoch")
                ax.plot(s.epoch, s[m], color=c, linewidth=1, alpha=0.35)
            mean = ok[ok.config == cfg].groupby("epoch")[m].mean()
            if mean.empty:
                continue
            ax.plot(mean.index, mean.values, color=c, label=CONFIG_NAMES[cfg])
            if m == "recall":
                end_label(ax, mean.index[-1], mean.values[-1], CONFIG_NAMES[cfg], c)
        ax.set_title(name)
        ax.text(1.0, 1.03, hint, transform=ax.transAxes, ha="right", fontsize=9, color=SOFT)
        ax.set_xlabel("Epoch")
        ax.set_xlim(left=0)
    axes[1].set_ylim(0, 1.02)
    axes[2].set_ylim(0, 1.02)
    excl = f"  ·  excluded (collapsed): {', '.join(collapsed)}" if collapsed else ""
    suptitle(fig, "Metrics during training",
             f"Bold: mean over converged seeds · thin: individual seeds{excl}")
    save(fig, "training_curves", out)


def fig_final_pr(df, collapsed, out):
    """Final precision vs recall of every training run: the main comparison of
    the training lever (quality and diversity at once)."""
    last = df[df.epoch == df.epoch.max()]
    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.grid(True, axis="both")
    for cfg in configs_in(last):
        c = CONFIG_COLORS[cfg]
        d = last[last.config == cfg]
        alive, dead = d[~d.run.isin(collapsed)], d[d.run.isin(collapsed)]
        ax.scatter(alive.recall, alive.precision, s=40, color=c, alpha=0.45, linewidths=0)
        if not dead.empty:
            ax.scatter(dead.recall, dead.precision, s=70, marker="x", color=c, linewidths=2)
            for _, r in dead.iterrows():
                ax.annotate(f"{r.run} (collapsed)", (r.recall, r.precision), xytext=(8, 0),
                            textcoords="offset points", fontsize=9, color=MUTED, va="center")
        if not alive.empty:
            mx, my = alive.recall.mean(), alive.precision.mean()
            ax.errorbar(mx, my, xerr=alive.recall.std() if len(alive) > 1 else 0,
                        yerr=alive.precision.std() if len(alive) > 1 else 0, fmt="o",
                        markersize=11, color=c, markeredgecolor="white", markeredgewidth=1.5,
                        elinewidth=1.5, capsize=0, zorder=5)
            ax.annotate(f"{CONFIG_NAMES[cfg]}\nFID {alive.fid.mean():.0f}", (mx, my),
                        xytext=(12, 10), textcoords="offset points", fontsize=10,
                        fontweight="bold", color=c)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Recall  →  more diverse")
    ax.set_ylabel("Precision  →  more realistic")
    ax.set_title("Final quality vs diversity of each training run")
    ax.text(0.0, -0.22, "Large dot: mean of converged seeds (± std) · small dots: seeds · ✕: collapsed run",
            transform=ax.transAxes, fontsize=9, color=MUTED)
    save(fig, "final_precision_recall", out)


def fig_training_dynamics(logs, collapsed, out):
    """Discriminator during training, ONE LINE PER RUN, so that a collapse stays
    visible instead of being averaged away."""
    df = logs.copy()
    df["gap"] = df["D_real"] - df["D_fake"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, (col, title) in zip(axes, [("loss_D", "Discriminator loss"),
                                       ("gap", "D(real) − D(fake): how well D separates")]):
        for cfg in configs_in(df):
            c = CONFIG_COLORS[cfg]
            for i, (run, s) in enumerate(df[df.config == cfg].groupby("run")):
                s = s.sort_values("epoch")
                dead = run in collapsed
                ax.plot(s.epoch, s[col], color=c, linewidth=1.8 if dead else 1.4,
                        linestyle=(0, (2, 2)) if dead else "-", alpha=0.9,
                        label=CONFIG_NAMES[cfg] if i == 0 else None)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
    chance = 2 * np.log(2)
    axes[0].axhline(chance, color=SOFT, linewidth=1, linestyle="--")
    axes[0].annotate("D at chance (2 ln 2)", (1.0, chance), xycoords=("axes fraction", "data"),
                     xytext=(-4, 6), textcoords="offset points", ha="right", fontsize=9, color=MUTED)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend(loc="upper right")
    sub = f"One line per run · dashed: collapsed run ({', '.join(collapsed)})" if collapsed \
        else "One line per run"
    suptitle(fig, "What the discriminator does during training", sub)
    save(fig, "training_dynamics", out)


def table_training(df, logs, collapsed, path):
    last = df[df.epoch == df.epoch.max()]
    cols = ["fid", "precision", "recall"]   # the metrics of the course platform
    out = []
    for title, d in (("converged runs only", last[~last.run.isin(collapsed)]), ("all runs", last)):
        if title == "all runs" and not collapsed:
            continue
        stats = mean_std(d, ["config"], cols)
        out += [f"## Training lever, {title} (epoch {df.epoch.max()}, vanilla sampling, mean ± std)", "",
                "| Config | Runs | Collapsed | " + " | ".join(HEAD[c] for c in cols) + " | Train time / run |",
                "|---|---|---|" + "---|" * len(cols) + "---|"]
        for cfg in configs_in(d):
            s = stats[stats.config == cfg].iloc[0]
            n = d[d.config == cfg].run.nunique()
            runs_cfg = set(last[last.config == cfg].run)
            n_dead = len([r for r in collapsed if r in runs_cfg])
            t = ""
            if logs is not None:
                per_run = logs[logs.config == cfg].groupby("run").epoch_time_s.sum().mean()
                t = f"{per_run / 60:.1f} min"
            vals = " | ".join(fmt(s[f"{c}_mean"], s[f"{c}_std"], 1 if c == "fid" else 3) for c in cols)
            out.append(f"| {CONFIG_NAMES[cfg]} | {n} | {n_dead}/{len(runs_cfg)} | {vals} | {t} |")
        out.append("")
    with open(path, "a") as f:
        f.write("\n".join(out) + "\n")


# ----------------------------------------------------------------------------
# Inference lever
# ----------------------------------------------------------------------------

def fig_sampling_vs_K(df, cfg, out):
    """Precision, recall and FID of DRS / OBRS as the budget K grows."""
    d = df[(df.config == cfg) & df.sampler.isin(["vanilla", "drs", "obrs"])]
    van = d[d.sampler == "vanilla"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (m, name, hint) in zip(axes, METRICS):
        for sampler in ("drs", "obrs"):
            for variant, style in (("final", "-"), ("calibrated", (0, (3, 2)))):
                part = d[(d.sampler == sampler) & (d.D_variant == variant)]
                if part.empty:
                    continue
                part = pd.concat([van, part])
                s = part.groupby("K")[m].agg(["mean", "std"]).fillna(0)
                c = SAMPLER_COLORS[sampler]
                ax.plot(s.index, s["mean"], color=c, linestyle=style, marker="o", markersize=5,
                        label=SAMPLER_NAMES[sampler] + (" · recalibrated D" if variant == "calibrated" else ""))
                ax.fill_between(s.index, s["mean"] - s["std"], s["mean"] + s["std"],
                                color=c, alpha=0.10, linewidth=0)
        if not van.empty:
            ax.axhline(van[m].mean(), color=SOFT, linewidth=1, linestyle=":")
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 20])
        ax.set_xticklabels(["1", "2", "5", "10", "20"])
        ax.minorticks_off()
        ax.set_title(name)
        ax.text(1.0, 1.03, hint, transform=ax.transAxes, ha="right", fontsize=9, color=SOFT)
        ax.set_xlabel("Budget K  (candidates per kept image)")
    axes[0].legend(loc="best", fontsize=9)
    n = d.run.nunique()
    suptitle(fig, f"Rejection sampling on the {CONFIG_NAMES[cfg]} generator",
             f"Mean ± std over {n} converged run(s) · dotted line: vanilla sampling (K = 1)")
    save(fig, f"sampling_vs_K_{cfg}", out)


def fig_pr_plane(df, cfg, out):
    """Truncation vs rejection in the precision-recall plane. Left: full view.
    Right: zoom around vanilla, where the rejection samplers move (their moves
    are small compared with truncation's)."""
    d = df[df.config == cfg]
    stats = mean_std(d, ["sampler", "D_variant", "K", "psi"], ["precision", "recall"])
    van = stats[stats.sampler == "vanilla"]
    if van.empty:
        return
    v = van.iloc[0]
    curves = {}
    for sampler, order in (("truncation", "psi"), ("drs", "K"), ("obrs", "K")):
        s = stats[(stats.sampler == sampler) & (stats.D_variant.isin(["none", "final"]))]
        if not s.empty:
            s = s.assign(_o=pd.to_numeric(s[order])).sort_values("_o", ascending=(order == "K"))
            curves[sampler] = pd.concat([van, s])

    def draw(a, label_points, ends_only=False):
        for sampler, s in curves.items():
            c = SAMPLER_COLORS[sampler]
            a.plot(s.recall_mean, s.precision_mean, color=c, marker="o", markersize=5,
                   label=SAMPLER_NAMES[sampler])
            if sampler in label_points:
                pts = s.iloc[1:]
                if ends_only:
                    pts = pts.iloc[[0, -1]] if len(pts) > 1 else pts
                for _, r in pts.iterrows():
                    tag = f"ψ={float(r.psi):g}" if sampler == "truncation" else f"K={r.K:g}"
                    a.annotate(tag, (r.recall_mean, r.precision_mean),
                               xytext=(6, -12 if sampler == "drs" else 6),
                               textcoords="offset points", fontsize=9, color=c)
        a.plot(v.recall_mean, v.precision_mean, marker="s", markersize=9, color=INK,
               linestyle="none", label="Vanilla", zorder=6)
        a.grid(True, axis="both")
        a.set_xlabel("Recall  →  more diverse")

    rej = [curves[k] for k in ("drs", "obrs") if k in curves]
    fig, axes = plt.subplots(1, 2 if rej else 1, figsize=(12.5 if rej else 6.5, 4.8), squeeze=False)
    ax = axes[0][0]
    draw(ax, label_points={"truncation"})
    ax.set_ylabel("Precision  →  more realistic")
    ax.set_title("Full view")
    ax.legend(loc="lower left")
    if rej:
        az = axes[0][1]
        draw(az, label_points={"drs", "obrs"}, ends_only=True)
        pts = pd.concat(rej)
        x0, x1 = pts.recall_mean.min(), pts.recall_mean.max()
        y0, y1 = pts.precision_mean.min(), pts.precision_mean.max()
        px, py = max((x1 - x0) * 0.3, 0.004), max((y1 - y0) * 0.3, 0.004)
        az.set_xlim(x0 - px, x1 + px)
        az.set_ylim(y0 - py, y1 + py)
        az.set_title("Zoom around vanilla: rejection sampling")
        # Same zoom box drawn on the full view.
        ax.add_patch(plt.Rectangle((x0 - px, y0 - py), x1 - x0 + 2 * px, y1 - y0 + 2 * py,
                                   fill=False, edgecolor=SOFT, linewidth=1, linestyle="--", zorder=1))
    n = d.run.nunique()
    suptitle(fig, f"Truncation vs rejection · {CONFIG_NAMES[cfg]} generator",
             f"Mean over {n} converged run(s) · every curve starts at vanilla sampling (■) · "
             "up-right is better")
    save(fig, f"pr_plane_{cfg}", out)


def table_sampling(df, path):
    cols = ["fid", "precision", "recall"]   # the metrics of the course platform
    stats = mean_std(df, ["config", "sampler", "D_variant", "K", "psi"], cols + ["acceptance_rate", "time_s"])
    lines = ["## Inference lever, converged runs only (final checkpoints, mean ± std over seeds)", "",
             "| Generator | Sampler | D | K | ψ | " + " | ".join(HEAD[c] for c in cols) + " | Acceptance | Time (s) |",
             "|---|---|---|---|---|" + "---|" * len(cols) + "---|---|"]
    for _, s in stats.iterrows():
        vals = " | ".join(fmt(s[f"{c}_mean"], s[f"{c}_std"], 1 if c == "fid" else 3) for c in cols)
        lines.append(f"| {CONFIG_NAMES.get(s.config, s.config)} | {s.sampler} | {s.D_variant} | {s.K:g} | "
                     f"{s.psi} | {vals} | {s.acceptance_rate_mean:.3f} | {s.time_s_mean:.1f} |")
    with open(path, "a") as f:
        f.write("\n".join(lines) + "\n\n")


# ----------------------------------------------------------------------------
# Samples
# ----------------------------------------------------------------------------

def fig_grids(runs_dir, train_df, collapsed, out, seed=0):
    """Final 64-sample grids, same latent batch for every configuration, with
    the final metrics of that run underneath."""
    paths = {}
    for cfg in CONFIG_COLORS:
        grids = sorted(glob.glob(os.path.join(runs_dir, f"{cfg}_seed{seed}", "grids", "epoch_*.png")))
        if grids:
            paths[cfg] = grids[-1]
    if not paths:
        return
    fig, axes = plt.subplots(1, len(paths), figsize=(3.4 * len(paths), 4.0), squeeze=False)
    for ax, (cfg, p) in zip(axes[0], paths.items()):
        ax.imshow(plt.imread(p), cmap="gray", interpolation="nearest")
        ax.set_title(CONFIG_NAMES[cfg], color=CONFIG_COLORS[cfg])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        run = f"{cfg}_seed{seed}"
        if train_df is not None:
            r = train_df[(train_df.run == run) & (train_df.epoch == train_df.epoch.max())]
            if not r.empty:
                r = r.iloc[0]
                txt = f"FID {r.fid:.0f} · P {r.precision:.2f} · R {r.recall:.2f}"
                if run in collapsed:
                    txt += "\n(collapsed run)"
                ax.set_xlabel(txt, fontsize=9, color=MUTED)
    suptitle(fig, "Generated digits", f"Same latent batch for every configuration (seed {seed}, last epoch)")
    save(fig, "grids", out)


def read_all(pattern, **kwargs):
    """Concatenate every CSV matching pattern (None if there is none)."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        return None
    df = pd.concat([pd.read_csv(p, **kwargs) for p in paths], ignore_index=True)
    return df.drop_duplicates(subset=[c for c in df.columns if c not in ("time_s",)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default="runs")
    parser.add_argument("--features", default="platform", choices=["platform", "mnist"],
                        help="Which evaluation to plot (never mix the two feature spaces).")
    parser.add_argument("--results_dir", default=None, help="Default: results/<features>")
    parser.add_argument("--out", default=None, help="Default: figures/<features>")
    args = parser.parse_args()
    args.results_dir = args.results_dir or os.path.join("results", args.features)
    args.out = args.out or os.path.join("figures", args.features)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    summary = os.path.join(args.results_dir, "summary.md")

    logs = load_logs(args.runs_dir)
    collapsed = collapsed_runs(logs)
    print(f"Collapsed runs: {collapsed or 'none'}")
    with open(summary, "w") as f:
        f.write(f"Collapsed runs (saturated D at the last epoch): {', '.join(collapsed) or 'none'}\n\n")
    if logs is not None:
        fig_training_dynamics(logs, collapsed, args.out)

    train = read_all(os.path.join(args.results_dir, "training_metrics*.csv"))
    if train is not None:
        fig_training_curves(train, collapsed, args.out)
        fig_final_pr(train, collapsed, args.out)
        table_training(train, logs, collapsed, summary)
    fig_grids(args.runs_dir, train, collapsed, args.out)

    samp = read_all(os.path.join(args.results_dir, "sampling_metrics*.csv"), keep_default_na=False)
    if samp is not None:
        samp["K"] = samp["K"].astype(float)
        ok = samp[~samp.run.isin(collapsed)]
        for cfg in configs_in(ok):
            fig_sampling_vs_K(ok, cfg, args.out)
            fig_pr_plane(ok, cfg, args.out)
        table_sampling(ok, summary)
    print(f"Tables written to {summary}")


if __name__ == "__main__":
    main()
