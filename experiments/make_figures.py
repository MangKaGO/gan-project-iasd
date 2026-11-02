"""Build every figure and summary table of the report from the result CSVs.

    python experiments/make_figures.py

Inputs : runs/*/log.csv, runs/*/grids/, results/training_metrics.csv,
         results/sampling_metrics.csv (each part is skipped if missing)
Outputs: figures/*.pdf (for the report) and *.png (for the slides),
         results/summary.md (tables, mean ± std over seeds)
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from common import run_label  # noqa: E402

# Fixed colour per entity (never per rank), colour-blind safe palette.
CONFIG_COLORS = {"baseline": "#2a78d6", "sn": "#eb6834", "ttur": "#1baf7a", "sn_ttur": "#eda100"}
CONFIG_NAMES = {"baseline": "Baseline", "sn": "Spectral norm", "ttur": "TTUR", "sn_ttur": "SN + TTUR"}
SAMPLER_COLORS = {"drs": "#2a78d6", "obrs": "#eb6834", "truncation": "#1baf7a"}
SAMPLER_NAMES = {"drs": "DRS", "obrs": "OBRS", "truncation": "Truncation"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
METRICS = [("fid", "FID ↓"), ("precision", "Precision ↑"), ("recall", "Recall ↑")]

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 11,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "legend.frameon": False, "savefig.dpi": 200,
    "savefig.bbox": "tight",
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


# ----------------------------------------------------------------------------
# Training lever
# ----------------------------------------------------------------------------

def fig_training_curves(df, out):
    """FID / precision / recall along training, mean ± std over seeds."""
    stats = mean_std(df, ["config", "epoch"], [m for m, _ in METRICS])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, (m, label) in zip(axes, METRICS):
        for cfg in configs_in(df):
            s = stats[stats.config == cfg].sort_values("epoch")
            c = CONFIG_COLORS[cfg]
            ax.plot(s.epoch, s[f"{m}_mean"], color=c, label=CONFIG_NAMES[cfg])
            ax.fill_between(s.epoch, s[f"{m}_mean"] - s[f"{m}_std"],
                            s[f"{m}_mean"] + s[f"{m}_std"], color=c, alpha=0.18, linewidth=0)
        ax.set_title(label)
        ax.set_xlabel("Epoch")
    axes[0].legend()
    fig.suptitle("Metrics during training (mean ± std over seeds)", x=0.01, ha="left", y=1.04)
    save(fig, "training_curves", out)


def fig_training_dynamics(runs_dir, out):
    """Discriminator scores during training: how confident does D become?"""
    rows = []
    for path in glob.glob(os.path.join(runs_dir, "*", "log.csv")):
        log = pd.read_csv(path)
        with open(os.path.join(os.path.dirname(path), "config.json")) as f:
            log["config"] = run_label(json.load(f))
        rows.append(log)
    if not rows:
        return None
    df = pd.concat(rows)
    df["gap"] = df["D_real"] - df["D_fake"]
    stats = mean_std(df, ["config", "epoch"], ["loss_D", "gap"])
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, (col, title) in zip(axes, [("loss_D", "Discriminator loss"),
                                       ("gap", "D(real) − D(fake)")]):
        for cfg in configs_in(df):
            s = stats[stats.config == cfg].sort_values("epoch")
            c = CONFIG_COLORS[cfg]
            ax.plot(s.epoch, s[f"{col}_mean"], color=c, label=CONFIG_NAMES[cfg])
            ax.fill_between(s.epoch, s[f"{col}_mean"] - s[f"{col}_std"],
                            s[f"{col}_mean"] + s[f"{col}_std"], color=c, alpha=0.18, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
    axes[0].legend()
    save(fig, "training_dynamics", out)
    return df


def table_training(df, logs, path):
    last = df[df.epoch == df.epoch.max()]
    cols = ["fid", "precision", "recall", "class_entropy"]
    stats = mean_std(last, ["config"], cols)
    lines = [f"## Training lever (epoch {df.epoch.max()}, vanilla sampling, mean ± std over seeds)", "",
             "| Config | Seeds | FID ↓ | Precision ↑ | Recall ↑ | Class entropy ↑ | Train time / run |",
             "|---|---|---|---|---|---|---|"]
    for cfg in configs_in(last):
        s = stats[stats.config == cfg].iloc[0]
        n = last[last.config == cfg].seed.nunique()
        t = ""
        if logs is not None:
            per_run = logs[logs.config == cfg].groupby(["config", "epoch"]).epoch_time_s.mean().sum()
            t = f"{per_run / 60:.1f} min"
        lines.append(f"| {CONFIG_NAMES[cfg]} | {n} | {fmt(s.fid_mean, s.fid_std, 2)} | "
                     f"{fmt(s.precision_mean, s.precision_std, 3)} | {fmt(s.recall_mean, s.recall_std, 3)} | "
                     f"{fmt(s.class_entropy_mean, s.class_entropy_std, 3)} | {t} |")
    with open(path, "a") as f:
        f.write("\n".join(lines) + "\n\n")


# ----------------------------------------------------------------------------
# Inference lever
# ----------------------------------------------------------------------------

def fig_sampling_vs_K(df, cfg, out):
    """Precision, recall and FID of DRS / OBRS as the budget K grows."""
    d = df[(df.config == cfg) & df.sampler.isin(["vanilla", "drs", "obrs"])].copy()
    d.loc[d.sampler == "vanilla", "D_variant"] = "final"
    rows = []
    for sampler in ("drs", "obrs"):
        part = d[(d.sampler == sampler) | (d.sampler == "vanilla")].copy()
        part["curve"] = sampler
        rows.append(part)
    d = pd.concat(rows)
    stats = mean_std(d, ["curve", "D_variant", "K"], [m for m, _ in METRICS])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, (m, label) in zip(axes, METRICS):
        for sampler in ("drs", "obrs"):
            for variant, style in (("final", "-"), ("calibrated", "--")):
                s = stats[(stats.curve == sampler) & (stats.D_variant == variant)].sort_values("K")
                if variant == "calibrated":     # vanilla point (K = 1) is shared
                    s = pd.concat([stats[(stats.curve == sampler) & (stats.K == 1)], s])
                if len(s) < 2:
                    continue
                c = SAMPLER_COLORS[sampler]
                lbl = SAMPLER_NAMES[sampler] + (" (calibrated D)" if variant == "calibrated" else "")
                ax.errorbar(s.K, s[f"{m}_mean"], yerr=s[f"{m}_std"], color=c, linestyle=style,
                            marker="o", markersize=5, capsize=2, label=lbl)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 20])
        ax.set_xticklabels(["1", "2", "5", "10", "20"])
        ax.set_title(label)
        ax.set_xlabel("Budget K (candidates per kept image)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Rejection sampling on the {CONFIG_NAMES[cfg]} generator", x=0.01, ha="left", y=1.04)
    save(fig, f"sampling_vs_K_{cfg}", out)


def fig_pr_plane(df, cfg, out):
    """Precision-recall plane: truncation vs rejection, all starting from vanilla."""
    d = df[df.config == cfg]
    stats = mean_std(d, ["sampler", "D_variant", "K", "psi"], ["precision", "recall"])
    van = stats[stats.sampler == "vanilla"]
    fig, ax = plt.subplots(figsize=(5, 4.2))
    for sampler, order in (("truncation", "psi"), ("drs", "K"), ("obrs", "K")):
        s = stats[(stats.sampler == sampler) & (stats.D_variant.isin(["none", "final"]))]
        if s.empty:
            continue
        s = s.sort_values(order, ascending=(order == "K"))
        s = pd.concat([van, s])
        ax.plot(s.recall_mean, s.precision_mean, marker="o", markersize=5,
                color=SAMPLER_COLORS[sampler], label=SAMPLER_NAMES[sampler])
        end = s.iloc[-1]
        tag = f"ψ={end.psi}" if sampler == "truncation" else f"K={end.K:g}"
        ax.annotate(tag, (end.recall_mean, end.precision_mean), textcoords="offset points",
                    xytext=(6, 4), fontsize=8, color=MUTED)
    if not van.empty:
        ax.plot(van.recall_mean, van.precision_mean, marker="s", markersize=9,
                color=INK, linestyle="none", label="Vanilla")
    ax.set_xlabel("Recall (diversity) ↑")
    ax.set_ylabel("Precision (quality) ↑")
    ax.set_title(f"Precision–recall trade-off ({CONFIG_NAMES[cfg]} generator)")
    ax.legend()
    save(fig, f"pr_plane_{cfg}", out)


def table_sampling(df, path):
    cols = ["fid", "precision", "recall", "class_entropy", "acceptance_rate", "time_s"]
    stats = mean_std(df, ["config", "sampler", "D_variant", "K", "psi"], cols)
    lines = ["## Inference lever (final checkpoints, mean ± std over seeds)", "",
             "| Generator | Sampler | D | K | ψ | FID ↓ | Precision ↑ | Recall ↑ | Acceptance | Time (s) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, s in stats.iterrows():
        lines.append(f"| {CONFIG_NAMES.get(s.config, s.config)} | {s.sampler} | {s.D_variant} | {s.K:g} | "
                     f"{s.psi} | {fmt(s.fid_mean, s.fid_std, 2)} | "
                     f"{fmt(s.precision_mean, s.precision_std, 3)} | {fmt(s.recall_mean, s.recall_std, 3)} | "
                     f"{s.acceptance_rate_mean:.3f} | {s.time_s_mean:.1f} |")
    with open(path, "a") as f:
        f.write("\n".join(lines) + "\n\n")


# ----------------------------------------------------------------------------
# Samples
# ----------------------------------------------------------------------------

def fig_grids(runs_dir, out, seed=0):
    """Final 64-sample grids, same latent batch for every configuration."""
    paths = {}
    for cfg in CONFIG_COLORS:
        grids = sorted(glob.glob(os.path.join(runs_dir, f"{cfg}_seed{seed}", "grids", "epoch_*.png")))
        if grids:
            paths[cfg] = grids[-1]
    if not paths:
        return
    fig, axes = plt.subplots(1, len(paths), figsize=(3.2 * len(paths), 3.5), squeeze=False)
    for ax, (cfg, p) in zip(axes[0], paths.items()):
        ax.imshow(plt.imread(p), cmap="gray")
        ax.set_title(CONFIG_NAMES[cfg])
        ax.axis("off")
    fig.suptitle(f"Samples from the same latent batch (seed {seed}, last epoch)", x=0.01, ha="left", y=1.04)
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
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    summary = os.path.join(args.results_dir, "summary.md")
    os.makedirs(args.results_dir, exist_ok=True)
    open(summary, "w").close()

    logs = fig_training_dynamics(args.runs_dir, args.out)
    fig_grids(args.runs_dir, args.out)

    # Every results/training_metrics*.csv is read, so that results computed on
    # different machines (e.g. training_metrics_<name>.csv) are merged.
    df = read_all(os.path.join(args.results_dir, "training_metrics*.csv"))
    if df is not None:
        fig_training_curves(df, args.out)
        table_training(df, logs, summary)

    df = read_all(os.path.join(args.results_dir, "sampling_metrics*.csv"), keep_default_na=False)
    if df is not None:
        df["K"] = df["K"].astype(float)
        for cfg in configs_in(df):
            fig_sampling_vs_K(df, cfg, args.out)
            fig_pr_plane(df, cfg, args.out)
        table_sampling(df, summary)
    print(f"Tables written to {summary}")


if __name__ == "__main__":
    main()
