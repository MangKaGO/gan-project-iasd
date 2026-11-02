# Experiments: spectral normalization and truncation on a fixed MLP GAN

Group 34: Nourcène Brahem, Lukas Tabouri, Manel Wafra.

This file explains what we ran, where everything is, and how to reproduce it.
The generator architecture is never changed; only the training of the
discriminator / the objective (training lever) and the sampling (inference lever) change.

## Question and hypotheses

- **Training lever: spectral normalization (SN)** of the discriminator
  (Miyato et al., ICLR 2018), used with the setting of the paper: hinge loss,
  Adam β1 = 0.5, 5 D steps per G step.
  *H1: SN makes training more stable and raises recall.*
- **Inference lever: latent truncation** (BigGAN, Brock et al., ICLR 2019), compared
  with **Discriminator Rejection Sampling** (DRS, Azadi et al., ICLR 2019).
  *H2: both raise precision; only truncation loses recall.*

## Branches and tags

| Where | What |
|---|---|
| `main` | Platform submission: generation code, `checkpoints/G.pth` (run `sn_paper_ndis5_seed2`) and `checkpoints/config.json`, which makes `generate.py` use truncation ψ = 1. |
| `experiments` | All code, result CSVs, training logs, sample grids, figures and slides. |
| tag `soumission-1-baseline` | First submission (starter baseline, seed 0), kept for reference. |

Checkpoints of all runs are only on the cluster (`runs/<run>/checkpoints/`), too large for git.

## Repository layout

| Path | Content |
|---|---|
| `model.py` | Generator (unchanged) and discriminator, optional SN (`spectral_norm=True`). |
| `utils.py` | Training steps: BCE (starter) and hinge; save / load helpers. |
| `train.py` | Training with options `--spectral_norm --loss {bce,hinge} --beta1 --lr_g --lr_d --n_dis --seed --run_dir`. Writes `config.json`, `log.csv`, checkpoints every `--save_every` epochs, sample grids. |
| `sampling.py` | Sampling methods: `vanilla`, `truncation` (ψ), `drs` and `obrs` (budget K). |
| `generate.py` | Platform contract, unchanged: `python generate.py --batch_size 256` writes 10 000 PNGs in `samples/`. The method is read from `checkpoints/config.json` (key `"sampling"`). |
| `evaluation/metrics.py` | FID (Inception, 10 000 images) and precision / recall (VGG-16, k = 5, 9 984 images), as on the course platform. |
| `experiments/` | Evaluation scripts (`eval_training.py`, `eval_sampling.py`, `calibrate_D.py`) and figure scripts (`figures_ablation.py`, `make_figures.py`). |
| `scripts/` | SLURM launchers for training and evaluation. |
| `runs/<config>_seed<k>/` | `config.json`, `log.csv`, `grids/` of each run. |
| `results/platform/` | `training_metrics.csv`, `sampling_metrics.csv`, `ablation_tables.md` / `.tex`. |
| `figures/ablation/` | Clean figures of the ablation and of the inference comparison. |
| `presentation/v2/` | Slides (`slides.pdf`, `slides.tex`), figures script, speaker notes. |

## Training configurations

All runs: batch 64, learning rate 2·10⁻⁴, β2 = 0.999, seeds 0, 1, 2.
One epoch = 938 batches. Runs with 100 epochs give G about 94k updates.

| Config | Options of `train.py` | Epochs | What it tests |
|---|---|---|---|
| `baseline` | none (starter code: BCE, β1 = 0.9) | 100 | Reference |
| `sn` | `--spectral_norm` | 100 | SN with the starter setting |
| `ttur` | `--lr_g 1e-4 --lr_d 4e-4` | 100 | TTUR alone (Heusel et al., 2017) |
| `sn_ttur` | `--spectral_norm --lr_g 1e-4 --lr_d 4e-4` | 100 | SN + TTUR, starter setting |
| `baseline_paper` | `--loss hinge --beta1 0.5` | 100 | Optimisation setting of the SN paper, without SN |
| `sn_paper` | `--spectral_norm --loss hinge --beta1 0.5` | 100 | + SN, 1 D step per G step |
| `sn_paper_ndis5` | `--spectral_norm --loss hinge --beta1 0.5 --n_dis 5` | 500 | + 5 D steps per G step. G is updated every 5 batches, so 500 epochs give the same 94k G updates; epoch 100 gives the same data and D updates as the others. |

## How to reproduce (on the cluster, from the repository root)

```bash
# 1. Training: one SLURM job per (config, seed); finished runs are skipped
CONFIGS="baseline sn" bash scripts/launch_training.sh
CONFIGS="baseline_paper sn_paper" bash scripts/launch_training.sh
CONFIGS="sn_paper_ndis5" TIME="03:00:00" bash scripts/launch_training.sh

# 2. Evaluation: checkpoints every 50 epochs, D recalibration, then sampling methods
#    (vanilla, truncation psi in {1.5, 1.0, 0.5}, DRS K in {2, 5, 10}).
#    Already evaluated lines are skipped, so it can be relaunched safely.
QUICK=1 bash scripts/launch_eval.sh all

# 3. Figures and tables (can run on a laptop: needs pandas and matplotlib)
python experiments/figures_ablation.py          # figures/ablation/, results/platform/ablation_tables.*
cd presentation/v2 && python figs.py            # figures of the slides

# 4. Generation as the platform does it
python generate.py --batch_size 256                                  # method from checkpoints/config.json
python generate.py --batch_size 256 --method truncation --psi 1.0    # override on the command line
```

## Evaluation protocol

- 3 seeds per configuration; tables give mean ± std over seeds.
- 10 000 generated images; FID on Inception features; precision and recall
  (Kynkäänniemi et al., 2019) on VGG-16 features with k = 5 and 9 984 images,
  as on the course platform. Check: same checkpoint, FID 31.4 with our code, 30.6 on the platform.
- A run is **collapsed** when, at the end of training, D outputs ~0 or ~1 on every fake
  (G gets no gradient). Collapsed runs are counted and excluded from the means.
- DRS: acceptance rate 1/K (K = 5: five candidates per kept image). The "recalibrated" D is
  trained 3 more epochs with G frozen (`experiments/calibrate_D.py`).

## Main results (mean over 3 seeds; full tables in `results/platform/ablation_tables.md`)

Training (vanilla sampling):

| Training | G updates | Time | Collapsed | FID ↓ | Precision ↑ | Recall ↑ |
|---|---|---|---|---|---|---|
| Starter baseline | 94k | 8 min | 1/3 | 29.5 ± 2.8 | 0.49 | 0.24 |
| + SN (starter setting) | 94k | 16 min | 0/3 | 177 ± 6 | 0.06 | 0.00 |
| Paper setting (hinge, β1 = 0.5) | 94k | 9 min | 0/3 | 28.1 ± 3.1 | 0.51 | 0.29 |
| + SN | 94k | 16 min | 0/3 | 52.7 ± 9.4 | 0.46 | 0.17 |
| + SN, 5 D steps, epoch 100 | 19k | 12 min | 0/3 | 60.7 ± 2.3 | 0.40 | 0.16 |
| + SN, 5 D steps, epoch 500 | 94k | 57 min | 0/3 | 27.6 ± 1.7 | 0.52 | 0.29 |

Inference (same generators):

| Sampling | Cost | Paper setting, no SN (FID / P / R) | SN, 5 D steps (FID / P / R) |
|---|---|---|---|
| Vanilla | 1× | 28.1 / 0.51 / 0.29 | 27.6 / 0.52 / 0.29 |
| DRS, recalibrated D, K = 5 | 5× + D recal. | 25.3 / 0.54 / 0.25 | 26.0 / 0.52 / 0.30 |
| Truncation ψ = 1 | 1× | 24.8 / 0.65 / 0.21 | 20.5 / 0.65 / 0.29 |

What we conclude:

1. The paper's optimisation setting (hinge, β1 = 0.5) removes the collapse and raises recall 0.24 → 0.29.
2. SN only matches it, and only with 5 D steps per G step (6× the training time). H1: stability yes, extra recall no.
3. Truncation ψ = 1 raises precision 0.52 → 0.65 for both models, but keeps recall only for the SN model (3/3 seeds).
4. DRS gives a small gain for 5× the samples. H2 is rejected here: free truncation beats rejection.

Submitted model: `sn_paper_ndis5_seed2` + truncation ψ = 1 (our measure: FID 17.2, precision 0.66, recall 0.27).
The number to report for the method is the 3-seed mean (FID 20.5), not this best seed.

## Limits and next steps

- 3 seeds only: FID gaps under ~3 points are within the seed spread.
- ψ was chosen on the reported metrics, without a held-out split.
- 5 D steps were never tested without SN (`baseline_paper` + `--n_dis 5`): the missing ablation.
- Next: SN + TTUR instead of 5 D steps; OBRS (Vérine et al., AISTATS 2024, already in `sampling.py`)
  against truncation at equal cost; measure how smooth G is in latent space with and without SN.
