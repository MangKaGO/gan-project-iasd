# DataLab A2: GAN on MNIST with a fixed generator

**Group CLM 34**: Nourcène Brahem, Lukas Tabouri, Manel Wafra

- `report.pdf`: the report (study, results, limits).
- `slides.pdf`: the presentation of 5 November.

We keep the provided MLP generator unchanged and study one lever, **how strong and how smooth the discriminator is**:

- **Training**: spectral normalization (SN) of the discriminator, crossed with the number of discriminator steps per generator step (1 or 5). We run it for the starter BCE loss and for the hinge loss, with 3 seeds and the same number of generator updates (93.8k) for every run.
- **Inference**: discriminator rejection sampling (DRS), with three estimators of the density ratio: the final D, the final D fine-tuned, and a new D trained from scratch.

## Main results

Our evaluator, final checkpoint, mean over 3 seeds (10,000 samples):

| Loss  | Discriminator   | FID ↓    | Precision ↑ | Recall ↑ |
|-------|-----------------|----------|-------------|----------|
| BCE   | starter         | 44.5 (1 seed of 3 collapses) | 0.35 | 0.44 |
| BCE   | SN + 5 D steps  | 37.4     | 0.55        | 0.46     |
| hinge | plain           | 42.0     | 0.44        | 0.46     |
| hinge | SN              | 43.1     | 0.49        | 0.33     |
| hinge | 5 D steps       | 136.5 (every seed: one digit) | 0.53 | 0.16 |
| hinge | **SN + 5 D steps** | **17.8** | **0.65** | **0.60** |

Platform scores of the submitted model (hinge + SN + 5 D steps, seed 1):

| Sampling                  | FID  | Precision | Recall |
|---------------------------|------|-----------|--------|
| latent truncation ψ = 1.5 | 14.4 | 0.82      | 0.61   |
| DRS, final D, K = 5       | 16.5 | 0.76      | 0.66   |

## The submission (`checkpoints/`)

- `checkpoints/G.pth`, `checkpoints/D.pth`: weights of `runs/sn_paper_ndis5_seed1`, which uses the hinge loss, SN, 5 D steps, Adam (2e-4, β = (0.5, 0.999)) and 500 epochs.
- `checkpoints/config.json`: its training configuration, plus the `"sampling"` entry that `generate.py` reads. The current setting is DRS with the final D, K = 5, and pixels saved as (x+1)/2.

```bash
python generate.py              # 10,000 PNG files in samples/, settings from checkpoints/config.json
python generate.py --method vanilla --rescale      # override the sampling from the command line
```

**Pixel conversion.** `train.py` normalises images to [-1, 1]. The starter `generate.py` saved the generator output clamped to [0, 1]. We save `(x+1)/2` instead (`"rescale": true`), which is the exact inverse of the training normalisation. The generator is unchanged.

## Repository layout

| Path | Content |
|------|---------|
| `train.py`, `model.py`, `utils.py` | training (options `--spectral_norm`, `--loss {bce,hinge}`, `--n_dis`, `--beta1`, `--beta2`, `--lr_g`, `--lr_d`); the generator architecture is the provided one |
| `sampling.py`, `generate.py` | vanilla sampling, latent truncation, DRS, OBRS |
| `evaluation/metrics.py` | FID (Inception), precision / recall (VGG-16 fc2, k = 5), digit-class entropy (diagnostic) |
| `experiments/eval_training.py` | metrics of every saved checkpoint → `results/<features>[_rescale]/training_metrics.csv` |
| `experiments/eval_sampling.py` | vanilla / truncation / DRS / OBRS on final checkpoints → `sampling_metrics.csv` |
| `experiments/calibrate_D.py` | D fine-tuned against the frozen G (`D_calibrated.pth`), or trained from scratch (`--from_scratch`, `D_scratch.pth`) |
| `experiments/eval_checkpoint.py` | metrics of a single checkpoint |
| `scripts/launch_training.sh`, `scripts/launch_eval.sh` | SLURM launchers (MesoNet / Juliet) |
| `results/platform_rescale/` | all metrics used in the report (pixels saved as (x+1)/2) |
| `runs/<config>_seed<k>/` | per run: `config.json`, `log.csv` (losses, D(x), D(G(z)), time per epoch), sample grids; the weights are not versioned |
| `report/` | LaTeX source of the report and `figs.py`, which rebuilds every figure from `results/` and `runs/` |

## Configurations

`scripts/launch_training.sh` has one name per configuration. Every configuration is run with seeds 0, 1 and 2.

| Name | Loss | SN | D steps | Epochs | In the report |
|------|------|----|---------|--------|---------------|
| `baseline` | BCE (starter) | – | 1 | 100 | starter |
| `sn` | BCE | ✓ | 1 | 100 | BCE + SN |
| `baseline_ndis5` | BCE | – | 5 | 500 | BCE + 5 D |
| `sn_ndis5` | BCE | ✓ | 5 | 500 | BCE + SN + 5 D |
| `baseline_paper` | hinge, β1 = 0.5 | – | 1 | 100 | Hinge |
| `sn_paper` | hinge, β1 = 0.5 | ✓ | 1 | 100 | Hinge + SN |
| `baseline_paper_ndis5` | hinge, β1 = 0.5 | – | 5 | 500 | Hinge + 5 D |
| `sn_paper_ndis5` | hinge, β1 = 0.5 | ✓ | 5 | 500 | Hinge + SN + 5 D |
| `baseline_paper` with `EPOCHS=500` | hinge, β1 = 0.5 | – | 1 | 500 | Hinge, 5× longer (equal D updates) |
| `sn_paperA_ndis5`, `baseline_paperA_ndis5` | hinge, lr 1e-4, β = (0.5, 0.9) | ✓ / – | 5 | 500 | setting A of Miyato et al.: failed, not analysed |

With 5 D steps, G is updated once every 5 batches. Five times more epochs therefore give every run the same number of generator updates.

## Reproducing

On Juliet (MesoNet), from the repository root, with the virtual environment set up (`python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`):

```bash
# training (one SLURM job per configuration and seed)
CONFIGS="baseline sn baseline_paper sn_paper" bash scripts/launch_training.sh
CONFIGS="baseline_ndis5 sn_ndis5 baseline_paper_ndis5 sn_paper_ndis5" TIME=03:00:00 bash scripts/launch_training.sh
CONFIGS="baseline_paper" EPOCHS=500 TIME=01:30:00 bash scripts/launch_training.sh

# evaluation (finished evaluations are skipped)
RESCALE=1 QUICK=1 bash scripts/launch_eval.sh all        # checkpoint metrics, truncation, DRS
RESCALE=1 bash scripts/launch_eval.sh scratch            # DRS with a discriminator trained from scratch

# figures of the report
python report/figs.py .
```

Outside Juliet, give `train.py` a `--data_path`. The code also runs on Apple Silicon (MPS) or CPU, but the Inception and VGG-16 metrics need a GPU.
