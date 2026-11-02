# DataLabAssignement2

## Using GPUs with MesoNet

### 1. Access MesoNet
To use the provided GPUs, you must first:
- Create a MesoNet account and request access to the project.
- Once accepted, add your SSH key to MesoNet for secure access. Follow the instructions [here](https://www.mesonet.fr/documentation/user-documentation/code_form/juliet/connexion).

### 2. Set Up Your Environment
On Juliet (MesoNet's cluster), you need to:

1. Create a virtual environment for Python:
   ```bash
   python -m venv venv
   ```

2. Activate the environment:
   ```bash
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Run Training or Generation
- Use the provided shell scripts (`scripts/train.sh` or `scripts/generate.sh`) to launch jobs.
- These scripts will automatically activate the virtual environment (`venv`).
- If you don't use Juliet cluster, please provide a data_path argument for the training scripts. Otherwise, everything should be automated.

### 4. Hardware Compatibility
The code is compatible with:
- **NVIDIA GPUs (CUDA)**
- **Apple Silicon (MPS, for M1/M2/M3/M4 chips)**
- **CPU-only mode** (not recommended for performance)

We recommend using CUDA (for NVIDIA GPUs) or MPS (for Apple Silicon).

### 5. Slurm Resources
- If you need help with Slurm (MesoNet's job scheduler), there will be a Slurm lecture on 29/10/2025 or refer to the official documentation: [Slurm Documentation](https://slurm.schedmd.com/documentation.html).

### 6. Example Commands
To submit a job to MesoNet:
```bash
# Make the script executable
chmod +x scripts/train.sh

# Submit the job to Slurm
scripts/train.sh
```
## train.py
This script performs adversarial training for a GAN. Before running it, ensure the `DATA` variable in `scripts/train.sh` points to your dataset directory. If left unchanged, the script will create a default `data` folder in the repository root. If you use Juliet, no changes are required—the default path is already configured. You can adjust the learning rate, number of epochs, and batch size directly in `scripts/train.sh`.

## generate.py
Use the file *generate.py* to generate 10000 samples of MNIST in the folder samples. You can launch this script on juliet with `scripts/generate.sh`

Example:
  > python3 generate.py --bacth_size 64

or 

```bash
# Make the script executable
chmod +x scripts/generate.sh

# Submit the job to Slurm
scripts/generate.sh
```

## requirements.txt
Among the good pratice of datascience, we encourage you to use conda or virtualenv to create python environment. 
To test your code on our platform, you are required to update the *requirements.txt*, with the different librairies you might use. 
When your code will be test, we will execute: 
  > pip install -r requirements.txt


## Checkpoints
Push the minimal amount of models in the folder *checkpoints*.


---

## Experiments (group 34)

The generator architecture (`Generator` in `model.py`) is unchanged. We study one
training intervention and one inference intervention, at a fixed budget of 100 epochs.

| Lever | Method | Where |
|---|---|---|
| Training | Spectral normalisation of the discriminator (`--spectral_norm`) | `model.py`, `train.py` |
| Training (ablation) | TTUR: `--lr_g 1e-4 --lr_d 4e-4` | `train.py` |
| Inference | Discriminator Rejection Sampling (DRS) | `sampling.py` |
| Inference (comparison) | OBRS at the same budget K, latent truncation | `sampling.py` |
| Inference (ablation) | D recalibrated for the frozen G | `experiments/calibrate_D.py` |

### Code layout

```
model.py            Generator (fixed) + Discriminator (optional spectral norm, logits())
utils.py            training steps, seeds, saving / loading
train.py            training; --run_dir runs/<name> saves config, log.csv, checkpoints, grids
sampling.py         vanilla / truncation / DRS / OBRS samplers
generate.py         platform contract: 10 000 PNGs in samples/ from checkpoints/
evaluation/         MNIST feature extractor, FID, precision, recall (k-NN, k = 3)
experiments/        eval_training.py, eval_sampling.py, calibrate_D.py, make_figures.py
scripts/            launch_training.sh, launch_eval.sh (SLURM, MesoNet)
```

### Reproduce (from the repository root, on Juliet)

```bash
./scripts/launch_training.sh                 # baseline + sn, seeds 0 1 2 (6 jobs, ~20 min each)
CONFIGS="ttur sn_ttur" ./scripts/launch_training.sh   # optional TTUR ablation
./scripts/launch_eval.sh all                 # metrics of all checkpoints + sampling comparison
python experiments/make_figures.py           # figures/platform/ and results/platform/summary.md
```

### Evaluation protocol

- 10 000 generated images vs the 10 000 MNIST test images, quantised to 8 bits like PNG files.
- Default features (`--features platform`), as the course platform: FID on Inception-v3
  features (pytorch-fid weights, 10 000 images); precision / recall (Kynkäänniemi et al., 2019)
  and density / coverage (Naeem et al., 2020) on VGG-16 fc2 features, k = 5, 9 984 images.
  Download the weights once on the login node:
  `python -c "from evaluation.metrics import download_weights; download_weights()"`.
- `--features mnist`: cheap 128-d MNIST-CNN features, k = 3, for quick checks only.
  Results go to `results/<features>/` and figures to `figures/<features>/`: never mix them.
- `python experiments/eval_checkpoint.py --ckpt_dir checkpoints` evaluates one checkpoint folder
  (used to check our numbers against the leaderboard).
- Same latent seed for every evaluation, 3 training seeds per configuration (mean ± std).
- Rejection samplers: budget K = average number of candidates per kept image; DRS and
  OBRS are calibrated to the same acceptance rate 1/K on 10 000 calibration samples.
  We report acceptance rate and total generation time (calibration included).

### Submission

`generate.py` reads the sampling method from `checkpoints/config.json`, e.g.
`"sampling": {"method": "drs", "K": 5}`. To submit a run, copy its
`runs/<name>/checkpoints/{G.pth,D.pth,config.json}` into `checkpoints/`
and add the `"sampling"` entry.
