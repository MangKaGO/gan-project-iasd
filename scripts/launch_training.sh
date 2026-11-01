#!/bin/bash
# Submit one SLURM job per (training configuration, seed), all at once.
# Run from the repository root:
#     ./scripts/launch_training.sh                 # baseline and sn, seeds 0 1 2
#     CONFIGS="ttur sn_ttur" ./scripts/launch_training.sh   # optional TTUR ablation
#
# Each run writes into runs/<config>_seed<seed>/ (see train.py).

CONFIGS=${CONFIGS:-"baseline sn"}
SEEDS=${SEEDS:-"0 1 2"}
EPOCHS=${EPOCHS:-100}
TIME=${TIME:-"00:45:00"}          # 100 epochs take about 20 min on one GPU

project_dir="$(pwd)"
mkdir -p "${project_dir}/outputs" "${project_dir}/runs"

for cfg in ${CONFIGS}; do
  # Training options of each configuration (lr = 2e-4 unless TTUR).
  case ${cfg} in
    baseline) opts="" ;;
    sn)       opts="--spectral_norm" ;;
    ttur)     opts="--lr_g 0.0001 --lr_d 0.0004" ;;
    sn_ttur)  opts="--spectral_norm --lr_g 0.0001 --lr_d 0.0004" ;;
    *) echo "Unknown configuration: ${cfg}"; exit 1 ;;
  esac

  for seed in ${SEEDS}; do
    run="${cfg}_seed${seed}"
    if [ -f "runs/${run}/checkpoints/G_epoch$(printf %03d ${EPOCHS}).pth" ]; then
      echo "Skipping ${run}: already finished."
      continue
    fi
    echo "Submitting ${run}"
    sbatch <<EOT
#!/bin/bash
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=${TIME}
#SBATCH --account=m25146
#SBATCH --job-name=${run}
#SBATCH --output=${project_dir}/outputs/%x_%j.out
#SBATCH --error=${project_dir}/outputs/%x_%j.err
set -e
cd "${project_dir}"
source "${project_dir}/venv/bin/activate"
export DATA="/projects/m25146/data/"
python train.py --epochs ${EPOCHS} --batch_size 64 --gpus 1 --seed ${seed} ${opts} --run_dir runs/${run}
EOT
  done
done
