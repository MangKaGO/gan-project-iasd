#!/bin/bash
# Submit ONE SLURM job that evaluates everything that is finished in runs/.
# Run from the repository root:
#     ./scripts/launch_eval.sh training     # metrics of every checkpoint (vanilla sampling)
#     ./scripts/launch_eval.sh calibrate    # D_calibrated.pth for every run (ablation)
#     ./scripts/launch_eval.sh sampling     # vanilla / truncation / DRS / OBRS on final checkpoints
#     ./scripts/launch_eval.sh all          # the three, in this order
#     GPU=0 ./scripts/launch_eval.sh all    # CPU only: slower, but no wait for a GPU
#     FEATURES=mnist ./scripts/launch_eval.sh all   # cheap MNIST-CNN metrics instead of
#                                                    # the platform ones (Inception / VGG-16)
# Results are appended to results/*.csv; finished evaluations are skipped.

STAGE=${1:-all}
GPU=${GPU:-1}
FEATURES=${FEATURES:-platform}
if [ "${GPU}" = "0" ]; then
  if [ "${FEATURES}" = "platform" ]; then
    echo "Inception / VGG-16 features are far too slow on CPU: use the GPU, or FEATURES=mnist."
    exit 1
  fi
  RES="#SBATCH -c 16"                       # no GPU line: CPU-only job
  TIME=${TIME:-"04:00:00"}
else
  RES="#SBATCH -c 8
#SBATCH --gres=gpu:1"
  # Inception + VGG-16 on 10 000 images: ~30 s per evaluation, ~200 evaluations.
  if [ "${FEATURES}" = "platform" ]; then TIME=${TIME:-"05:00:00"}; else TIME=${TIME:-"01:30:00"}; fi
fi
project_dir="$(pwd)"
mkdir -p "${project_dir}/outputs" "${project_dir}/results"

case ${STAGE} in
  training)  cmd="python experiments/eval_training.py --features ${FEATURES}" ;;
  calibrate) cmd="for r in runs/*/; do python experiments/calibrate_D.py --run_dir \$r; done" ;;
  sampling)  cmd="python experiments/eval_sampling.py --features ${FEATURES}" ;;
  all)       cmd="python experiments/eval_training.py --features ${FEATURES} && for r in runs/*/; do python experiments/calibrate_D.py --run_dir \$r; done && python experiments/eval_sampling.py --features ${FEATURES}" ;;
  *) echo "Usage: $0 [training|calibrate|sampling|all]"; exit 1 ;;
esac

sbatch <<EOT
#!/bin/bash
#SBATCH -p mesonet
#SBATCH -N 1
${RES}
#SBATCH --mem=32G
#SBATCH --time=${TIME}
#SBATCH --account=m25146
#SBATCH --job-name=eval_${STAGE}_${FEATURES}
#SBATCH --output=${project_dir}/outputs/%x_%j.out
#SBATCH --error=${project_dir}/outputs/%x_%j.err
set -e
cd "${project_dir}"
source "${project_dir}/venv/bin/activate"
export DATA="/projects/m25146/data/"
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-8}
${cmd}
EOT
