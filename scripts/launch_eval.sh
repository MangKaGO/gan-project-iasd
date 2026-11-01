#!/bin/bash
# Submit ONE SLURM job that evaluates everything that is finished in runs/.
# Run from the repository root:
#     ./scripts/launch_eval.sh training     # metrics of every checkpoint (vanilla sampling)
#     ./scripts/launch_eval.sh calibrate    # D_calibrated.pth for every run (ablation)
#     ./scripts/launch_eval.sh sampling     # vanilla / truncation / DRS / OBRS on final checkpoints
#     ./scripts/launch_eval.sh all          # the three, in this order
# Results are appended to results/*.csv; finished evaluations are skipped.

STAGE=${1:-all}
TIME=${TIME:-"01:30:00"}
project_dir="$(pwd)"
mkdir -p "${project_dir}/outputs" "${project_dir}/results"

case ${STAGE} in
  training)  cmd="python experiments/eval_training.py" ;;
  calibrate) cmd="for r in runs/*/; do python experiments/calibrate_D.py --run_dir \$r; done" ;;
  sampling)  cmd="python experiments/eval_sampling.py" ;;
  all)       cmd="python experiments/eval_training.py && for r in runs/*/; do python experiments/calibrate_D.py --run_dir \$r; done && python experiments/eval_sampling.py" ;;
  *) echo "Usage: $0 [training|calibrate|sampling|all]"; exit 1 ;;
esac

sbatch <<EOT
#!/bin/bash
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=${TIME}
#SBATCH --account=m25146
#SBATCH --job-name=eval_${STAGE}
#SBATCH --output=${project_dir}/outputs/%x_%j.out
#SBATCH --error=${project_dir}/outputs/%x_%j.err
set -e
cd "${project_dir}"
source "${project_dir}/venv/bin/activate"
export DATA="/projects/m25146/data/"
${cmd}
EOT
