#!/bin/bash
# Submit ONE SLURM job that evaluates everything that is finished in runs/.
# Run from the repository root:
#     ./scripts/launch_eval.sh training     # metrics of every checkpoint (vanilla sampling)
#     ./scripts/launch_eval.sh calibrate    # D_calibrated.pth for every run (ablation)
#     ./scripts/launch_eval.sh scratch      # D_scratch.pth: a NEW D trained from scratch against
#                                           # each frozen G (runs of SCRATCH_CONFIGS only), then
#                                           # DRS with it: RESCALE=1 bash scripts/launch_eval.sh scratch
#     ./scripts/launch_eval.sh sampling     # vanilla / truncation / DRS / OBRS on final checkpoints
#     ./scripts/launch_eval.sh all          # the three, in this order
#     GPU=0 ./scripts/launch_eval.sh all    # CPU only: slower, but no wait for a GPU
#     FEATURES=mnist ./scripts/launch_eval.sh all   # cheap MNIST-CNN metrics instead of
#                                                    # the platform ones (Inception / VGG-16)
#     RESCALE=1 QUICK=1 ./scripts/launch_eval.sh all   # images saved as (x+1)/2 instead of
#                                           # clamped; results in results/<features>_rescale/
#     QUICK=1 ./scripts/launch_eval.sh all  # reduced plan (~40 min on GPU): epochs 50 and 100,
#                                           # sampling on baseline and sn_ttur only, DRS and
#                                           # truncation, K in {2, 5, 10}
# Results are appended to results/*.csv; finished evaluations are skipped.

STAGE=${1:-all}
USER_TIME=${TIME}                           # a TIME given on the command line always wins
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
if [ "${QUICK:-0}" = "1" ]; then
  TRAIN_OPTS="--every 50"
  SAMPLE_OPTS="--configs baseline baseline_paper sn_paper sn_paper_ndis5 baseline_paper_ndis5 baseline_paper_ep500 sn_paperA_ndis5 baseline_paperA_ndis5 --samplers vanilla truncation drs --Ks 2 5 10 --psis 1.5 1.0 0.5"
  TIME=${USER_TIME:-"02:00:00"}
fi
if [ "${RESCALE:-0}" = "1" ]; then
  TRAIN_OPTS="${TRAIN_OPTS} --rescale"; SAMPLE_OPTS="${SAMPLE_OPTS} --rescale"; TAG="_rescale"; RESC="--rescale"
fi
# Runs that get a discriminator trained from scratch (~5 min per run on GPU).
SCRATCH_CONFIGS=${SCRATCH_CONFIGS:-"baseline baseline_paper sn_paper_ndis5"}
scratch_runs=""
for c in ${SCRATCH_CONFIGS}; do scratch_runs="${scratch_runs} runs/${c}_seed*/"; done
project_dir="$(pwd)"
mkdir -p "${project_dir}/outputs" "${project_dir}/results"

case ${STAGE} in
  training)  cmd="python experiments/eval_training.py --features ${FEATURES} ${TRAIN_OPTS}" ;;
  calibrate) cmd="for r in runs/*/; do python experiments/calibrate_D.py --run_dir \$r; done" ;;
  scratch)   cmd="for r in ${scratch_runs}; do python experiments/calibrate_D.py --run_dir \$r --from_scratch; done && python experiments/eval_sampling.py --features ${FEATURES} --configs ${SCRATCH_CONFIGS} --samplers vanilla drs --Ks 2 5 10 ${RESC}" ;;
  sampling)  cmd="python experiments/eval_sampling.py --features ${FEATURES} ${SAMPLE_OPTS}" ;;
  all)       cmd="python experiments/eval_training.py --features ${FEATURES} ${TRAIN_OPTS} && for r in runs/*/; do python experiments/calibrate_D.py --run_dir \$r; done && python experiments/eval_sampling.py --features ${FEATURES} ${SAMPLE_OPTS}" ;;
  *) echo "Usage: $0 [training|calibrate|scratch|sampling|all]"; exit 1 ;;
esac

sbatch <<EOT
#!/bin/bash
#SBATCH -p mesonet
#SBATCH -N 1
${RES}
#SBATCH --mem=32G
#SBATCH --time=${TIME}
#SBATCH --account=m25146
#SBATCH --job-name=eval_${STAGE}_${FEATURES}${TAG}
#SBATCH --output=${project_dir}/outputs/%x_%j.out
#SBATCH --error=${project_dir}/outputs/%x_%j.err
set -e
cd "${project_dir}"
source "${project_dir}/venv/bin/activate"
export DATA="/projects/m25146/data/"
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-8}
${cmd}
EOT
