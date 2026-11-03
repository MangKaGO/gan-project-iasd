#!/bin/bash
# Submit one SLURM job per (training configuration, seed), all at once.
# Run from the repository root:
#     ./scripts/launch_training.sh                 # baseline and sn, seeds 0 1 2
#     CONFIGS="ttur sn_ttur" ./scripts/launch_training.sh   # optional TTUR ablation
#     CONFIGS="sn_ndis5" TIME="03:00:00" ./scripts/launch_training.sh   # SN, 5 D steps per G step
#     CONFIGS="baseline_paper sn_paper" ./scripts/launch_training.sh    # hinge + beta1 0.5, +/- SN
#     CONFIGS="sn_paper_ndis5" TIME="03:00:00" ./scripts/launch_training.sh   # + 5 D steps per G step
#     CONFIGS="sn_paperA_ndis5 baseline_paperA_ndis5" TIME="03:00:00" ./scripts/launch_training.sh
#         exact setting A of the SN paper, with and without SN
#     CONFIGS="baseline_paper" EPOCHS=400 TIME="01:30:00" ./scripts/launch_training.sh
#         longer training: runs go to runs/<config>_ep400_seed<k>/ and never overwrite the 100-epoch runs
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
  epochs=${EPOCHS}
  case ${cfg} in
    baseline) opts="" ;;
    sn)       opts="--spectral_norm" ;;
    ttur)     opts="--lr_g 0.0001 --lr_d 0.0004" ;;
    sn_ttur)  opts="--spectral_norm --lr_g 0.0001 --lr_d 0.0004" ;;
    # SN with 5 D updates per G update (Miyato et al.). G is updated once every
    # 5 batches, so 5x more epochs give G the same number of updates as the others.
    sn_ndis5) opts="--spectral_norm --n_dis 5 --save_every 50"; epochs=$((EPOCHS * 5)) ;;
    # Setting of the spectral-norm paper (hinge loss, Adam beta1 = 0.5), with and without SN.
    baseline_paper) opts="--loss hinge --beta1 0.5" ;;
    sn_paper)       opts="--spectral_norm --loss hinge --beta1 0.5" ;;
    # Same + 5 D updates per G update (setting "A" of the paper); 5x more epochs so
    # that G gets the same number of updates (~94k) as every other run.
    sn_paper_ndis5) opts="--spectral_norm --loss hinge --beta1 0.5 --n_dis 5 --save_every 50"; epochs=$((EPOCHS * 5)) ;;
    # Exact setting A of Table 1 of the SN paper (lr 1e-4, beta1 0.5, beta2 0.9, n_dis 5),
    # with and without SN (the second one isolates the effect of the 5 D steps).
    sn_paperA_ndis5)       opts="--spectral_norm --loss hinge --lr 0.0001 --beta1 0.5 --beta2 0.9 --n_dis 5 --save_every 50"; epochs=$((EPOCHS * 5)) ;;
    baseline_paperA_ndis5) opts="--loss hinge --lr 0.0001 --beta1 0.5 --beta2 0.9 --n_dis 5 --save_every 50"; epochs=$((EPOCHS * 5)) ;;
    *) echo "Unknown configuration: ${cfg}"; exit 1 ;;
  esac

  for seed in ${SEEDS}; do
    if [ "${EPOCHS}" = "100" ]; then run="${cfg}_seed${seed}"; else run="${cfg}_ep${EPOCHS}_seed${seed}"; fi
    if [ -f "runs/${run}/checkpoints/G_epoch$(printf %03d ${epochs}).pth" ]; then
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
python train.py --epochs ${epochs} --batch_size 64 --gpus 1 --seed ${seed} --save_every ${SAVE_EVERY:-10} ${opts} --run_dir runs/${run}
EOT
  done
done
