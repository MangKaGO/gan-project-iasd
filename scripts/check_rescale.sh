#!/bin/bash
# Effect of the pixel conversion on the platform metrics, for one run (~5 min on GPU).
#   clamp   : starter generate.py, x clamped to [0, 1] (negative pixels become black)
#   rescale : (x + 1) / 2, exact inverse of the training normalisation
# Run from the repository root:
#     RUN=sn_paper_ndis5_seed2 bash scripts/check_rescale.sh
RUN=${RUN:-sn_paper_ndis5_seed2}
project_dir="$(pwd)"
mkdir -p "${project_dir}/outputs"
sbatch <<EOT
#!/bin/bash
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --account=m25146
#SBATCH --job-name=rescale_${RUN}
#SBATCH --output=${project_dir}/outputs/%x_%j.out
#SBATCH --error=${project_dir}/outputs/%x_%j.err
cd "${project_dir}"
source "${project_dir}/venv/bin/activate"
export DATA="/projects/m25146/data/"
C=runs/${RUN}/checkpoints
for opt in "" "--rescale"; do
  echo "== vanilla \${opt}";        python experiments/eval_checkpoint.py --ckpt_dir \$C \${opt}
  echo "== truncation 1.0 \${opt}"; python experiments/eval_checkpoint.py --ckpt_dir \$C --method truncation --psi 1.0 \${opt}
  echo "== truncation 1.5 \${opt}"; python experiments/eval_checkpoint.py --ckpt_dir \$C --method truncation --psi 1.5 \${opt}
done
EOT
