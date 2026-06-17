#!/bin/bash
#SBATCH --job-name=pubmed_selftrain_grid
#SBATCH --partition=gpu-a100
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --time=24:00:00
#SBATCH --array=0-5
#SBATCH --output=/scratch/%u/SLAPS_repro/logs/pubmed_selftrain_%A_%a.out
#SBATCH --error=/scratch/%u/SLAPS_repro/logs/pubmed_selftrain_%A_%a.err

set -ux

export APPTAINER_CACHEDIR=/scratch/${USER}/apptainer_cache
export APPTAINER_TMPDIR=/scratch/${USER}/apptainer_tmp

WORK_DIR="/scratch/${USER}/SLAPS_repro"
CONTAINER="${WORK_DIR}/containers/slaps.sif"
LOG_DIR="${WORK_DIR}/logs"
RESULTS_BASE="${WORK_DIR}/results/pubmed_selftraining_grid"
CHECKPOINT_BASE="${WORK_DIR}/checkpoints/pubmed_selftraining_grid"

mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"
mkdir -p "${LOG_DIR}" "${RESULTS_BASE}" "${CHECKPOINT_BASE}"

cd "${WORK_DIR}"

ZETAS=(50 100 200 300 400 500)
ZETA="${ZETAS[$SLURM_ARRAY_TASK_ID]}"

RUN_NAME="pubmed_selftraining_zeta${ZETA}"
MODEL_ID="PUBMED_SELFTRAIN_ZETA${ZETA}"
RESULTS_DIR="${RESULTS_BASE}/${RUN_NAME}"
CHECKPOINT_DIR="${CHECKPOINT_BASE}/${RUN_NAME}"

mkdir -p "${RESULTS_DIR}" "${CHECKPOINT_DIR}"

echo "============================================"
echo " PubMed SLAPS self-training grid"
echo " Zeta:          ${ZETA}"
echo " Model ID:      ${MODEL_ID}"
echo " Work dir:      ${WORK_DIR}"
echo " Container:     ${CONTAINER}"
echo " Job ID:        ${SLURM_JOB_ID}"
echo " Array task:    ${SLURM_ARRAY_TASK_ID}"
echo " Node:          ${SLURMD_NODENAME}"
echo " Started:       $(date)"
echo " Results dir:   ${RESULTS_DIR}"
echo " Checkpoints:   ${CHECKPOINT_DIR}"
echo "============================================"

nvidia-smi || true

echo "Checking PubMed data files..."
ls -lh "${WORK_DIR}/data/ind.pubmed."* || {
    echo "ERROR: PubMed data files not found in ${WORK_DIR}/data"
    echo "Expected files like:"
    echo "  ${WORK_DIR}/data/ind.pubmed.x"
    echo "  ${WORK_DIR}/data/ind.pubmed.y"
    echo "  ${WORK_DIR}/data/ind.pubmed.tx"
    echo "  ${WORK_DIR}/data/ind.pubmed.ty"
    echo "  ${WORK_DIR}/data/ind.pubmed.allx"
    echo "  ${WORK_DIR}/data/ind.pubmed.ally"
    echo "  ${WORK_DIR}/data/ind.pubmed.graph"
    echo "  ${WORK_DIR}/data/ind.pubmed.test.index"
    exit 1
}

# IMPORTANT:
# Stay in WORK_DIR so relative paths like data/ind.pubmed.x work correctly.
cd "${WORK_DIR}"

apptainer exec --nv \
    -B /scratch:/scratch \
    --env PYTHONPATH="${WORK_DIR}" \
    --env PYTHONUNBUFFERED=1 \
    "${CONTAINER}" \
    python -u "${WORK_DIR}/main.py" \
        --is_training 1 \
        --train_epochs 2000 \
        --model_id "${MODEL_ID}" \
        --dataset pubmed \
        --model SLAPS_MLP \
        --input_dim 500 \
        --hidden_dim 32 \
        --output_dim 3 \
        --r 10 \
        --eta 5 \
        --noise_type "zero" \
        --generator MLP \
        --gen_input_dim 500 \
        --gen_layers_size 2 \
        --gen_k 15 \
        --use_gpu True \
        --gpu 0 \
        --lr_c 0.01 \
        --lr_DAE 0.01 \
        --dropout_c 0.5 \
        --dropout_DAE 0.5 \
        --lambda_val 10 \
        --itr 10 \
        --patience 500 \
        --weight_decay_c 0.0005 \
        --hidden_dim_dae 128 \
        --self_training \
        --self_training_zeta "${ZETA}" \
        --checkpoints "${CHECKPOINT_DIR}"

EXIT_CODE=$?

echo "============================================"
echo " Finished: $(date)"
echo " Exit code: ${EXIT_CODE}"
echo " Zeta: ${ZETA}"
echo " Results dir: ${RESULTS_DIR}"
echo " Checkpoint dir: ${CHECKPOINT_DIR}"
echo "============================================"

exit ${EXIT_CODE}