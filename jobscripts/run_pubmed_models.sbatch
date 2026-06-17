#!/bin/bash
#SBATCH --job-name=pubmed_slaps_models
#SBATCH --partition=gpu-a100
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/%u/SLAPS_repro/logs/pubmed_models_%A_%a.out
#SBATCH --error=/scratch/%u/SLAPS_repro/logs/pubmed_models_%A_%a.err

set -ux

export APPTAINER_CACHEDIR=/scratch/${USER}/apptainer_cache
export APPTAINER_TMPDIR=/scratch/${USER}/apptainer_tmp

WORK_DIR="/scratch/${USER}/SLAPS_repro"
CONTAINER="${WORK_DIR}/containers/slaps.sif"
LOG_DIR="${WORK_DIR}/logs"
CHECKPOINT_BASE="${WORK_DIR}/checkpoints/pubmed_models"
RESULTS_BASE="${WORK_DIR}/results/pubmed_models"

mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"
mkdir -p "${LOG_DIR}" "${CHECKPOINT_BASE}" "${RESULTS_BASE}"

cd "${WORK_DIR}"

if [ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]; then
    MODEL_ID="PUBMED_MLPD_TEST"
    MODEL="SLAPS_MLPD"
    GENERATOR="MLP-D"
    LR_DAE=0.01
    DROPOUT_C=0.25
    DROPOUT_DAE=0.25
    LAMBDA_VAL=100
    RUN_NAME="pubmed_slaps_mlpd"
elif [ "${SLURM_ARRAY_TASK_ID}" -eq 1 ]; then
    MODEL_ID="PUBMED_MLP_TEST"
    MODEL="SLAPS_MLP"
    GENERATOR="MLP"
    LR_DAE=0.01
    DROPOUT_C=0.5
    DROPOUT_DAE=0.5
    LAMBDA_VAL=10
    RUN_NAME="pubmed_slaps_mlp"
else
    echo "Unknown array task id: ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

CHECKPOINT_DIR="${CHECKPOINT_BASE}/${RUN_NAME}"
RESULTS_DIR="${RESULTS_BASE}/${RUN_NAME}"

mkdir -p "${CHECKPOINT_DIR}" "${RESULTS_DIR}"

echo "============================================"
echo " PubMed SLAPS model run"
echo " Model ID:      ${MODEL_ID}"
echo " Model:         ${MODEL}"
echo " Generator:     ${GENERATOR}"
echo " Work dir:      ${WORK_DIR}"
echo " Container:     ${CONTAINER}"
echo " Job ID:        ${SLURM_JOB_ID}"
echo " Array task:    ${SLURM_ARRAY_TASK_ID}"
echo " Node:          ${SLURMD_NODENAME}"
echo " Started:       $(date)"
echo " Checkpoints:   ${CHECKPOINT_DIR}"
echo " Results:       ${RESULTS_DIR}"
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
        --model "${MODEL}" \
        --input_dim 500 \
        --hidden_dim 32 \
        --output_dim 3 \
        --r 10 \
        --eta 5 \
        --noise_type "zero" \
        --generator "${GENERATOR}" \
        --gen_input_dim 500 \
        --gen_layers_size 2 \
        --gen_k 15 \
        --use_gpu True \
        --gpu 0 \
        --lr_c 0.01 \
        --lr_DAE "${LR_DAE}" \
        --dropout_c "${DROPOUT_C}" \
        --dropout_DAE "${DROPOUT_DAE}" \
        --lambda_val "${LAMBDA_VAL}" \
        --itr 10 \
        --patience 500 \
        --weight_decay_c 0.0005 \
        --hidden_dim_dae 128 \
        --checkpoints "${CHECKPOINT_DIR}"

EXIT_CODE=$?

echo "============================================"
echo " Finished: $(date)"
echo " Exit code: ${EXIT_CODE}"
echo " Model ID: ${MODEL_ID}"
echo " Results dir: ${RESULTS_DIR}"
echo " Checkpoint dir: ${CHECKPOINT_DIR}"
echo "============================================"

exit ${EXIT_CODE}