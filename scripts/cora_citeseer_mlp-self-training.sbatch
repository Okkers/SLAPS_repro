#!/bin/bash
#SBATCH --job-name=slaps_selftrain_grid
#SBATCH --partition=gpu-a100
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --time=24:00:00
#SBATCH --array=0-23
#SBATCH --output=/scratch/%u/SLAPS_repro/logs/selftrain_grid_%A_%a.out
#SBATCH --error=/scratch/%u/SLAPS_repro/logs/selftrain_grid_%A_%a.err

set -ux

export APPTAINER_CACHEDIR=/scratch/${USER}/apptainer_cache
export APPTAINER_TMPDIR=/scratch/${USER}/apptainer_tmp

WORK_DIR="/scratch/${USER}/SLAPS_repro"
CONTAINER="${WORK_DIR}/containers/slaps.sif"
LOG_DIR="${WORK_DIR}/logs"
RESULTS_BASE="${WORK_DIR}/results/selftraining_grid"
CHECKPOINT_BASE="${WORK_DIR}/checkpoints/selftraining_grid"

mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"
mkdir -p "${LOG_DIR}" "${RESULTS_BASE}" "${CHECKPOINT_BASE}"

cd "${WORK_DIR}"

DATASETS=("cora" "citeseer" "cora390" "citeseer370")
ZETAS=(50 100 200 300 400 500)

DATASET_INDEX=$((SLURM_ARRAY_TASK_ID / 6))
ZETA_INDEX=$((SLURM_ARRAY_TASK_ID % 6))

DATASET="${DATASETS[$DATASET_INDEX]}"
ZETA="${ZETAS[$ZETA_INDEX]}"

case "${DATASET}" in
    cora)
        MODEL_ID="CORA_SELFTRAIN_ZETA${ZETA}"
        INPUT_DIM=1433
        OUTPUT_DIM=7
        GEN_K=20
        NOISE_TYPE="zero"
        PATIENCE=500
        HIDDEN_DIM_DAE=512
        ;;
    citeseer)
        MODEL_ID="CITESEER_SELFTRAIN_ZETA${ZETA}"
        INPUT_DIM=3703
        OUTPUT_DIM=6
        GEN_K=30
        NOISE_TYPE="not_zero"
        PATIENCE=200
        HIDDEN_DIM_DAE=1024
        ;;
    cora390)
        MODEL_ID="CORA390_SELFTRAIN_ZETA${ZETA}"
        INPUT_DIM=1433
        OUTPUT_DIM=7
        GEN_K=20
        NOISE_TYPE="not_zero"
        PATIENCE=500
        HIDDEN_DIM_DAE=512
        ;;
    citeseer370)
        MODEL_ID="CITESEER370_SELFTRAIN_ZETA${ZETA}"
        INPUT_DIM=3703
        OUTPUT_DIM=6
        GEN_K=30
        NOISE_TYPE="not_zero"
        PATIENCE=200
        HIDDEN_DIM_DAE=1024
        ;;
    *)
        echo "Unknown dataset: ${DATASET}"
        exit 1
        ;;
esac

RUN_NAME="${DATASET}_selftraining_zeta${ZETA}"
RESULTS_DIR="${RESULTS_BASE}/${RUN_NAME}"
CHECKPOINT_DIR="${CHECKPOINT_BASE}/${RUN_NAME}"

mkdir -p "${RESULTS_DIR}" "${CHECKPOINT_DIR}"

echo "============================================"
echo " SLAPS self-training grid search"
echo " Dataset:       ${DATASET}"
echo " Zeta:          ${ZETA}"
echo " Model ID:      ${MODEL_ID}"
echo " Input dim:     ${INPUT_DIM}"
echo " Output dim:    ${OUTPUT_DIM}"
echo " gen_k:         ${GEN_K}"
echo " noise_type:    ${NOISE_TYPE}"
echo " patience:      ${PATIENCE}"
echo " hidden_dim_dae:${HIDDEN_DIM_DAE}"
echo " Work dir:      ${WORK_DIR}"
echo " Container:     ${CONTAINER}"
echo " Job ID:        ${SLURM_JOB_ID}"
echo " Array task:    ${SLURM_ARRAY_TASK_ID}"
echo " Node:          ${SLURMD_NODENAME}"
echo " Started:       $(date)"
echo "============================================"

nvidia-smi || true

apptainer exec --nv \
    -B /scratch:/scratch \
    --env PYTHONPATH="${WORK_DIR}" \
    --env PYTHONUNBUFFERED=1 \
    ${CONTAINER} \
    python -u ${WORK_DIR}/main.py \
        --is_training 1 \
        --train_epochs 2000 \
        --model_id "${MODEL_ID}" \
        --dataset "${DATASET}" \
        --model SLAPS_MLP \
        --input_dim "${INPUT_DIM}" \
        --hidden_dim 32 \
        --output_dim "${OUTPUT_DIM}" \
        --r 10 \
        --eta 5 \
        --noise_type "${NOISE_TYPE}" \
        --generator MLP \
        --gen_input_dim "${INPUT_DIM}" \
        --gen_layers_size 2 \
        --gen_k "${GEN_K}" \
        --use_gpu True \
        --gpu 0 \
        --lr_c 0.01 \
        --lr_DAE 0.001 \
        --dropout_c 0.25 \
        --dropout_DAE 0.5 \
        --lambda_val 10 \
        --itr 10 \
        --patience "${PATIENCE}" \
        --is_discrete \
        --weight_decay_c 0.0005 \
        --hidden_dim_dae "${HIDDEN_DIM_DAE}" \
        --self_training \
        --self_training_zeta "${ZETA}" \
        --checkpoints "${CHECKPOINT_DIR}" 

EXIT_CODE=$?

echo "============================================"
echo " Finished: $(date)"
echo " Exit code: ${EXIT_CODE}"
echo " Dataset: ${DATASET}"
echo " Zeta: ${ZETA}"
echo " Results dir: ${RESULTS_DIR}"
echo " Checkpoint dir: ${CHECKPOINT_DIR}"
echo "============================================"

exit ${EXIT_CODE}