#!/bin/bash
#SBATCH --job-name=slaps_adaedge_grid
#SBATCH --partition=gpu-a100
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --time=24:00:00
#SBATCH --array=0-23
#SBATCH --output=/scratch/%u/SLAPS_repro/logs/adaedge_grid_%A_%a.out
#SBATCH --error=/scratch/%u/SLAPS_repro/logs/adaedge_grid_%A_%a.err

set -ux

export APPTAINER_CACHEDIR=/scratch/${USER}/apptainer_cache
export APPTAINER_TMPDIR=/scratch/${USER}/apptainer_tmp

WORK_DIR="/scratch/${USER}/SLAPS_repro"
CONTAINER="${WORK_DIR}/containers/slaps.sif"
LOG_DIR="${WORK_DIR}/logs"
RESULTS_BASE="${WORK_DIR}/results/adaedge_grid"
CHECKPOINT_BASE="${WORK_DIR}/checkpoints/adaedge_grid"

mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"
mkdir -p "${LOG_DIR}" "${RESULTS_BASE}" "${CHECKPOINT_BASE}"

cd "${WORK_DIR}"

DATASETS=("cora" "citeseer" "cora390" "citeseer370")
CONFIG_NAMES=("conservative" "threshold_08" "threshold_095" "more_edges_50" "add_only" "remove_first")

DATASET_INDEX=$((SLURM_ARRAY_TASK_ID / 6))
CONFIG_INDEX=$((SLURM_ARRAY_TASK_ID % 6))

DATASET="${DATASETS[$DATASET_INDEX]}"
CONFIG_NAME="${CONFIG_NAMES[$CONFIG_INDEX]}"

case "${DATASET}" in
    cora)
        MODEL_ID_PREFIX="CORA_ADAEDGE"
        INPUT_DIM=1433
        OUTPUT_DIM=7
        GEN_K=20
        NOISE_TYPE="zero"
        PATIENCE=500
        HIDDEN_DIM_DAE=512
        ;;
    citeseer)
        MODEL_ID_PREFIX="CITESEER_ADAEDGE"
        INPUT_DIM=3703
        OUTPUT_DIM=6
        GEN_K=30
        NOISE_TYPE="not_zero"
        PATIENCE=200
        HIDDEN_DIM_DAE=1024
        ;;
    cora390)
        MODEL_ID_PREFIX="CORA390_ADAEDGE"
        INPUT_DIM=1433
        OUTPUT_DIM=7
        GEN_K=20
        NOISE_TYPE="not_zero"
        PATIENCE=500
        HIDDEN_DIM_DAE=512
        ;;
    citeseer370)
        MODEL_ID_PREFIX="CITESEER370_ADAEDGE"
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

case "${CONFIG_NAME}" in
    conservative)
        CONF_THRESHOLD=0.9
        MAX_ADD=10
        MAX_REMOVE=10
        MAX_ITER=10
        ORDER="add"
        ;;
    threshold_08)
        CONF_THRESHOLD=0.8
        MAX_ADD=10
        MAX_REMOVE=10
        MAX_ITER=10
        ORDER="add"
        ;;
    threshold_095)
        CONF_THRESHOLD=0.95
        MAX_ADD=10
        MAX_REMOVE=10
        MAX_ITER=10
        ORDER="add"
        ;;
    more_edges_50)
        CONF_THRESHOLD=0.9
        MAX_ADD=50
        MAX_REMOVE=50
        MAX_ITER=10
        ORDER="add"
        ;;
    add_only)
        CONF_THRESHOLD=0.9
        MAX_ADD=50
        MAX_REMOVE=0
        MAX_ITER=10
        ORDER="add"
        ;;
    remove_first)
        CONF_THRESHOLD=0.9
        MAX_ADD=10
        MAX_REMOVE=10
        MAX_ITER=10
        ORDER="remove"
        ;;
    *)
        echo "Unknown AdaEdge config: ${CONFIG_NAME}"
        exit 1
        ;;
esac

RUN_NAME="${DATASET}_adaedge_${CONFIG_NAME}"
MODEL_ID="${MODEL_ID_PREFIX}_${CONFIG_NAME}"
RESULTS_DIR="${RESULTS_BASE}/${RUN_NAME}"
CHECKPOINT_DIR="${CHECKPOINT_BASE}/${RUN_NAME}"

mkdir -p "${RESULTS_DIR}" "${CHECKPOINT_DIR}"

echo "============================================"
echo " SLAPS AdaEdge grid search"
echo " Dataset:        ${DATASET}"
echo " Config:         ${CONFIG_NAME}"
echo " Model ID:       ${MODEL_ID}"
echo " Input dim:      ${INPUT_DIM}"
echo " Output dim:     ${OUTPUT_DIM}"
echo " gen_k:          ${GEN_K}"
echo " noise_type:     ${NOISE_TYPE}"
echo " patience:       ${PATIENCE}"
echo " hidden_dim_dae: ${HIDDEN_DIM_DAE}"
echo " conf_threshold: ${CONF_THRESHOLD}"
echo " max_add:        ${MAX_ADD}"
echo " max_remove:     ${MAX_REMOVE}"
echo " max_iter:       ${MAX_ITER}"
echo " order:          ${ORDER}"
echo " Work dir:       ${WORK_DIR}"
echo " Container:      ${CONTAINER}"
echo " Job ID:         ${SLURM_JOB_ID}"
echo " Array task:     ${SLURM_ARRAY_TASK_ID}"
echo " Node:           ${SLURMD_NODENAME}"
echo " Started:        $(date)"
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
        --ada_edge \
        --conf_threshold "${CONF_THRESHOLD}" \
        --max_add "${MAX_ADD}" \
        --max_remove "${MAX_REMOVE}" \
        --max_iter "${MAX_ITER}" \
        --order "${ORDER}" \
        --checkpoints "${CHECKPOINT_DIR}"

EXIT_CODE=$?

echo "============================================"
echo " Finished: $(date)"
echo " Exit code: ${EXIT_CODE}"
echo " Dataset: ${DATASET}"
echo " AdaEdge config: ${CONFIG_NAME}"
echo " Results dir: ${RESULTS_DIR}"
echo " Checkpoint dir: ${CHECKPOINT_DIR}"
echo "============================================"

exit ${EXIT_CODE}