#!/bin/sh
#SBATCH --job-name=digits_slaps_adaedge
#SBATCH --partition=gpu-a100
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/SLAPS_repro/logs/digits_slaps_adaedge_%j.out
#SBATCH --error=/scratch/%u/SLAPS_repro/logs/digits_slaps_adaedge_%j.err

set -ux

# Use /scratch for temporary/cache storage.
export APPTAINER_CACHEDIR=/scratch/${USER}/apptainer_cache
export APPTAINER_TMPDIR=/scratch/${USER}/apptainer_tmp

mkdir -p ${APPTAINER_CACHEDIR} ${APPTAINER_TMPDIR}

WORK_DIR="/scratch/${USER}/SLAPS_repro"
CONTAINER="${WORK_DIR}/containers/slaps.sif"
RESULTS_DIR="${WORK_DIR}/results"
LOG_DIR="${WORK_DIR}/logs"
CHECKPOINT_DIR="${WORK_DIR}/checkpoints"

mkdir -p ${LOG_DIR} ${RESULTS_DIR} ${CHECKPOINT_DIR}

cd ${WORK_DIR}

echo "============================================"
echo " SLAPS + AdaEdge experiment"
echo " Work dir:    ${WORK_DIR}"
echo " Container:   ${CONTAINER}"
echo " Job ID:      ${SLURM_JOB_ID}"
echo " Node:        ${SLURMD_NODENAME}"
echo " Started:     $(date)"
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
        --model_id DIGITS_TEST \
        --dataset digits \
        --model SLAPS_MLP \
        --input_dim 64 \
        --hidden_dim 32 \
        --output_dim 10 \
        --r 5 \
        --eta 5 \
        --noise_type "not_zero" \
        --generator MLP \
        --gen_input_dim 64 \
        --gen_layers_size 2 \
        --gen_k 20 \
        --use_gpu True \
        --lr_c 0.01 \
        --lr_DAE 0.001 \
        --dropout_c 0.25 \
        --dropout_DAE 0.5 \
        --lambda_val 10 \
        --itr 10 \
        --patience 300 \
        --ada_edge \
        --conf_threshold 0.9 \
        --max_add 10 \
        --max_remove 10 \
        --max_iter 10 \
        --order "add" \
        --gpu 0 
        

EXIT_CODE=$?

echo "============================================"
echo " Finished: $(date)"
echo " Exit code: ${EXIT_CODE}"
echo " Results dir: ${RESULTS_DIR}"
echo " Logs:"
echo "   stdout: ${LOG_DIR}/digits_slaps_adaedge_${SLURM_JOB_ID}.out"
echo "   stderr: ${LOG_DIR}/digits_slaps_adaedge_${SLURM_JOB_ID}.err"
echo "============================================"

exit ${EXIT_CODE}