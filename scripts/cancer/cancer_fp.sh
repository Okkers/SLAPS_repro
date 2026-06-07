export CUDA_VISIBLE_DEVICES=0

python -u main.py \
    --is_training 1 \
    --train_epochs 2000 \
    --model_id CANCER_TEST \
    --dataset cancer \
    --model SLAPS_FP \
    --input_dim 30 \
    --hidden_dim 32 \
    --output_dim 2 \
    --r 20 \
    --eta 5 \
    --noise_type "not_zero" \
    --generator FP \
    --gen_input_dim 30 \
    --gen_layers_size 2 \
    --gen_k 20 \
    --use_gpu True \
    --lr_c 0.01 \
    --lr_DAE 0.001 \
    --dropout_c 0.5 \
    --dropout_DAE 0.25 \
    --lambda_val 0.1 \
    --itr 10 \
    --patience 500 \
    --hidden_dim_dae 64 \
    --weight_decay_c 0.002

