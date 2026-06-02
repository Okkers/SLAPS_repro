export CUDA_VISIBLE_DEVICES=1

python -u main.py \
    --is_training 1 \
    --train_epochs 2000 \
    --model_id CITESEER_TEST0 \
    --dataset citeseer \
    --model SLAPS_FP \
    --input_dim 3703 \
    --hidden_dim 32 \
    --output_dim 6 \
    --r 10 \
    --eta 1 \
    --noise_type "not_zero" \
    --generator FP \
    --gen_input_dim 3703 \
    --gen_layers_size 2 \
    --gen_k 30 \
    --use_gpu True \
    --lr_c 0.01 \
    --lr_DAE 0.01 \
    --dropout_c 0.5 \
    --dropout_DAE 0.5 \
    --lambda_val 1.0 \
    --itr 10 \
    --patience 200 \
    --is_discrete \
    --hidden_dim_dae 1024 \
    --weight_decay_c 0.05
