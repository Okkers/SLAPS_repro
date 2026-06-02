export CUDA_VISIBLE_DEVICES=1

python -u main.py \
    --is_training 1 \
    --train_epochs 2000 \
    --model_id CITESEER_TEST \
    --dataset citeseer370 \
    --model SLAPS_MLP \
    --input_dim 3703 \
    --hidden_dim 32 \
    --output_dim 6 \
    --r 10 \
    --eta 5 \
    --noise_type "not_zero" \
    --generator MLP \
    --gen_input_dim 3703 \
    --gen_layers_size 2 \
    --gen_k 30 \
    --use_gpu True \
    --lr_c 0.01 \
    --lr_DAE 0.001 \
    --dropout_c 0.25 \
    --dropout_DAE 0.5 \
    --lambda_val 10 \
    --itr 10 \
    --patience 200 \
    --is_discrete \
    --hidden_dim_dae 1024 \
    --weight_decay_c 0.0005




