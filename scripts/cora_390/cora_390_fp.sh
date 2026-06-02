export CUDA_VISIBLE_DEVICES=1

python -u main.py \
    --is_training 1 \
    --train_epochs 2000 \
    --model_id CORA_TEST \
    --dataset cora390 \
    --model SLAPS_FP \
    --input_dim 1433 \
    --hidden_dim 32 \
    --output_dim 7 \
    --r 10 \
    --eta 5 \
    --noise_type "not_zero" \
    --generator FP \
    --gen_input_dim 1433 \
    --gen_layers_size 2 \
    --gen_k 20 \
    --use_gpu True \
    --lr_c 0.01 \
    --lr_DAE 0.01 \
    --dropout_c 0.25 \
    --dropout_DAE 0.5 \
    --lambda_val 100 \
    --itr 10 \
    --patience 500 \
    --is_discrete \
    --hidden_dim_dae 512 \
    --weight_decay_c 0.0005
