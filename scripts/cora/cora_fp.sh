export CUDA_VISIBLE_DEVICES=1

python -u main.py \
    --is_training 1 \
    --train_epochs 2000 \
    --model_id CORA_TEST \
    --dataset cora \
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
    --gen_k 30 \
    --use_gpu True \
    --lr_c 0.001 \
    --lr_DAE 0.01 \
    --dropout_c 0.5 \
    --dropout_DAE 0.25 \
    --lambda_val 10 \
    --itr 10 \
    --patience 500
