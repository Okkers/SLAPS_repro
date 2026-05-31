export CUDA_VISIBLE_DEVICES=1

python -u main.py \
    --is_training 1 \
    --train_epochs 2000 \
    --model_id CITESEER_TEST \
    --dataset citeseer \
    --model SLAPS_MLPD \
    --input_dim 3703 \
    --hidden_dim 32 \
    --output_dim 6 \
    --r 10 \
    --eta 5 \
    --noise_type "not_zero" \
    --generator MLP-D \
    --gen_input_dim 3703 \
    --gen_layers_size 2 \
    --gen_k 20 \
    --use_gpu True \
    --lr_c 0.001 \
    --lr_DAE 0.01 \
    --dropout_c 0.5 \
    --dropout_DAE 0.5 \
    --lambda_val 10 \
    --itr 10 \
    --patience 200 \
    --is_discrete \
    --weight_decay_c 0.05 \
    --hidden_dim_dae 1024
