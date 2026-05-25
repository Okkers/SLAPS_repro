export CUDA_VISIBLE_DEVICES=1

python -u main.py \
    --is_training 1 \
    --train_epochs 2000 \
    --model_id WINE_TEST \
    --dataset wine \
    --model SLAPS_MLPD \
    --input_dim 13 \
    --hidden_dim 32 \
    --output_dim 3 \
    --r 5 \
    --eta 5 \
    --noise_type "not_zero" \
    --generator MLP-D \
    --gen_input_dim 13 \
    --gen_layers_size 2 \
    --gen_k 10 \
    --use_gpu True \
    --lr_c 0.01 \
    --lr_DAE 0.01 \
    --dropout_c 0.25 \
    --dropout_DAE 0.5 \
    --lambda_val 1.0 \
    --itr 10 \
    --patience 15
