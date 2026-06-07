export CUDA_VISIBLE_DEVICES=0

python -u main.py \
    --is_training 1 \
    --train_epochs 2000 \
    --model_id WINE_TEST \
    --dataset wine \
    --model SLAPS_MLP \
    --input_dim 13 \
    --hidden_dim 32 \
    --output_dim 3 \
    --r 5 \
    --eta 5 \
    --noise_type "not_zero" \
    --generator MLP \
    --gen_input_dim 13 \
    --gen_layers_size 2 \
    --gen_k 20 \
    --use_gpu True \
    --lr_c 0.01 \
    --lr_DAE 0.001 \
    --dropout_c 0.5 \
    --dropout_DAE 0.25 \
    --lambda_val 0.1 \
    --itr 10 \
    --patience 2000

