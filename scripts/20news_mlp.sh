export CUDA_VISIBLE_DEVICES=1

python -u main.py \
    --is_training 1 \
    --train_epochs 2000 \
    --model_id 20NEWS_TEST \
    --dataset 20news \
    --model SLAPS_MLP \
    --input_dim 236 \
    --hidden_dim 32 \
    --output_dim 10 \
    --r 5 \
    --eta 5 \
    --noise_type "not_zero" \
    --generator MLP \
    --gen_input_dim 236 \
    --gen_layers_size 2 \
    --gen_k 20 \
    --use_gpu True \
    --lr_c 0.001 \
    --lr_DAE 0.001 \
    --dropout_c 0.25 \
    --dropout_DAE 0.5 \
    --lambda_val 500 \
    --itr 10 \
    --patience 15
