export CUDA_VISIBLE_DEVICES=1

python -u main.py \
    --is_training 1 \
    --train_epochs 2000 \
    --model_id DIGITS_TEST \
    --dataset digits \
    --model SLAPS_FP \
    --input_dim 64 \
    --hidden_dim 32 \
    --output_dim 10 \
    --r 5 \
    --eta 5 \
    --noise_type "not_zero" \
    --generator FP \
    --gen_input_dim 64 \
    --gen_layers_size 2 \
    --gen_k 20 \
    --use_gpu True \
    --lr_c 0.01 \
    --lr_DAE 0.001 \
    --dropout_c 0.25 \
    --dropout_DAE 0.5 \
    --lambda_val 0.1 \
    --itr 10 \
    --patience 15
