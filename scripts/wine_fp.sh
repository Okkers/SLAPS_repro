export CUDA_VISIBLE_DEVICES=1

python -u main.py \
    --is_training 1 \
    --model_id TESTER_MODEL \
    --dataset wine \
    --model SLAPS \
    --train_epochs 100 \
    --input_dim 13 \
    --hidden_dim 32 \
    --output_dim 3 \
    --is_discrete True \
    --r 5 \
    --eta 5 \
    --noise_type "0" \
    --generator "FP" \
    --gen_input_dim 13 \
    --gen_layers_size 178 \
    --gen_k 20 \
    --use_gpu True \
    --lr_c 0.01 \
    --lr_DAE 0.001 \
    --dropout_c 0.5 \
    --dropout_DAE 0.5 \
    --lambda_val 0.1 \
