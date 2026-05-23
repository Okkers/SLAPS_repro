export CUDA_VISIBLE_DEVICES=1

python -u main.py \
    --is_training 1 \
    --model_id TESTER_MODEL \
    --model SLAPS \
    --input_dim 13 \
    --hidden_dim 32 \
    --output_dim 3 \
    --is_discrete True \
    --r 5 \
    --eta 5 \
    --noise_type "0" \
    --generator "MLP-D" \
    --gen_input_dim 13 \
    --gen_layers_size 13 \
    --gen_k 10 \
    --use_gpu True \
    --lr_c 0.01 \
    --lr_DAE 0.01 \
    --dropout_c 0.25 \
    --dropout_DAE 0.5 \
    --lambda 1.0 \
