export CUDA_VISIBLE_DEVICES=1

python -u main.py \
    --is_training 0 \
    --model_id TESTER_MODEL \
    --model SLAPS \
    --input_dim 5 \
    --hidden_dim 10 \
    --output_dim 5 \
    --is_discrete True \
    --r 0.25 \
    --eta 0.4 \
    --noise_type "0" \
    --generator "MLP" \
    --gen_input_dim 10 \
    --gen_layers_size 10 \
    --gen_k 10 \
    --use_gpu False 
