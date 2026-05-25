export CUDA_VISIBLE_DEVICES=1

python -u main.py \
    --is_training 1 \
    --train_epochs 2000 \
    --model_id CANCER_TEST \
    --dataset cancer \
    --model SLAPS_MLPD \
    --input_dim 30 \
    --hidden_dim 32 \
    --output_dim 2 \
    --r 5 \
    --eta 5 \
    --noise_type "not_zero" \
    --generator MLP-D \
    --gen_input_dim 30 \
    --gen_layers_size 2 \
    --gen_k 20 \
    --use_gpu True \
    --lr_c 0.01 \
    --lr_DAE 0.01 \
    --dropout_c 0.5 \
    --dropout_DAE 0.5 \
    --lambda_val 0.1 \
    --itr 10 \
    --patience 15
